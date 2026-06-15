"""
Decision / state-machine layer for SpeedTrials2D.

AgentState tracks everything that persists across control cycles (speed
multipliers, active timed effects, rear-event flags, light state, ...).
decide() is the pure brain: given this cycle's perception outputs and the
current AgentState, it returns (steering, acceleration) and mutates the state
in place (expiring timers, applying token effects, etc).

Callers are responsible for taking data_lock around state reads/mutations —
this module does no locking itself so it stays testable in isolation.
"""

import time
from dataclasses import dataclass, field

import perception

# ---------------------------------------------------------
# Tunable policy constants — calibrate against live sim behaviour
# ---------------------------------------------------------
POLICY_CONFIG = {
    'yellow_effect_duration': 5.0,       # seconds, per the lab rules

    'green_boost': 0.10,
    'green_boost_light_on': 0.05,
    'red_penalty': 0.20,

    'collision_penalty': 0.50,
    'collision_penalty_duration': 3.0,

    # Token "collection" proxy: a token whose distance_estimate exceeds this
    # is considered close enough that the car is about to drive over it
    'token_collection_distance': 600.0,

    # Rear-event "very close" thresholds used as collision/ignored-event proxies
    'rear_very_close_distance': 4000.0,
    'rear_close_streak_required': 5,     # consecutive close detections before triggering

    # Front-obstacle thresholds — mirror the rear ones but for forward
    # collision risk (the more urgent threat: ramming a car ahead).
    # STILL NEEDS calibration against the live sim.
    'front_obstacle_evade_distance': 3500.0,    # triggers evasive lane change
    'front_obstacle_brake_distance': 5000.0,    # triggers braking fallback when boxed in
    'front_obstacle_brake_acceleration': 0.0,   # coast (not reverse) fallback

    # --- Discrete lane-change tap parameters ---
    # Per the lab PDF, steering is TAPPED (brief pulse to +-1.0, then back to
    # 0.0) to shift exactly one lane — not held proportionally. The actual
    # pulse is timed in sample_drive.send_controls_task (it runs at 200Hz,
    # giving ~5ms timing precision vs. this module's ~30Hz decision cadence).
    # STILL NEEDS calibration: start the duration high (overshoot — moving
    # >1 lane — is obvious; undershoot looks like "nothing happened").
    'lane_pulse_duration': 0.12,         # seconds steering is held at +-1.0
    'lane_change_settle_time': 0.5,      # seconds to wait after a tap before requesting another

    'min_acceleration': 0.3,             # never crawl to a stop chasing tokens
}


@dataclass
class AgentState:
    # --- speed model ---
    base_speed_factor: float = 1.0
    light_on: bool = False

    # --- timed effects (yellow token outcomes) ---
    hide_next_token_type: bool = False
    tokens_invisible_until: float = 0.0
    camera_delay_until: float = 0.0
    control_delay_until: float = 0.0
    camera_corruption_until: float = 0.0

    # --- rear-event flags ---
    faster_car_behind: bool = False
    faster_car_lane: int = None
    police_behind: bool = False
    must_collect_red: bool = False
    collision_penalty_until: float = 0.0

    # --- temporal tracking for the rear-event "closing distance" heuristic ---
    last_rear_distance: float = None
    rear_close_streak: int = 0

    # --- low-brightness / headlight mode ---
    low_brightness: bool = False

    # --- token bookkeeping (for yellow-effect symptom detection) ---
    last_token_seen_at: float = field(default_factory=time.time)
    last_token_count: int = 0

    # --- discrete lane tracking & tap-request state ---
    # current_lane is fused from two sources: perception's debounced
    # lane_index (trusted as ground truth once confident — see
    # _update_current_lane) and an optimistic command-based update applied
    # the instant a tap completes (written by sample_drive.send_controls_task,
    # under data_lock, for low-latency belief ahead of the next confident
    # vision reading).
    current_lane: int = None
    lane_confidence: float = 0.0
    num_lanes: int = field(default_factory=lambda: perception.CONFIG['num_lanes'])

    # Set by decide() (the "what should we do" layer); consumed and cleared by
    # send_controls_task's tap pulse generator (the "how do we do it, precisely
    # timed" layer — it runs at 200Hz vs. decide()'s ~30Hz). lane_change_request
    # is the trigger; lane_change_in_progress / lane_change_settle_until are the
    # cross-task signals that block decide() from issuing overlapping requests.
    lane_change_request: str = None       # 'left' | 'right' | None
    lane_change_in_progress: bool = False
    lane_change_settle_until: float = 0.0

    # --- cached outputs for the fast control loop ---
    # NOTE: target_steering is owned by the tap pulse generator in
    # send_controls_task once a lane change is requested — decide() must NOT
    # write it directly (see update_effective_acceleration).
    target_steering: float = 0.0
    target_acceleration: float = 1.0

    run_start_time: float = field(default_factory=time.time)

    def effective_speed_multiplier(self, now):
        mult = self.base_speed_factor
        if now < self.collision_penalty_until:
            mult *= (1.0 - POLICY_CONFIG['collision_penalty'])
        return max(0.0, mult)


def update_effective_acceleration(state, acceleration):
    """
    Clamps and writes target_acceleration only.

    target_steering is intentionally NOT touched here: once a lane change is
    requested, send_controls_task's tap pulse generator owns it moment-to-
    moment (driving brief +-1.0 pulses on its own 200Hz wall-clock timer).
    Overwriting it from here on every ~33ms decide() cycle would stomp an
    in-flight pulse mid-tap.
    """
    state.target_acceleration = max(-1.0, min(1.0, acceleration))


def try_toggle_light(state):
    """
    Stub for the "turn the light ON" rule. The control protocol only exposes
    (steering, acceleration) floats — there's no obvious dedicated "light"
    channel. This is the #1 open question flagged in the plan: observe whether
    the light toggles automatically server-side, or requires a specific
    gesture / extra protocol message, then fill this in.

    Default assumption (no-op): treat brightness/light as an environmental
    state we only *observe* (it affects how we value green tokens), not one
    we can actively control.
    """
    pass


def _expire_timed_effects(state, now):
    if state.tokens_invisible_until and now >= state.tokens_invisible_until:
        state.tokens_invisible_until = 0.0
    if state.camera_delay_until and now >= state.camera_delay_until:
        state.camera_delay_until = 0.0
    if state.control_delay_until and now >= state.control_delay_until:
        state.control_delay_until = 0.0
    if state.camera_corruption_until and now >= state.camera_corruption_until:
        state.camera_corruption_until = 0.0
    if state.collision_penalty_until and now >= state.collision_penalty_until:
        state.collision_penalty_until = 0.0


def _update_brightness(state, brightness, cfg_threshold):
    state.low_brightness = brightness < cfg_threshold
    if state.low_brightness and not state.light_on:
        try_toggle_light(state)


def _update_current_lane(state, lane_index, confidence, now):
    """
    Fuses the vision-based lane reading into state.current_lane.

    Vision (perception's debounced lane_index) is ground truth — trust it
    whenever it's confidently known. The optimistic command-based update
    (applied by send_controls_task the instant a tap completes, under
    data_lock) only fills the gap until the next confident vision reading
    arrives; this just reconciles the two by always preferring vision once
    it's available.
    """
    state.lane_confidence = confidence
    if lane_index is not None:
        state.current_lane = lane_index


def _direction_to(state, target_lane):
    """Returns 'left'/'right'/None — the single-step direction from
    state.current_lane toward target_lane (None if already there or unknown)."""
    if state.current_lane is None or target_lane is None or target_lane == state.current_lane:
        return None
    return 'right' if target_lane > state.current_lane else 'left'


def _request_lane_change(state, direction, now):
    """
    Records a one-lane-shift request for the tap pulse generator (in
    send_controls_task) to execute, IF AND ONLY IF:
      - no tap is currently in flight or settling (prevents overlapping/
        spammed pulses, which per the lab PDF's "tap" example would
        overshoot by more than one lane), and
      - we have a confident fix on our current lane (don't guess blind), and
      - the resulting lane stays within [0, num_lanes-1] (refuse moves that
        would drive off the edge of the road).
    No-ops silently otherwise — this is the "blocking" behaviour.
    """
    if direction is None:
        return
    if state.lane_change_request is not None or state.lane_change_in_progress:
        return
    if now < state.lane_change_settle_until:
        return
    if state.current_lane is None:
        return

    target = state.current_lane + (1 if direction == 'right' else -1)
    if not (0 <= target < state.num_lanes):
        return

    state.lane_change_request = direction


def _handle_front_obstacles(state, obstacles, now):
    """
    Returns (direction: 'left'|'right'|None, should_brake: bool).

    If an obstacle in state.current_lane is at or beyond
    front_obstacle_evade_distance, tries to step into whichever neighboring
    lane (current_lane -1 then +1, preferring left) doesn't itself have an
    obstacle at or beyond that same threshold. If both neighbors are blocked
    (or off-road) or current_lane is unknown, no direction is returned;
    should_brake is True only in that boxed-in case, and only if the
    obstacle ahead is close enough to cross front_obstacle_brake_distance.

    Like _handle_rear_events/_handle_tokens, this does not call
    _request_lane_change directly — decide() owns request-issuing so it can
    apply priority ordering across all evasion sources.
    """
    if not obstacles or state.current_lane is None:
        return None, False

    # Closest obstacle per lane (obstacles is sorted closest-first already).
    closest_by_lane = {}
    for obs in obstacles:
        if obs.lane not in closest_by_lane:
            closest_by_lane[obs.lane] = obs

    ahead = closest_by_lane.get(state.current_lane)
    if ahead is None or ahead.distance_estimate < POLICY_CONFIG['front_obstacle_evade_distance']:
        return None, False

    candidates = []
    if state.current_lane > 0:
        candidates.append(state.current_lane - 1)
    if state.current_lane < state.num_lanes - 1:
        candidates.append(state.current_lane + 1)

    for target in candidates:
        blocker = closest_by_lane.get(target)
        if blocker is None or blocker.distance_estimate < POLICY_CONFIG['front_obstacle_evade_distance']:
            return _direction_to(state, target), False

    should_brake = ahead.distance_estimate >= POLICY_CONFIG['front_obstacle_brake_distance']
    return None, should_brake


def _handle_rear_events(state, rear, now):
    """
    Returns a discrete evasion direction ('left'|'right'|None). Mutates state
    for flags/penalties exactly as before — only the final "turn this into
    steering" step changes (continuous bias -> discrete one-lane request).
    """
    state.faster_car_behind = bool(rear and rear.faster_car)
    state.faster_car_lane = rear.lane if (rear and rear.faster_car) else None
    state.police_behind = bool(rear and rear.police_car)
    if state.police_behind:
        state.must_collect_red = True

    if rear is None or rear.distance_estimate is None:
        state.last_rear_distance = None
        state.rear_close_streak = 0
        return None

    is_closing = (state.last_rear_distance is not None
                  and rear.distance_estimate > state.last_rear_distance)
    is_very_close = rear.distance_estimate >= POLICY_CONFIG['rear_very_close_distance']
    state.rear_close_streak = state.rear_close_streak + 1 if (is_closing and is_very_close) else 0
    state.last_rear_distance = rear.distance_estimate

    if state.rear_close_streak >= POLICY_CONFIG['rear_close_streak_required']:
        # Heuristic collision/ignored-event proxy — see plan's "known approximation" note.
        state.collision_penalty_until = now + POLICY_CONFIG['collision_penalty_duration']
        state.rear_close_streak = 0

    if rear.faster_car and state.current_lane is not None and rear.lane == state.current_lane:
        # It's bearing down on us in our own lane — step toward whichever
        # neighbouring lane is available, preferring left, falling back to
        # right when we're already in the leftmost lane.
        target = state.current_lane - 1 if state.current_lane > 0 else state.current_lane + 1
        return _direction_to(state, target)

    return None


def _handle_tokens(state, tokens, now):
    """Returns (direction: 'left'|'right'|None, just_collected_color)."""
    # Symptom-based yellow-effect detection: if tokens we expected to still be
    # visible vanish abruptly, infer the "tokens become invisible" effect.
    if tokens:
        state.last_token_seen_at = now
    elif (now - state.last_token_seen_at < 1.0 and state.last_token_count > 0
          and not state.tokens_invisible_until):
        state.tokens_invisible_until = now + POLICY_CONFIG['yellow_effect_duration']
    state.last_token_count = len(tokens)

    if not tokens:
        return None, None

    # Pick which token to chase: prioritize red if police is on our tail,
    # otherwise prioritize green; treat yellow and 'hidden' (the silver badge
    # shown when a token's type is concealed) as neutral/avoid, since their
    # real effect is unknowable in advance and yellow's outcomes skew negative.
    def priority(t):
        if state.must_collect_red and t.color == 'red':
            return 0
        if t.color == 'green':
            return 1
        if t.color == 'red':
            return 2
        return 3  # yellow / hidden / unknown

    target = min(tokens, key=lambda t: (priority(t), -t.distance_estimate))

    direction = _direction_to(state, target.lane)

    collected_color = None
    if target.lane == state.current_lane and target.distance_estimate >= POLICY_CONFIG['token_collection_distance']:
        collected_color = target.color

    return direction, collected_color


def _apply_collection_effect(state, color, now):
    if color == 'green':
        boost = POLICY_CONFIG['green_boost_light_on'] if state.light_on else POLICY_CONFIG['green_boost']
        state.base_speed_factor *= (1.0 + boost)
    elif color == 'red':
        state.base_speed_factor *= (1.0 - POLICY_CONFIG['red_penalty'])
        state.must_collect_red = False
        state.police_behind = False
    elif color == 'yellow':
        # The simulator applies one of 5 random effects server-side; we can't
        # know which one client-side. Approximate with the fixed 5s duration
        # from the rules and let the symptom-based checks above (and future
        # observation) refine which timer actually matters.
        state.hide_next_token_type = True
        if not state.camera_corruption_until:
            state.camera_corruption_until = now + POLICY_CONFIG['yellow_effect_duration']


def decide(front_detections, rear_detection, brightness, state, now):
    """
    front_detections: {'lane_index': int|None, 'lane_confidence': float,
                       'lane_offset': float, 'tokens': list[TokenDetection],
                       'obstacles': list[ObstacleDetection]}
    rear_detection: RearDetection or None
    brightness: float (mean V channel, 0-255)
    state: AgentState — mutated in place
    now: float (time.time())

    Returns acceleration_input, clipped to [-1, 1].

    Steering is NOT returned/owned here: per the lab PDF, lane changes are
    executed as brief steering "taps", precisely timed by send_controls_task's
    200Hz loop. This function only ever *requests* a one-lane shift (via
    state.lane_change_request) — it never writes target_steering directly.
    """
    _expire_timed_effects(state, now)
    _update_brightness(state, brightness, perception.CONFIG['brightness_threshold_v'])
    _update_current_lane(state, front_detections.get('lane_index'),
                         front_detections.get('lane_confidence', 0.0), now)

    # Front-obstacle evasion takes the HIGHEST priority — ramming a car
    # directly ahead at full throttle is the most urgent threat, more so
    # than being rear-ended or missing a token.
    obstacles = front_detections.get('obstacles', [])
    front_direction, should_brake = _handle_front_obstacles(state, obstacles, now)
    _request_lane_change(state, front_direction, now)

    # Rear-event evasion only claims the lane-change slot if front evasion
    # didn't already use it this cycle. Mirrors the existing
    # must_collect_red priority pattern. Only one lane-change request can be
    # in flight, so if evasion claims it, token-seeking waits for next cycle.
    rear_direction = _handle_rear_events(state, rear_detection, now)
    if front_direction is None:
        _request_lane_change(state, rear_direction, now)

    tokens = [] if state.tokens_invisible_until else front_detections.get('tokens', [])
    token_direction, collected_color = _handle_tokens(state, tokens, now)
    if front_direction is None and rear_direction is None:
        _request_lane_change(state, token_direction, now)

    if collected_color is not None:
        _apply_collection_effect(state, collected_color, now)
        if state.hide_next_token_type and collected_color != 'yellow':
            state.hide_next_token_type = False

    acceleration = state.effective_speed_multiplier(now)
    acceleration = max(POLICY_CONFIG['min_acceleration'], min(1.0, acceleration))

    # Braking fallback: when boxed in with an obstacle bearing down and no
    # evasion lane available, override the normal acceleration entirely —
    # this is a deliberate bypass of min_acceleration (whose purpose is to
    # avoid crawling to a stop while chasing tokens, not to prevent slowing
    # down ahead of an imminent collision).
    if should_brake:
        acceleration = POLICY_CONFIG['front_obstacle_brake_acceleration']

    return acceleration
