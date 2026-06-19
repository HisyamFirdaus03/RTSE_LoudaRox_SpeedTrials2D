"""
event_manager.py
================
The arbiter that replaces the old "last .apply() wins" chain in sample_drive.py.

Every frame:
  1. The baseline brain (agent_policy.HeuristicPolicy) proposes (steer, accel).
  2. Each event handler is asked `evaluate(front, back, ctx) -> Override | None`.
  3. The highest-priority active Override wins; otherwise the baseline passes through.

This makes event conflicts explicit (e.g. Darkness wants a full brake while a car
closes from behind -> the collision dodge at priority 100 beats Darkness at 80, so
we don't rear-end; Darkness brakes only once the rear is clear).

`ctx` carries everything a handler needs that isn't in the frame: the game clock,
the latest HUD snapshot (real on-screen counts), the 5-lane model, the baseline
action, and the shared `passed` set used to track the Tactical win condition.

The Tactical win = (green - red) >= 60 AND every event passed at least once, and it
beats raw distance. We can't *score* the game from here, but we track our best
estimate of it to drive endgame behaviour.
"""

import time
from dataclasses import dataclass

import lanes

# ---- game constants (from the game-day brief) ----
GAME_DURATION = 180.0      # seconds
CYCLE = 60.0               # events rotate once per 60s
TACTICAL_NET = 60          # need green - red >= this
ALL_EVENTS = ("darkness", "police", "chasingA", "chasingB", "golden")

# Map the in-game EV1..EV5 status indicators to our event names. When the game
# turns an EV box green, that event is passed -- the authoritative signal.
EV_TO_EVENT = {
    "EV1": "darkness",
    "EV2": "police",
    "EV3": "chasingA",
    "EV4": "chasingB",
    "EV5": "golden",
}

# Per-cycle windows (seconds within a 60s cycle) when each event MAY start. Used
# only as a prior to raise sensitivity / disambiguate the two chasing cars -- every
# event is still confirmed from vision. Verify/adjust against real play.
WINDOWS = {
    "darkness": (0, 30),
    "police":   (30, 50),
    "chasingA": (0, 30),
    "chasingB": (0, 50),
    "golden":   (0, 55),
}


@dataclass
class Override:
    """A handler's bid to take over this frame. Higher priority wins."""
    steering: float
    acceleration: float
    priority: int
    label: str


class Ctx:
    """Per-frame context handed to every handler's evaluate()."""

    def __init__(self, t, hud, base_steer, base_accel, passed, lane_model):
        self.t = t                      # seconds since first frame
        self.cycle_t = t % CYCLE        # phase within the current 60s cycle
        self.hud = hud                  # {green,red,yellow,net,distance} or None
        self.base_steer = base_steer    # baseline brain's proposal this frame
        self.base_accel = base_accel
        self.passed = passed            # shared set, handlers add their event name
        self.lanes = lane_model         # LaneModel already updated with this frame

    def in_window(self, name):
        """True if we're inside the rotation window where `name` may fire."""
        lo, hi = WINDOWS.get(name, (0, CYCLE))
        return lo <= self.cycle_t <= hi


class EventManager:
    """Runs the baseline policy + event handlers and arbitrates between them."""

    def __init__(self, policy, handlers, hud_getter=None):
        """
        policy      : the baseline brain (HeuristicPolicy) -- has .act(front, back).
        handlers    : list of objects exposing evaluate(front, back, ctx)->Override|None.
        hud_getter  : optional callable returning the latest HUD dict (live counts).
        """
        self.policy = policy
        self.handlers = handlers
        self.hud_getter = hud_getter
        self.lane_model = lanes.LaneModel()
        self._t0 = None
        self.passed = set()
        self.tactical_locked = False
        self._prev_red = None
        self._last_label = "POLICY"

    def _elapsed(self):
        if self._t0 is None:
            self._t0 = time.time()
        return time.time() - self._t0

    def act(self, front_frame, back_frame=None):
        """Return (steering, acceleration) for this frame."""
        base_steer, base_accel = self.policy.act(front_frame, back_frame)
        if front_frame is None:
            return base_steer, base_accel

        hud = self.hud_getter() if self.hud_getter else None
        self.lane_model.update(front_frame)
        ctx = Ctx(self._elapsed(), hud, base_steer, base_accel, self.passed, self.lane_model)

        overrides = []
        for h in self.handlers:
            try:
                o = h.evaluate(front_frame, back_frame, ctx)
            except Exception:
                o = None      # a faulty handler must never crash the control loop
            if o is not None:
                overrides.append(o)

        if overrides:
            chosen = max(overrides, key=lambda o: o.priority)
            steer, accel, self._last_label = chosen.steering, chosen.acceleration, chosen.label
        else:
            steer, accel, self._last_label = base_steer, base_accel, "POLICY"

        self._update_tactical(ctx)
        return steer, accel

    # -- Tactical-win tracking (best-effort; the game is the real judge) --------
    def _update_tactical(self, ctx):
        hud = ctx.hud
        if hud is not None:
            # Authoritative: the game's own EV1..EV5 indicators turning green.
            events = hud.get("events") or {}
            for ev, name in EV_TO_EVENT.items():
                if events.get(ev) == "green":
                    self.passed.add(name)
            red = hud.get("red")
            # Police is passed when the red count ticks up while a cop is around.
            if red is not None and self._prev_red is not None and red > self._prev_red \
                    and self._last_label.startswith("POLICE"):
                self.passed.add("police")
            if red is not None:
                self._prev_red = red
            net = hud.get("net")
            if net is not None and net >= TACTICAL_NET and len(self.passed) >= len(ALL_EVENTS):
                self.tactical_locked = True

    # -- telemetry --------------------------------------------------------------
    def status(self):
        return {
            "t": round(self._elapsed(), 1),
            "label": self._last_label,
            "passed": sorted(self.passed),
            "tactical_locked": self.tactical_locked,
        }
