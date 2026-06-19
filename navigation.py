"""
navigation.py
=============
Decision component: given detected tokens (from token_detection.py), picks the
best green token to chase and steers away from nearby red/yellow hazards.

Steering is edge-triggered in this game: sending +1 (or -1) shifts exactly one
lane, and the game needs to see the input drop back to 0 before it will count
another shift -- holding it at +1 only ever moves one lane. So instead of
holding a locked steer value, we PULSE it: hold +-1 for `pulse_hold_frames`,
release to 0 for `pulse_release_frames`, repeat -- each cycle fires one more
lane shift toward the desired direction, continuously, until the target is
reached. Chatter from flipping direction near zero offset is prevented with
hysteresis on the *decision* (tracked separately from the pulsed output).
"""

import debug_view

CONFIG = {
    "steer_deadzone": 0.05,    # |lateral offset| below this (fraction of half-width) -> no shift needed
    "steer_hysteresis": 0.10,  # extra margin beyond deadzone required to flip an already-locked direction
    "pulse_hold_frames": 2,    # frames to hold steer at +-1 per lane-shift pulse; ASSUMED, tune in-game
    "pulse_release_frames": 2, # frames to hold steer at 0 between pulses, so the game registers the reset
    "hazard_near_y_frac": 0.75,     # hazard counts as "close" below this fraction of H
    "hazard_lateral_band_frac": 0.25,
    "panic_brake_accel": 0.1,       # throttle when a hazard is dead-ahead and close
    "base_acceleration": 0.6,
    "turn_acceleration_cut": 0.3,   # ease off throttle proportional to |steering|
    "debug_overlay": True,
}

_lock_dir = 0.0          # last decided direction (-1/0/1), used for hysteresis on the decision
_pulse_state = "release" # "hold" or "release"
_pulse_count = 0


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _decide_direction(norm_offset, lock_dir, deadzone, hysteresis):
    """Hysteresis direction decision (-1/0/1) for `norm_offset`, tracked against
    the last *decided* direction (not the pulsed output) so it only flips once
    the offset clearly crosses to the other side."""
    if lock_dir > 0 and norm_offset > -(deadzone + hysteresis):
        return 1.0
    if lock_dir < 0 and norm_offset < (deadzone + hysteresis):
        return -1.0
    if abs(norm_offset) < deadzone:
        return 0.0
    return 1.0 if norm_offset > 0 else -1.0


def _pulse(desired_dir):
    """Turn a desired direction into an edge-triggered steer pulse: alternates
    between holding `desired_dir` and releasing to 0 so each cycle registers as
    one lane shift in-game, repeating continuously while `desired_dir` != 0."""
    global _pulse_state, _pulse_count

    if desired_dir == 0.0:
        _pulse_state, _pulse_count = "release", 0
        return 0.0

    if _pulse_state == "hold":
        steer = desired_dir
        _pulse_count += 1
        if _pulse_count >= CONFIG["pulse_hold_frames"]:
            _pulse_state, _pulse_count = "release", 0
    else:  # "release"
        steer = 0.0
        _pulse_count += 1
        if _pulse_count >= CONFIG["pulse_release_frames"]:
            _pulse_state, _pulse_count = "hold", 0
    return steer


def select_best_green_target(green_blobs, w, h):
    if not green_blobs:
        return None
    def score(b):
        cx, cy, _ = b
        return (cy / h) - abs(cx - w / 2.0) / (w / 2.0)
    cx, cy, _ = max(green_blobs, key=score)
    return cx, cy


def find_nearest_hazard(red_blobs, yellow_blobs, w, h):
    near_y = h * CONFIG["hazard_near_y_frac"]
    band = w * CONFIG["hazard_lateral_band_frac"]
    center_x = w / 2.0
    in_way = [b for b in (red_blobs + yellow_blobs)
              if b[1] >= near_y and abs(b[0] - center_x) <= band]
    if not in_way:
        return None
    cx, cy, _ = max(in_way, key=lambda b: b[1])
    return cx, cy


def compute_steering_and_throttle(frame, green_blobs, red_blobs, yellow_blobs,
                                   roi_top, roi_bottom, top_half_width_px, bottom_half_width_px):
    """Decide (steering, acceleration) in [-1,1] from this frame's detections."""
    global _lock_dir

    h, w = frame.shape[:2]
    center_x = w / 2.0

    target = select_best_green_target(green_blobs, w, h)
    hazard = find_nearest_hazard(red_blobs, yellow_blobs, w, h)

    if hazard is not None:
        # Avoidance takes priority over chasing -- no hysteresis (a real hazard
        # must never be allowed to "stick" toward the wrong direction).
        hx, _ = hazard
        desired_dir = 1.0 if hx < center_x else -1.0  # hazard on left -> full right
    elif target is not None:
        tx, _ = target
        norm_offset = (tx - center_x) / (w / 2.0)
        desired_dir = _decide_direction(norm_offset, _lock_dir,
                                         CONFIG["steer_deadzone"], CONFIG["steer_hysteresis"])
    else:
        desired_dir = 0.0  # no target visible -> go straight

    _lock_dir = desired_dir
    steer = _pulse(desired_dir)

    acceleration = CONFIG["base_acceleration"] - CONFIG["turn_acceleration_cut"] * abs(desired_dir)

    if hazard is not None:
        hx, hy = hazard
        dead_ahead = abs(hx - center_x) <= w * (CONFIG["hazard_lateral_band_frac"] / 2.0)
        if dead_ahead and hy >= h * CONFIG["hazard_near_y_frac"]:
            acceleration = CONFIG["panic_brake_accel"]

    acceleration = _clamp(acceleration)

    if CONFIG["debug_overlay"]:
        debug_view.draw_debug_overlay(frame, roi_top, roi_bottom, top_half_width_px, bottom_half_width_px,
                                       green_blobs, red_blobs, yellow_blobs,
                                       target, hazard, steer, acceleration)

    return steer, acceleration
