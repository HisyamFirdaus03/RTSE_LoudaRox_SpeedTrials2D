"""
chasing_car.py
==============
Challenge 2 (Chasing Car) handler for SpeedTrials2D, kept completely separate from
the vision brain (`agent_policy.py`) and the networking / scheduler scaffold
(`sample_drive.py`). Same shape as the Challenge 1 handler in `low_light.py`.

Challenge 2: a chasing car appears BEHIND the player twice during a run (the first
time you get ~10s to react, the second time only ~3s). If it collides with the
player they lose 50% of their speed. The chaser is the distinctive teal/cyan car,
and it shows up in the BACK camera (`:8082`, shared_data['latest_back_frame']).

This component is a thin post-processing override that sits AFTER the policy (and
the low-light handler) in the pipeline:

    (steering, acceleration)  ->  ChasingCarController.apply(back_frame, ...)  ->  (steering, acceleration)

When a teal car is detected closing in from behind, it OVERRIDES only the steering:
it swerves the car aside out of the chaser's path. Acceleration is passed through
untouched, so the upstream policy / low-light handler keeps control of speed. When
no chaser is closing in, both controls pass through untouched, so normal driving /
token-seeking / low-light recovery are unaffected.

The handler is purely reactive (no timers / occurrence counting), so it naturally
covers both the 10s and the 3s appearances.
"""

import cv2
import numpy as np

# =========================================================================
# Tunables -- calibrate on real BACK-camera frames (HSV is OpenCV's: H 0-179)
# =========================================================================
# Teal/cyan body of the chasing car. Measured via colour picker:
# rgb(0, 133, 122) -> OpenCV HSV ~ (88, 255, 133). Band is centred on H=88.
# NOTE: hue 88 actually falls INSIDE the green-TOKEN range (agent_policy.py green
# reaches hue 92), so we CANNOT separate the chaser from a green orb by hue alone.
# We don't try to -- the SIZE gate does it: evasion only fires when the blob is
# >= CLOSE_AREA_FRAC of the frame for ARM_FRAMES straight (a closing CAR), which a
# small green orb never reaches. S/V floors are widened to tolerate shading/edges.
# Re-tune with calibrate.py / auto_calibrate.py on port :8082.
TEAL_LOWER = (84, 80, 50)
TEAL_UPPER = (102, 255, 255)

# Only look at the road BEHIND us -- the lower part of the back frame. Everything
# above this fraction (sky, horizon, distant scenery) is ignored, which kills the
# big cyan-sky blob that used to trip the detector regardless of position.
ROI_TOP_FRAC = 0.45

MIN_AREA_FRAC   = 0.004   # ignore teal blobs smaller than this * (W*H) (noise / far)
CLOSE_AREA_FRAC = 0.03    # blob bigger than this * (W*H) => chaser is "closing in"

ARM_FRAMES   = 3          # require this many consecutive "closing" frames before evading
                          # (so a one-frame false blob can't yank the wheel)
DODGE_STEER  = 1.0        # full-commit lateral steer while evading (steering only;
                          # acceleration is left to the upstream policy)
INVERT_DODGE = False      # flip if the back camera is mirrored (car steers INTO chaser)
HOLD_FRAMES  = 8          # hysteresis: keep evading this many frames after last sighting

DEBUG = True              # draw the "Chaser Debug" overlay (tuning only)


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def detect_chaser(back_frame):
    """Find the largest teal blob in the back-camera frame.
    Returns {cx, cy, area_frac} (pixel centroid + area as a fraction of the frame)
    for the biggest blob above MIN_AREA_FRAC, or None if nothing qualifies."""
    h, w = back_frame.shape[:2]
    hsv = cv2.cvtColor(back_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(TEAL_LOWER, np.uint8), np.array(TEAL_UPPER, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # Restrict to the road region behind us: zero out everything above ROI_TOP_FRAC
    # (sky / horizon / distant scenery) so only an on-road chaser can be detected.
    y0 = int(ROI_TOP_FRAC * h)
    mask[:y0, :] = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, mask

    c = max(cnts, key=cv2.contourArea)
    area_frac = cv2.contourArea(c) / float(w * h)
    if area_frac < MIN_AREA_FRAC:
        return None, mask

    M = cv2.moments(c)
    if M["m00"] == 0:
        return None, mask
    return {
        "cx": M["m10"] / M["m00"],
        "cy": M["m01"] / M["m00"],
        "area_frac": area_frac,
    }, mask


class ChasingCarController:
    """Detects the teal chaser in the back camera and, while it is closing in,
    overrides STEERING ONLY to swerve aside (acceleration passes through).
    Controls pass through untouched whenever no chaser is closing in."""

    def __init__(self):
        self._arm = 0             # consecutive "closing" frames seen so far
        self._hold = 0            # frames of evasion remaining (hysteresis)
        self._last_dodge = DODGE_STEER  # remembered dodge sign if detection flickers
        self.evading = False
        self.chaser = None

    def apply(self, back_frame, steering, acceleration):
        """Take the upstream (steering, acceleration); return possibly-overridden
        controls. While the chaser is closing in: steer away + full throttle."""
        if back_frame is None:
            return steering, acceleration

        w = back_frame.shape[1]
        self.chaser, mask = detect_chaser(back_frame)

        # "Closing in" = a big enough on-road teal blob (proximity). No position
        # OR-clause: a distant/false blob can't trigger evasion on size alone.
        closing = self.chaser is not None and self.chaser["area_frac"] >= CLOSE_AREA_FRAC

        # Arming: only commit to evasion after ARM_FRAMES consecutive sightings, so
        # a one-frame false positive can't yank the wheel.
        self._arm = min(self._arm + 1, ARM_FRAMES) if closing else 0

        # Hysteresis: once armed, hold the dodge for a few frames so a brief dropout
        # doesn't abort an in-progress evade.
        if self._arm >= ARM_FRAMES:
            self._hold = HOLD_FRAMES
        elif self._hold > 0:
            self._hold -= 1
        self.evading = self._hold > 0

        if not self.evading:
            self._maybe_debug(back_frame, mask, steering, "CLEAR")
            return steering, acceleration

        # Evade: steer AWAY from the chaser's lateral position, floor the throttle.
        if self.chaser is not None:
            err = (self.chaser["cx"] - w / 2.0) / (w / 2.0)   # -1 (left) .. +1 (right)
            dodge = -np.sign(err) * DODGE_STEER               # move opposite the chaser
            if dodge == 0:                                    # chaser dead-center: pick a side
                dodge = self._last_dodge
            if INVERT_DODGE:                                  # mirrored back cam
                dodge = -dodge
            self._last_dodge = dodge
        else:
            dodge = self._last_dodge   # lost it mid-dodge: hold the committed direction

        steer_out = _clamp(dodge)
        self._maybe_debug(back_frame, mask, steer_out, "EVADE")
        # Override STEERING only -- dodge aside. Acceleration is passed through so
        # the upstream policy / low-light handler keeps control of speed.
        return steer_out, acceleration

    # -- debug overlay -------------------------------------------------------
    def _maybe_debug(self, frame, mask, steer, mode):
        if not DEBUG:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            # tint what the teal mask caught
            tint = np.zeros_like(vis)
            tint[mask > 0] = (255, 255, 0)   # cyan
            vis = cv2.addWeighted(vis, 0.7, tint, 0.5, 0)
            if self.chaser is not None:
                p = (int(self.chaser["cx"]), int(self.chaser["cy"]))
                cv2.circle(vis, p, 10, (0, 0, 255), 2)
                cv2.putText(vis, f"area={self.chaser['area_frac']:.3f}",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # dodge arrow from bottom-center
            cx = w // 2
            tip = int(cx + steer * w * 0.25)
            color = (0, 0, 255) if mode == "EVADE" else (0, 255, 0)
            cv2.arrowedLine(vis, (cx, h - 5), (tip, h - 40), color, 3, tipLength=0.3)
            cv2.putText(vis, f"{mode}  steer={steer:+.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Chaser Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
