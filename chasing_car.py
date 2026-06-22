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
# small green orb never reaches.
#
# SATURATION floor (2nd number, was 80) rejects the PALE white road light
# rgb(221,244,255) -> HSV (100,34,255): in-hue but near-white, so S<120 drops it.
# UPPER HUE (1st number of TEAL_UPPER, was 102) is pulled down to 97 to open a gap
# between the CAR (hue 88) and the bluer road LIGHTS, which cluster at hue 100-112
# (rgb(80,99,113)->103, rgb(36,61,83)->104, rgb(0,6,23)->112). The sampled lights
# were already out, but their saturated GLOW ring sits ~97-102 and leaked in; the
# tighter upper bound excludes that ring while the car at 88 stays well inside.
# Re-tune with calibrate.py / auto_calibrate.py on :8082 if the car's rim shifts.
TEAL_LOWER = (84, 120, 50)
TEAL_UPPER = (97, 255, 255)

# Trim only the very top sliver of the back frame (fixed UI / far sky). Kept SMALL
# (was 0.45) on purpose: on a DOWNHILL the road behind tilts upward and the chaser
# rides HIGH in the frame -- a big top-crop would hide it until it's right on our
# bumper (too late). Sky is now rejected by SIZE (MAX_AREA_FRAC) instead of by
# position, so we can afford to search almost the whole frame.
ROI_TOP_FRAC = 0.20      # lowered from 0.35 to EVADE EARLIER: a far chaser sits high
                         # near the horizon, so searching more of the frame catches it
                         # sooner. The MAX_AREA_FRAC size cap still rejects the sky.

MIN_AREA_FRAC   = 0.002   # ignore teal blobs smaller than this * (W*H) (noise / far).
                          # Lowered from 0.004 to react to a SMALLER (farther) chaser,
                          # i.e. evade earlier. Raise back toward 0.004 if distant
                          # noise that passes the colour/shape filters triggers it.
MAX_AREA_FRAC   = 0.30    # ignore blobs BIGGER than this * (W*H): a teal sky/background
                          # region covers far more of the frame than any car, so size
                          # rejects it no matter where it sits (works on hills, unlike
                          # a position-based crop). Lower toward the car's max on-screen
                          # size if a large cyan background still leaks through.
CLOSE_AREA_FRAC = 0.002   # blob bigger than this * (W*H) => chaser is "closing in".
                          # Set EQUAL to MIN_AREA_FRAC so that as soon as a car is
                          # DETECTED (which is already tightly filtered by hue, sat,
                          # value, size and aspect), it evades -- no waiting for it to
                          # grow. RAISE this above MIN_AREA_FRAC if you only want to
                          # dodge once the chaser is closer (read the "area=" value in
                          # the Chaser Debug window and set it just below that).

MIN_ASPECT = 0.55         # SHAPE gate: reject tall-thin blobs (width/height below this).
                          # Road lights are VERY THIN VERTICAL strips (aspect ~0.1-0.3);
                          # the chaser is a TRAPEZIUM, roughly as wide as tall (aspect
                          # ~0.8-1.5). So even if a light's colour sneaks into the band,
                          # its shape excludes it. Lower toward 0.3 if the car is seen
                          # very narrow (far away / partly off-frame).

ARM_FRAMES   = 2          # require this many consecutive "closing" frames before evading
                          # (so a one-frame false blob can't yank the wheel)
DODGE_STEER  = 1.0        # full-commit lateral steer while evading (steering only;
                          # acceleration is left to the upstream policy)
INVERT_DODGE = True      # flip if the back camera is mirrored (car steers INTO chaser)
HOLD_FRAMES  = 8          # hysteresis: keep evading this many frames after last sighting

DEBUG = True              # draw the "Chaser Debug" overlay (tuning only)


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def detect_chaser(back_frame):
    """Find the teal chaser in the back-camera frame.
    Returns {cx, cy, area_frac} (centroid + size as a fraction of the frame), or
    None if nothing qualifies.

    The chaser is a BLACK car with a teal OUTLINE, so the HSV band only matches its
    thin glowing rim -- the raw matched area is tiny even when the car is big and
    close. We therefore (1) DILATE the mask to bridge the rim fragments into one
    blob, and (2) measure the largest blob's CONVEX HULL, so 'area_frac' reflects
    the whole car's spread, not just the lit edge. A small coin's hull stays small,
    so the size gate still tells a car apart from a coin."""
    h, w = back_frame.shape[:2]
    hsv = cv2.cvtColor(back_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(TEAL_LOWER, np.uint8), np.array(TEAL_UPPER, np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # bridge the (outline-only) teal rim into a single connected blob
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))

    # Restrict to the road region behind us: zero out everything above ROI_TOP_FRAC
    # (sky / horizon / distant scenery) so only an on-road chaser can be detected.
    y0 = int(ROI_TOP_FRAC * h)
    mask[:y0, :] = 0

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Pick the largest CAR-SIZED blob: hull area within [MIN_AREA_FRAC, MAX_AREA_FRAC).
    # Skipping anything >= MAX_AREA_FRAC drops a teal sky/background region (too big to
    # be a car) regardless of its vertical position -- so detection still works on
    # hills, where the chaser can ride high in the frame.
    best_hull = None
    best_area = 0.0
    for c in cnts:
        hull = cv2.convexHull(c)                   # encloses the whole car outline
        area_frac = cv2.contourArea(hull) / float(w * h)
        if area_frac < MIN_AREA_FRAC or area_frac >= MAX_AREA_FRAC:
            continue
        # SHAPE gate: drop tall-thin blobs (road lights are thin vertical strips;
        # the chaser is a wide trapezium). aspect = width / height of the box.
        _, _, bw, bh = cv2.boundingRect(hull)
        if bh == 0 or (bw / float(bh)) < MIN_ASPECT:
            continue
        if area_frac > best_area:
            best_area = area_frac
            best_hull = hull

    if best_hull is None:
        return None, mask

    M = cv2.moments(best_hull)
    if M["m00"] == 0:
        return None, mask
    return {
        "cx": M["m10"] / M["m00"],
        "cy": M["m01"] / M["m00"],
        "area_frac": best_area,
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

    def reset(self):
        """Clear per-episode state (called between runs)."""
        self._arm = 0
        self._hold = 0
        self._last_dodge = DODGE_STEER
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

            # DIAGNOSTIC: raw teal mask on the FULL frame (before the ROI crop), so
            # you can see whether the HSV band matches the car at all. If the car
            # lights up here but NOT in the main window, the ROI line is cropping it.
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            raw = cv2.inRange(hsv, np.array(TEAL_LOWER, np.uint8), np.array(TEAL_UPPER, np.uint8))
            cv2.imshow("Chaser Mask (raw band)", raw)

            # tint what the teal mask caught (after ROI crop -- what detection sees)
            tint = np.zeros_like(vis)
            tint[mask > 0] = (255, 255, 0)   # cyan
            vis = cv2.addWeighted(vis, 0.7, tint, 0.5, 0)

            # ROI cut line: detection ignores everything ABOVE this line.
            y0 = int(ROI_TOP_FRAC * h)
            cv2.line(vis, (0, y0), (w, y0), (0, 165, 255), 1)
            cv2.putText(vis, "ROI: detect below this line", (10, y0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

            if self.chaser is not None:
                p = (int(self.chaser["cx"]), int(self.chaser["cy"]))
                cv2.circle(vis, p, 10, (0, 0, 255), 2)
                cv2.putText(vis, f"area={self.chaser['area_frac']:.3f}",
                            (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # DIAGNOSTIC: nothing was accepted. Show the largest blob the processed
                # mask DID contain and WHY it was rejected, so we can see which gate is
                # filtering the car out (size cap, aspect, or ROI). Drawn in orange.
                dcnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if dcnts:
                    c = max(dcnts, key=cv2.contourArea)
                    hull = cv2.convexHull(c)
                    af = cv2.contourArea(hull) / float(w * h)
                    x, y, bw, bh = cv2.boundingRect(hull)
                    asp = bw / float(bh) if bh else 0.0
                    if af < MIN_AREA_FRAC:
                        why = "area<MIN"
                    elif af >= MAX_AREA_FRAC:
                        why = "area>MAX"
                    elif asp < MIN_ASPECT:
                        why = "aspect<MIN"
                    else:
                        why = "(passes? ROI?)"
                    cv2.drawContours(vis, [hull], -1, (0, 165, 255), 2)
                    cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 165, 255), 1)
                    cv2.putText(vis, f"REJECTED {why}  area={af:.3f} asp={asp:.2f}",
                                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
            # dodge arrow from bottom-center. The back camera faces the OPPOSITE
            # way to the front camera, so we flip the arrow horizontally (cx - steer)
            # to match the Agent Debug arrow's on-screen direction for the same
            # steering command. Display-only -- the value sent to the car is unchanged.
            cx = w // 2
            tip = int(cx - steer * w * 0.25)
            color = (0, 0, 255) if mode == "EVADE" else (0, 255, 0)
            cv2.arrowedLine(vis, (cx, h - 5), (tip, h - 40), color, 3, tipLength=0.3)
            cv2.putText(vis, f"{mode}  steer={steer:+.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Chaser Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
