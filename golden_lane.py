"""
golden_lane.py
==============
Challenge (Golden Lane) handler for SpeedTrials2D, kept separate from the vision
brain (`agent_policy.py`) and the scheduler scaffold (`sample_drive.py`), same
shape as `low_light.py` / `chasing_car.py`.

Golden Lane event: one of the road's lanes briefly turns "all green" -- driving in
that lane banks a burst of green. The road has FIVE lanes separated by dotted white
lines. We reuse the SAME drivable-road TRAPEZOID as the token brain
(`agent_policy._roi_polygon`) and split it into 5 lane slices, each mapped to a real
road lane. We then pick the GOLDEN lane = the slice that is most filled with green
(no fragile banner OCR needed -- the "all green" lane is, by definition, the greenest
slice), steer into it and hold while the event lasts.

    (steering, acceleration)  ->  GoldenLaneController.apply(front_frame, ...)  ->  (steering, acceleration)

Override is STEERING ONLY (acceleration passes through). It is LOWER priority than
the chasing-car dodge: in the pipeline it is applied BEFORE `CHASER`, so a rear-car
dodge overrides a golden-lane move. When no golden lane is present, controls pass
through untouched, so normal token-seeking is unaffected.
"""

import cv2
import numpy as np

from agent_policy import CONFIG, _build_color_mask

# =========================================================================
# Tunables
# =========================================================================
N_LANES = 5               # the road has 5 lanes separated by dotted white lines

# --- Banner gate ---------------------------------------------------------
# The golden-lane event is announced by a bright indicator BANNER at the top of the
# screen. We ONLY activate when that banner is present, which stops random green
# clutter on the road from triggering a lane move. BANNER_BOX is a small top-centre
# region (x0, y0, x1, y1 as fractions of W/H) -- calibrate it to sit exactly over
# the banner text on a real 640x480 frame (keep it clear of the corner HUD counters).
BANNER_BOX      = (0.30, 0.02, 0.70, 0.13)
BANNER_VAL_MIN  = 180     # banner text pixels are brighter than this (grayscale 0-255)
BANNER_MIN_FRAC = 0.02    # banner present when > this fraction of the box is bright text

# A lane counts as the "golden" (all-green) lane only when this fraction of its
# slice is green. Kept HIGH so ordinary scattered green tokens during normal driving
# never trip it -- only a whole lane painted green does. Lower if the real golden
# lane isn't fully saturated green; raise if normal driving false-activates.
ACTIVATE_GREEN_FRAC = 0.30

STEER_GAIN = 1.8          # P-gain steering toward the target lane centre
SMOOTH     = 0.4          # low-pass on steering (higher = snappier)
HOLD_FRAMES = 10          # keep holding the lane this many frames after green fades
                          # (bridges flicker; the event lasts well beyond this)

DEBUG = True              # draw the "Golden Lane Debug" overlay (tuning only)


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def _lane_polygons(w, h):
    """Split the agent_policy road TRAPEZOID into N_LANES quadrilateral slices,
    left-to-right. Lane 0 is the leftmost road lane, lane N_LANES-1 the rightmost.
    Returns a list of (4,2) int arrays."""
    cx = w * CONFIG["center_x_frac"]
    top_y = h * CONFIG["roi_top_y"]
    bot_y = h * CONFIG["roi_bottom_y"]
    top_hw = w * CONFIG["roi_top_half_w"]
    bot_hw = w * CONFIG["roi_bot_half_w"]

    polys = []
    for i in range(N_LANES):
        f0 = i / N_LANES          # left edge fraction across the road width
        f1 = (i + 1) / N_LANES    # right edge fraction
        tl = cx - top_hw + f0 * (2 * top_hw)
        tr = cx - top_hw + f1 * (2 * top_hw)
        bl = cx - bot_hw + f0 * (2 * bot_hw)
        br = cx - bot_hw + f1 * (2 * bot_hw)
        polys.append(np.array([[tl, top_y], [tr, top_y],
                               [br, bot_y], [bl, bot_y]], np.int32))
    return polys


def _lane_center_x(w, lane):
    """Target image-x for the centre of `lane` (0-based) at the BOTTOM of the
    trapezoid -- i.e. where the car physically sits."""
    cx = w * CONFIG["center_x_frac"]
    bot_hw = w * CONFIG["roi_bot_half_w"]
    lane_w = (2 * bot_hw) / N_LANES
    return (cx - bot_hw) + (lane + 0.5) * lane_w


def _banner_rect(w, h):
    """Pixel rect (x0, y0, x1, y1) of the banner box for this frame size."""
    fx0, fy0, fx1, fy1 = BANNER_BOX
    return int(fx0 * w), int(fy0 * h), int(fx1 * w), int(fy1 * h)


def detect_banner(front_frame):
    """Return (present, bright_fraction). The golden-lane banner is bright text in a
    small top-centre box; we report it present when enough of that box is bright."""
    h, w = front_frame.shape[:2]
    x0, y0, x1, y1 = _banner_rect(w, h)
    box = front_frame[y0:y1, x0:x1]
    if box.size == 0:
        return False, 0.0
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    bright = int(np.count_nonzero(gray >= BANNER_VAL_MIN))
    frac = bright / float(gray.size)
    return frac >= BANNER_MIN_FRAC, frac


def detect_golden_lane(front_frame):
    """Return (lane_index, fill_fraction, per_lane_fracs) for the greenest lane.
    `lane_index` is 0-based; `fill_fraction` is how much of that lane slice is green.
    The caller decides it's the golden lane only if fill_fraction is high enough."""
    h, w = front_frame.shape[:2]
    hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)
    green = _build_color_mask(hsv, CONFIG["green"])   # same green range as the token brain

    fracs = []
    for poly in _lane_polygons(w, h):
        lane_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(lane_mask, [poly], 255)
        lane_area = int(cv2.countNonZero(lane_mask))
        if lane_area == 0:
            fracs.append(0.0)
            continue
        green_in_lane = int(cv2.countNonZero(cv2.bitwise_and(green, lane_mask)))
        fracs.append(green_in_lane / float(lane_area))

    best = int(np.argmax(fracs))
    return best, fracs[best], fracs


class GoldenLaneController:
    """Detects the all-green ('golden') lane among the 5 road-lane slices of the
    trapezoid and steers into it while the event lasts. Steering only; lower
    priority than the chaser (applied earlier in the pipeline)."""

    def __init__(self):
        self._hold = 0
        self._lane = None          # currently targeted lane (0-based)
        self._prev_steer = 0.0
        self.active = False
        self.fracs = []
        self.banner = False        # is the top banner currently showing?
        self.banner_frac = 0.0

    def apply(self, front_frame, steering, acceleration):
        if front_frame is None:
            return steering, acceleration

        w = front_frame.shape[1]
        self.banner, self.banner_frac = detect_banner(front_frame)
        lane, fill, self.fracs = detect_golden_lane(front_frame)

        # Activate ONLY when the top banner is showing AND a lane is strongly green.
        # The banner gate is what stops random road green from triggering a lane move.
        golden = self.banner and fill >= ACTIVATE_GREEN_FRAC
        if golden:
            self._hold = HOLD_FRAMES
            self._lane = lane
        elif self._hold > 0:
            self._hold -= 1
        self.active = self._hold > 0 and self._lane is not None

        if not self.active:
            self._prev_steer = 0.0
            self._maybe_debug(front_frame, steering, "CLEAR")
            return steering, acceleration

        # Steer toward the centre of the golden lane (P-controller, low-pass smoothed).
        target_x = _lane_center_x(w, self._lane)
        err = (target_x - w * CONFIG["center_x_frac"]) / (w * 0.5)
        steer = _clamp(STEER_GAIN * err)
        steer = SMOOTH * steer + (1 - SMOOTH) * self._prev_steer
        self._prev_steer = steer
        steer = _clamp(steer)

        self._maybe_debug(front_frame, steer, "GOLDEN")
        return steer, acceleration   # steering only; speed left to the policy

    # -- debug overlay -------------------------------------------------------
    def _maybe_debug(self, frame, steer, mode):
        if not DEBUG:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            # banner detection box (cyan when present, grey when not)
            bx0, by0, bx1, by1 = _banner_rect(w, h)
            bcolor = (255, 255, 0) if self.banner else (140, 140, 140)
            cv2.rectangle(vis, (bx0, by0), (bx1, by1), bcolor, 2)
            cv2.putText(vis, f"BANNER {'ON' if self.banner else 'off'} {self.banner_frac:.2f}",
                        (bx0, by0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bcolor, 1)
            for i, poly in enumerate(_lane_polygons(w, h)):
                is_target = (self.active and i == self._lane)
                color = (0, 215, 255) if is_target else (180, 180, 180)  # gold vs grey
                cv2.polylines(vis, [poly], True, color, 2 if is_target else 1)
                frac = self.fracs[i] if i < len(self.fracs) else 0.0
                cx_i = int(np.mean(poly[:, 0]))
                cv2.putText(vis, f"{i+1}:{frac:.2f}", (cx_i - 18, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            color, 2 if is_target else 1)
            cx = int(w * CONFIG["center_x_frac"])
            tip = int(cx + steer * w * 0.25)
            acolor = (0, 215, 255) if mode == "GOLDEN" else (0, 255, 0)
            cv2.arrowedLine(vis, (cx, h - 5), (tip, h - 40), acolor, 3, tipLength=0.3)
            lane_txt = f"lane {self._lane + 1}" if self._lane is not None else "-"
            cv2.putText(vis, f"{mode} {lane_txt} steer={steer:+.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, acolor, 2)
            cv2.imshow("Golden Lane Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
