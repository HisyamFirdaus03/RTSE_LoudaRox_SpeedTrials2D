"""
chasing_car.py
==============
Challenge EV3 / EV4 (Chasing Car A & B) handler. The chasing cars approach from
BEHIND, so this reads the BACK camera (`back_frame`, previously unused). Colliding
with a chasing car is to be avoided; we evade by changing lanes on the front
steering and (when it's nearly on us) flooring the throttle to pull away.

Detection is intentionally swappable because we don't yet have the sprite captured
from real play:
  - If `chase_template.png` exists, use template matching (most robust).
  - Otherwise fall back to the largest vivid car-sized blob in a configurable
    hue band (CHASE["color"]) inside a rear ROI that excludes our own car.

"Closing" = the detection's bbox area is large and/or growing across frames.

IMPORTANT verify-in-game items (flagged, not assumed):
  * the back camera's horizontal axis may be MIRRORED relative to front steering
    (rear-view style). CHASE["mirror_x"] flips the evade direction; confirm the
    sign on a real frame.
  * capture the real sprite/colour and set CHASE["color"] / chase_template.png.

The Tactical "pass" for each chasing car is latched when a tracked car has come
and gone without us being hit (we survived its window).
"""

import os
import cv2
import numpy as np

from event_manager import Override
import lanes

CHASE_TEMPLATE = "chase_template.png"

CFG = {
    # ---- detection ----
    # The chasing car is the teal "ghost" DeLorean (asset/36254_ghost*.png). Its
    # body is a tight, extremely saturated TEAL cluster measured off the sprites:
    # hue ~86-90, S ~250-255, V varies with shading. Band widened to 82-102 / S>=120
    # to survive camera JPEG, while staying clear of grass (hue 40-70), green tokens
    # (<~83), and the blue headlights / cop lightbar (>=100). H 0-179, S/V 0-255.
    "color": [((82, 120, 50), (102, 255, 255))],
    "roi_top_frac":    0.10,   # rear ROI: skip the sky/horizon
    "roi_bottom_frac": 0.82,   # ...and the player's own car at the very bottom
    "min_area_frac":   0.004,  # ignore blobs smaller than this * (W*H)
    "min_ar": 0.4, "max_ar": 3.0,   # car bbox aspect gate

    # ---- closing thresholds (area as a fraction of the back frame) ----
    "imminent_area_frac": 0.06,  # this big behind us -> imminent contact (pri 100)
    "warn_area_frac":     0.02,  # visible & notable -> early evade (pri 40)
    "history": 5,                # frames of area history to judge "growing"

    # ---- evasion ----
    "mirror_x":   True,   # back cam is mirrored relative to front steering (VERIFY)
    "steer_gain": 2.2,
    "accel_evade":    1.0,   # imminent: floor it to pull away while changing lane
    "accel_warn":     0.9,   # early evade: keep moving, change lane
    "lost_frames":    8,     # car considered gone after this many misses
    "debug": True,
}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class ChasingCarController:
    """Evades chasing cars detected on the back camera by lane-changing on the
    front steering. Tracks up to the two cars (A then B) for Tactical passing."""

    def __init__(self, config=None):
        self.cfg = dict(CFG)
        if config:
            self.cfg.update(config)
        self.template = None
        if os.path.exists(CHASE_TEMPLATE):
            self.template = cv2.imread(CHASE_TEMPLATE, cv2.IMREAD_COLOR)
        self._area_hist = []
        self._miss = 0
        self._tracking = False     # a car is currently on the back cam
        self._chase_seen = 0       # how many distinct chasing cars have come & gone

    # -- arbiter interface ----------------------------------------------
    def evaluate(self, front_frame, back_frame, ctx):
        if back_frame is None:
            self._on_miss(ctx)
            return None

        h, w = back_frame.shape[:2]
        det = self._detect(back_frame, w, h)

        if det is None:
            self._on_miss(ctx)
            self._maybe_debug(back_frame, None, "CLEAR")
            return None

        cx, area_frac = det
        self._miss = 0
        self._tracking = True
        self._area_hist.append(area_frac)
        self._area_hist = self._area_hist[-self.cfg["history"]:]
        growing = len(self._area_hist) >= 2 and self._area_hist[-1] > self._area_hist[0]

        # Which side is the car on? (front-steering orientation; flip if the back
        # cam is mirrored). Then evade two lanes toward the open side, relative to
        # the lane we're actually in (from the live LaneModel).
        car_x = (w - cx) if self.cfg["mirror_x"] else cx
        cur = ctx.lanes.current_lane()
        if car_x < w / 2:                       # car behind-left -> move right
            evade_lane = min(lanes.N_LANES, cur + 2)
        else:                                   # car behind-right -> move left
            evade_lane = max(1, cur - 2)

        if area_frac >= self.cfg["imminent_area_frac"]:
            steer = ctx.lanes.steer_to_lane(evade_lane, self.cfg["steer_gain"])
            self._maybe_debug(back_frame, det, f"IMMINENT lane{cur}->{evade_lane}")
            return Override(_clamp(steer), self.cfg["accel_evade"], 100, "CHASE-DODGE")

        if area_frac >= self.cfg["warn_area_frac"] and growing:
            steer = ctx.lanes.steer_to_lane(evade_lane, self.cfg["steer_gain"])
            self._maybe_debug(back_frame, det, f"WARN lane{cur}->{evade_lane}")
            return Override(_clamp(steer), self.cfg["accel_warn"], 40, "CHASE-EVADE")

        # Seen but far/not closing: let the baseline drive, keep watching.
        self._maybe_debug(back_frame, det, "WATCH")
        return None

    # -- helpers --------------------------------------------------------
    def _on_miss(self, ctx):
        """A frame with no car. If we were tracking one and it's now gone for good,
        count it as survived and latch the Tactical pass for the next chasing slot."""
        if not self._tracking:
            return
        self._miss += 1
        if self._miss >= self.cfg["lost_frames"]:
            self._tracking = False
            self._area_hist = []
            self._chase_seen += 1
            ctx.passed.add("chasingA" if self._chase_seen == 1 else "chasingB")

    def _detect(self, frame, w, h):
        """Return (cx, area_frac) of the chasing car, or None."""
        if self.template is not None:
            return self._detect_template(frame, w, h)
        return self._detect_color(frame, w, h)

    def _detect_color(self, frame, w, h):
        cfg = self.cfg
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in cfg["color"]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        roi = np.zeros((h, w), np.uint8)
        roi[int(h * cfg["roi_top_frac"]):int(h * cfg["roi_bottom_frac"]), :] = 255
        mask = cv2.bitwise_and(mask, roi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = cfg["min_area_frac"] * (w * h)
        best, best_area = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area <= best_area:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if bh == 0:
                continue
            ar = bw / bh
            if ar < cfg["min_ar"] or ar > cfg["max_ar"]:
                continue
            best, best_area = c, area
        if best is None:
            return None
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        return M["m10"] / M["m00"], best_area / (w * h)

    def _detect_template(self, frame, w, h):
        cfg = self.cfg
        res = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < 0.55:                      # tune the match threshold in-game
            return None
        th, tw = self.template.shape[:2]
        cx = maxloc[0] + tw / 2.0
        return cx, (tw * th) / (w * h)

    def _maybe_debug(self, frame, det, mode):
        if not self.cfg["debug"]:
            return
        try:
            vis = frame.copy()
            if det is not None:
                cx, af = det
                cv2.line(vis, (int(cx), 0), (int(cx), vis.shape[0]), (0, 0, 255), 2)
                cv2.putText(vis, f"area={af:.3f}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(vis, f"CHASE:{mode}  seen={self._chase_seen}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Chase Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
