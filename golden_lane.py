"""
golden_lane.py
==============
Challenge EV5 (Golden Lane) handler. A text flashes a lane number (1..5); for the
next 5 seconds every token in that lane is green, and you PASS the event by being
in that lane the moment the 5s timer expires.

Flow:
  1. Within the Golden Lane rotation window, read the flashed lane number by OCR,
     reusing hud.py's digit atlas + the same template-match scoring.
  2. On a confident read, start a 5s hold timer and lock the target lane.
  3. While the timer runs, steer to that lane and HOLD (priority 50). The lane is
     full of green tokens, so this also feeds the greedy/Tactical goal.
  4. When the timer expires, if the car is in the target lane, latch "golden".

Observed in-game: "LANE N -- ALL GREEN! (3s)" flashes in yellow just above the
EV1..EV5 row. flash_roi/hue CFG below are tuned to that; re-verify if the layout
changes.
"""

import time
import cv2
import numpy as np

from event_manager import Override
from hud import load_atlas, _segment_digits, _canon

CFG = {
    # Region (fractions of W,H) to scan for the flashed instruction. Observed: the
    # banner reads "LANE N -- ALL GREEN! (3s)" in yellow, sitting just above the
    # EV1..EV5 row (hud.py's EVENT_BOXES, y=88..110 of 480). Narrowed to just the
    # "LANE N" portion so the "(3s)" countdown digit isn't mistaken for the lane.
    "flash_roi": (0.30, 0.02, 0.80, 0.10),   # (x0,y0,x1,y1)
    "hue_lo": 18,    # yellow text hue band (OpenCV H 0-179), matches hud.py's
    "hue_hi": 38,    # COUNTER_HSV["yellow"] band
    "sat_min": 100,
    "val_min": 100,
    "match_min_score": 0.80,   # template-match confidence floor to trust a digit
    "hold_seconds": 5.0,    # the event's own 5s timer
    "cooldown_seconds": 8.0,   # ignore re-triggers for this long after a hold ends
    "steer_gain": 2.4,      # a touch snappier than cruise -> commit to the lane
    "accel_hold": 0.9,      # keep moving to sweep up the lane's green tokens
    "debug": True,
}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class GoldenLaneController:
    """Reads the flashed lane number and holds that lane until the 5s timer expires."""

    def __init__(self, config=None):
        self.cfg = dict(CFG)
        if config:
            self.cfg.update(config)
        self.atlas = load_atlas()
        self.target_lane = None
        self._timer_end = 0.0
        self._cooldown_until = 0.0
        self._last_mode = "IDLE"

    # -- arbiter interface ----------------------------------------------
    def evaluate(self, front_frame, back_frame, ctx):
        if front_frame is None:
            return None
        h, w = front_frame.shape[:2]
        now = time.time()

        # Active hold: steer to the locked lane until the 5s timer expires.
        if self.target_lane is not None and now < self._timer_end:
            steer = ctx.lanes.steer_to_lane(self.target_lane, self.cfg["steer_gain"])
            self._last_mode = f"HOLD lane{self.target_lane} ({self._timer_end-now:.1f}s)"
            self._maybe_debug(front_frame, self._last_mode)
            return Override(_clamp(steer), self.cfg["accel_hold"], 50, "GOLDEN")

        # Timer just expired: judge the pass, then start cooldown.
        if self.target_lane is not None and now >= self._timer_end:
            if ctx.lanes.current_lane() == self.target_lane:
                ctx.passed.add("golden")
            self.target_lane = None
            self._cooldown_until = now + self.cfg["cooldown_seconds"]

        # Idle: look for a new announcement (only inside its rotation window).
        if now >= self._cooldown_until and ctx.in_window("golden"):
            n = self._read_lane_number(front_frame, w, h)
            if n is not None:
                self.target_lane = n
                self._timer_end = now + self.cfg["hold_seconds"]
                self._last_mode = f"LOCK lane{n}"
                self._maybe_debug(front_frame, self._last_mode)
                steer = ctx.lanes.steer_to_lane(n, self.cfg["steer_gain"])
                return Override(_clamp(steer), self.cfg["accel_hold"], 50, "GOLDEN")

        self._maybe_debug(front_frame, "IDLE")
        return None

    # -- OCR ------------------------------------------------------------
    def _read_lane_number(self, frame, w, h):
        """Return a lane number 1..5 read from the flash text, or None.
        Picks the single digit-component with the best atlas match in range."""
        if not self.atlas:
            return None
        x0, y0, x1, y1 = self.cfg["flash_roi"]
        crop = frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
        if crop.size == 0:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           np.array((self.cfg["hue_lo"], self.cfg["sat_min"], self.cfg["val_min"]), np.uint8),
                           np.array((self.cfg["hue_hi"], 255, 255), np.uint8))
        best_d, best_score = None, self.cfg["match_min_score"]
        for _, dcrop in _segment_digits(mask):
            d, score = self._match(dcrop)
            if d is not None and 1 <= d <= 5 and score > best_score:
                best_d, best_score = d, score
        return best_d

    def _match(self, crop):
        """(digit, score) of the best atlas template (normalized correlation)."""
        c = _canon(crop).astype(np.float32)
        best_d, best_score = None, -1.0
        for d, tmpl in self.atlas.items():
            t = tmpl.astype(np.float32)
            den = float(np.linalg.norm(c) * np.linalg.norm(t)) + 1e-6
            score = float((c * t).sum()) / den
            if score > best_score:
                best_d, best_score = d, score
        return best_d, best_score

    def _maybe_debug(self, frame, mode):
        if not self.cfg["debug"]:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            x0, y0, x1, y1 = self.cfg["flash_roi"]
            cv2.rectangle(vis, (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h)),
                          (0, 215, 255), 1)
            cv2.putText(vis, f"GOLDEN:{mode}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
            cv2.imshow("Golden Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
