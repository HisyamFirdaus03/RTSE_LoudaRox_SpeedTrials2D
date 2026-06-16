import time
import cv2
import numpy as np

from agent_policy import (
    CONFIG,
    _build_color_mask,
    _fill_blobs,
    _detect_tokens,
    _roi_polygon,
)


# =========================================================================
# CONFIG -- tune on real frames (HSV is OpenCV's: H 0-179, S/V 0-255)
# =========================================================================
PCFG = {
    # ---- Police-car detection (the blue livery / lightbar) -------------
    # The cop's blue is the one cue that separates it from red tokens and the
    # red player car. Wide-ish band with S/V floors to ignore dim/grey scenery.
    "blue": [((100, 80, 60), (130, 255, 255))],
    # A car is much bigger than a token. This area gate (fraction of W*H) is the
    # main false-trigger guard: raise it if blue scenery triggers, lower it if
    # the cop is missed.
    "police_min_area_frac": 0.010,

    # ---- Steering toward the red token --------------------------------
    "steer_gain":   2.2,    # P-gain on normalized horizontal error (matches brain)
    "smoothing":    0.5,    # low-pass alpha: s = a*new + (1-a)*prev (higher = snappier)
    "center_x_frac": 0.5,   # where the car sits horizontally (bottom-center)

    # ---- Cop collision avoidance (dominates -- cop = game over) --------
    "cop_near_y_frac":   0.55,  # "close" = cop centroid below this fraction of H
    "cop_path_half_w":   0.22,  # "in our path" = |cop_cx - center| < this * W
    "cop_avoid_steer":   1.0,   # steer magnitude used to swerve around a blocking cop
    "cop_bias_gain":     0.8,   # mild lateral push away from the cop even when not critical

    # ---- Throttle ------------------------------------------------------
    "accel_seek":   0.85,   # forward throttle while hunting the red token
    "accel_dodge":  0.40,   # ease off so a swerve around the cop actually lands

    # ---- Red-token detection (reuse the brain's tuned settings) --------
    "min_token_area_frac": CONFIG["min_token_area_frac"],
    "min_circularity":     CONFIG["min_circularity"],

    # ---- Tooling -------------------------------------------------------
    "debug": True,          # draw the live overlay window (tuning only)
    "police_duration": 10.0,  # cop lifetime in seconds (telemetry only; logic is presence-driven)
}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class PoliceCarController:
    """Detects the stationary police car via its blue livery and, while it is on
    screen, takes over steering to grab the nearest red token while hard-avoiding
    the cop. When the cop is gone, the policy's output passes through unchanged."""

    def __init__(self, config=None):
        self.cfg = dict(PCFG)
        if config:
            self.cfg.update(config)
        self.police_active = False
        self.cop_pos = None        # (cx, cy) of the detected cop, or None
        self.cop_box = None        # (x, y, w, h) bounding box, or None
        self._prev_steer = 0.0
        self._first_seen_t = None  # timestamp of first sighting (telemetry only)

    # -- main entry ------------------------------------------------------
    def apply(self, front_frame, steering, acceleration):
        """Take the policy's (steering, acceleration); return possibly-overridden
        (steering, acceleration). While the cop is visible we fully take over."""
        if front_frame is None:
            return steering, acceleration

        h, w = front_frame.shape[:2]
        cfg = self.cfg
        center_x = w * cfg["center_x_frac"]

        # Road ROI (trapezoid) -- reuse the brain's road geometry. The player's
        # own car sits below roi_bottom_y and is excluded, so any car-sized blob
        # we find here is the cop, not us.
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi_mask, [_roi_polygon(w, h)], 255)
        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)

        # --- 1. Detect the cop via its blue livery ----------------------
        self.cop_pos, self.cop_box = self._detect_cop(hsv, roi_mask, w, h)

        if self.cop_pos is None:
            # Presence-driven revert: no cop -> policy stays in control.
            if self.police_active:
                self.police_active = False
                self._first_seen_t = None
            self._maybe_debug(front_frame, None, None, steering, "CLEAR")
            return steering, acceleration

        # --- Cop is present: FULL TAKEOVER ------------------------------
        if not self.police_active:
            self.police_active = True
            self._first_seen_t = time.time()

        cop_cx, cop_cy = self.cop_pos

        # --- 2. Find red tokens on the road, excluding the cop's body ---
        red = self._red_token_mask(hsv, roi_mask, w, h)
        reds = _detect_tokens(red,
                              cfg["min_token_area_frac"] * (w * h),
                              cfg["min_circularity"])
        target = max(reds, key=lambda t: t["cy"]) if reds else None  # closest to car

        # --- 3. Decide steering / throttle ------------------------------
        cop_blocking = (cop_cy >= cfg["cop_near_y_frac"] * h and
                        abs(cop_cx - center_x) <= cfg["cop_path_half_w"] * w)

        if cop_blocking:
            # Cop dead-ahead and close -> dodge to the side with more room and
            # brake. This dominates the red-seek (a collision is game over).
            direction = -1.0 if cop_cx > center_x else 1.0   # steer away from the cop
            steer = direction * cfg["cop_avoid_steer"]
            accel = cfg["accel_dodge"]
            mode = "COP-DODGE"
        elif target is not None:
            # Steer toward the red token, with a mild push away from the cop so
            # we thread past it instead of clipping it.
            err = (target["cx"] - center_x) / (0.5 * w)
            steer = cfg["steer_gain"] * err
            steer += cfg["cop_bias_gain"] * (-(cop_cx - center_x) / (0.5 * w)) * \
                self._cop_proximity(cop_cy, h)
            accel = cfg["accel_seek"]
            mode = "SEEK-RED"
        else:
            # Cop present but no red visible yet -> keep moving, steer toward the
            # open side away from the cop to keep hunting within the 10s window.
            steer = (-(cop_cx - center_x) / (0.5 * w)) * cfg["cop_bias_gain"]
            accel = cfg["accel_seek"]
            mode = "SEARCH-RED"

        steer = self._smooth(_clamp(steer))
        self._maybe_debug(front_frame, target, self.cop_box, steer, mode)
        return steer, accel

    # -- detection helpers ----------------------------------------------
    def _detect_cop(self, hsv, roi_mask, w, h):
        """Return ((cx, cy), (x, y, bw, bh)) for the largest car-sized blue blob
        on the road, or (None, None) if none qualifies."""
        blue = cv2.bitwise_and(_build_color_mask(hsv, self.cfg["blue"]), roi_mask)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))   # merge livery + lightbar
        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = self.cfg["police_min_area_frac"] * (w * h)
        best, best_area = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area <= best_area:
                continue
            best, best_area = c, area
        if best is None:
            return None, None
        x, y, bw, bh = cv2.boundingRect(best)
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None, None
        return (M["m10"] / M["m00"], M["m01"] / M["m00"]), (x, y, bw, bh)

    def _red_token_mask(self, hsv, roi_mask, w, h):
        """Red mask on the road, with the cop's bounding box zeroed out so the
        cop's red body is never chased as a token."""
        red = cv2.bitwise_and(_build_color_mask(hsv, CONFIG["red"]), roi_mask)
        red = _fill_blobs(red)
        if self.cop_box is not None:
            x, y, bw, bh = self.cop_box
            # pad the box a little so the cop's red rim is fully excluded too
            pad = int(0.04 * w)
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            red[y0:y1, x0:x1] = 0
        return red

    def _cop_proximity(self, cop_cy, h):
        """0..1 weight: closer cop (larger cy) -> stronger avoidance bias."""
        return float(np.clip((cop_cy / h) ** 2, 0.0, 1.0))

    def _smooth(self, steer):
        a = self.cfg["smoothing"]
        self._prev_steer = a * steer + (1 - a) * self._prev_steer
        return _clamp(self._prev_steer)

    # -- debug overlay ---------------------------------------------------
    def _maybe_debug(self, frame, target, cop_box, steer, mode):
        if not self.cfg["debug"]:
            return
        try:
            vis = frame.copy()
            h, w = vis.shape[:2]
            cv2.polylines(vis, [_roi_polygon(w, h)], True, (255, 255, 255), 1)
            if cop_box is not None:
                x, y, bw, bh = cop_box
                cv2.rectangle(vis, (x, y), (x + bw, y + bh), (255, 0, 0), 2)
                cv2.putText(vis, "COP", (x, max(15, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            if target is not None:
                p = (int(target["cx"]), int(target["cy"]))
                cv2.circle(vis, p, 10, (0, 0, 255), 2)
                cv2.putText(vis, "RED", (p[0] + 8, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            remain = ""
            if self._first_seen_t is not None:
                remain = f"  t~{max(0.0, self.cfg['police_duration'] - (time.time() - self._first_seen_t)):.1f}s"
            cv2.putText(vis, f"POLICE:{mode}  steer={steer:+.2f}{remain}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Police Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
