import time
import cv2
import numpy as np

from event_manager import Override
from agent_policy import (
    CONFIG,
    _build_color_mask,
    _fill_blobs,
    _detect_tokens,
    _roi_polygon,
)


# =========================================================================
# CONFIG -- tune on real frames (HSV: H 0-179, S/V 0-255)
# =========================================================================
PCFG = {
    # ---- Cop detection (blue/purple lightbar) ----
    "blue": [((100, 80, 60), (150, 255, 255))],   # lightbar hue + S/V floor
    "police_min_area_frac": 0.003,  # min blob size (small = far cop still seen)
    "police_min_ar":   0.5,   # bbox w/h min (drop thin verticals)
    "police_max_ar":   2.5,   # bbox w/h max (drop wide horizontals)
    "police_red_pad":  12,    # px around blue bbox to look for the red body
    "police_red_frac": 0.05,  # min red fraction in that box to confirm a cop

    # ---- Steering toward the red token ----
    "steer_gain":   2.2,    # P-gain on horizontal error
    "smoothing":    0.5,    # low-pass alpha (higher = snappier)
    "center_x_frac": 0.5,   # car's horizontal position

    # ---- Cop avoidance (cop = game over, so it dominates) ----
    "cop_near_y_frac":   0.55,  # hard-dodge: cop below this fraction of H
    "cop_path_half_w":   0.22,  # hard-dodge path width
    "cop_avoid_steer":   1.0,   # hard-dodge steer magnitude
    "cop_bias_gain":     0.6,   # push away from cop while seeking a token
    "cop_warn_y_frac":   0.18,  # early-warn: cop below this -> start dodging
    "cop_warn_half_w":   0.50,  # warn path width (whole road)
    "cop_warn_steer":    0.75,  # warn steer magnitude
    "cop_lane_half_w":   0.18,  # don't chase tokens this close to the cop's x
    "cop_memory_frames": 6,     # keep dodging this long after the cop is lost

    # ---- Throttle ----
    "accel_seek":        0.85,  # hunting a token, no cop
    "accel_cop_present": 0.20,  # cop on screen -> slow down
    "accel_seek_cop":    0.60,  # chasing a token with cop still around
    "accel_dodge":       0.10,  # hard brake during a close dodge

    # ---- Red-token detection (reuse the brain's settings) ----
    "min_token_area_frac": CONFIG["min_token_area_frac"],
    "min_circularity":     CONFIG["min_circularity"],

    # ---- Tooling ----
    "debug": True,            # draw the live overlay
    "police_duration": 10.0,  # cop lifetime (telemetry only)
}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class PoliceCarController:
    """While the police car is on screen, takes over steering to grab the nearest
    red token while avoiding the cop. When the cop is gone, the policy passes through."""

    def __init__(self, config=None):
        self.cfg = dict(PCFG)
        if config:
            self.cfg.update(config)
        self.police_active = False
        self.cop_pos = None        # (cx, cy) or None
        self.cop_box = None        # (x, y, w, h) or None
        self._cop_miss = 0         # frames since the cop was last seen
        self._prev_steer = 0.0
        self._prev_mode = "CLEAR"
        self._first_seen_t = None  # first-sighting time (telemetry)
        self._cop_detect_count = 0

    # -- main entry ------------------------------------------------------
    def apply(self, front_frame, steering, acceleration):
        """Take the policy's (steering, acceleration) and override it while the cop
        is visible; otherwise return them unchanged."""
        if front_frame is None:
            return steering, acceleration

        h, w = front_frame.shape[:2]
        cfg = self.cfg
        center_x = w * cfg["center_x_frac"]

        hsv = cv2.cvtColor(front_frame, cv2.COLOR_BGR2HSV)

        # Cop ROI: full-width band (cop can be anywhere side-to-side); vertical
        # bounds exclude the player's own car at the bottom.
        cop_roi = np.zeros((h, w), np.uint8)
        top_y = int(h * CONFIG["roi_top_y"])
        bot_y = int(h * CONFIG["roi_bottom_y"])
        cop_roi[top_y:bot_y, :] = 255

        # Token ROI: road trapezoid (fewer grass/scenery false positives).
        roi_mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(roi_mask, [_roi_polygon(w, h)], 255)

        # --- 1. Detect the cop ------------------------------------------
        det_pos, det_box = self._detect_cop(hsv, cop_roi, w, h)

        if det_pos is not None:
            # Fresh sighting: trust it, reset the miss counter.
            self.cop_pos, self.cop_box = det_pos, det_box
            self._cop_miss = 0
        elif self.police_active and self._cop_miss < cfg["cop_memory_frames"] \
                and self.cop_pos is not None:
            # Lost for a frame (e.g. cop hidden behind its token): keep dodging
            # the last known spot instead of handing control back.
            self._cop_miss += 1
        else:
            # Cop gone: policy stays in control.
            if self.police_active:
                self.police_active = False
                self._first_seen_t = None
            self.cop_pos, self.cop_box = None, None
            self._cop_miss = 0
            self._maybe_debug(front_frame, None, None, steering, "CLEAR")
            return steering, acceleration

        # --- Cop present: take over -------------------------------------
        if not self.police_active:
            self.police_active = True
            self._first_seen_t = time.time()
            self._cop_detect_count += 1

        cop_cx, cop_cy = self.cop_pos

        # --- 2. Find red tokens, excluding the cop's body ---------------
        red = self._red_token_mask(hsv, roi_mask, w, h)
        reds = _detect_tokens(red,
                              cfg["min_token_area_frac"] * (w * h),
                              cfg["min_circularity"])
        # Drop tokens in the cop's lane -- chasing one steers into the cop.
        cop_lane = cfg["cop_lane_half_w"] * w
        safe_reds = [t for t in reds if abs(t["cx"] - cop_cx) > cop_lane]
        target = max(safe_reds, key=lambda t: t["cy"]) if safe_reds else None

        # --- 3. Decide steering / throttle ------------------------------
        # Tier 1: cop close AND dead ahead.
        cop_blocking = (cop_cy >= cfg["cop_near_y_frac"] * h and
                        abs(cop_cx - center_x) <= cfg["cop_path_half_w"] * w)
        # Tier 2: cop anywhere in a broad path ahead.
        cop_warning = (not cop_blocking and
                       cop_cy >= cfg["cop_warn_y_frac"] * h and
                       abs(cop_cx - center_x) <= cfg["cop_warn_half_w"] * w)

        base_accel = cfg["accel_cop_present"]  # slow down while the cop is up

        if cop_blocking:
            # Hard dodge: full steer to the open side + brake.
            steer = self._escape_direction(cop_cx, w, center_x) * cfg["cop_avoid_steer"]
            accel = cfg["accel_dodge"]
            mode = "COP-DODGE"
        elif cop_warning:
            # Early dodge: steer away while there's still room.
            steer = self._escape_direction(cop_cx, w, center_x) * cfg["cop_warn_steer"]
            accel = base_accel
            mode = "COP-WARN"
        elif target is not None:
            # Chase a token that's clear of the cop; bias still bends away from it.
            err = (target["cx"] - center_x) / (0.5 * w)
            steer = cfg["steer_gain"] * err
            steer += cfg["cop_bias_gain"] * (-(cop_cx - center_x) / (0.5 * w)) * \
                self._cop_proximity(cop_cy, h)
            accel = cfg["accel_seek_cop"]
            mode = "SEEK-RED"
        else:
            # Cop up, no safe token: commit to a lane change toward the open side.
            steer = self._escape_direction(cop_cx, w, center_x) * cfg["cop_warn_steer"]
            accel = base_accel
            mode = "SEARCH-RED"

        # Avoidance needs instant steer -- skip the low-pass filter.
        if mode in ("COP-DODGE", "COP-WARN", "SEARCH-RED"):
            steer = _clamp(steer)
            self._prev_steer = steer
        else:
            # Clear the inherited dodge steer when going back to seeking.
            if self._prev_mode in ("COP-DODGE", "COP-WARN", "SEARCH-RED"):
                self._prev_steer = 0.0
            steer = self._smooth(_clamp(steer))
        self._prev_mode = mode
        self._maybe_debug(front_frame, target, self.cop_box, steer, mode)
        return steer, accel

    # -- arbiter interface (used by EventManager) ------------------------
    def evaluate(self, front_frame, back_frame, ctx):
        """Bid to take over while the cop is on screen. A close dead-ahead cop is a
        game-over hazard -> priority 100 (collision dodge); otherwise seeking the red
        token sits at 60. Returns None when no cop is present (policy passes through).
        The Police pass (collecting a red within 5s) is latched by EventManager from
        the HUD red-count tick while this handler is active."""
        steer, accel = self.apply(front_frame, ctx.base_steer, ctx.base_accel)
        if not self.police_active:
            return None
        pri = 100 if self._prev_mode == "COP-DODGE" else 60
        return Override(steer, accel, pri, f"POLICE:{self._prev_mode}")

    # -- detection helpers ----------------------------------------------
    def _detect_cop(self, hsv, roi_mask, w, h):
        """Return ((cx, cy), (x, y, bw, bh)) for the largest car-sized blue blob that
        also passes the aspect-ratio and red co-presence checks, else (None, None)."""
        cfg = self.cfg
        blue = cv2.bitwise_and(_build_color_mask(hsv, cfg["blue"]), roi_mask)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))   # merge lightbar pixels
        cnts, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = cfg["police_min_area_frac"] * (w * h)

        # Red mask, full-frame so the body at the road edge isn't clipped.
        red_mask = _build_color_mask(hsv, CONFIG["red"])

        best, best_area = None, 0.0
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area <= best_area:
                continue

            x, y, bw, bh = cv2.boundingRect(c)

            # Gate 1: aspect ratio (cop sprite is roughly square).
            if bh == 0:
                continue
            ar = bw / bh
            if ar < cfg["police_min_ar"] or ar > cfg["police_max_ar"]:
                continue

            # Gate 2: red body must sit near the blue lightbar.
            pad = cfg["police_red_pad"]
            rx0, ry0 = max(0, x - pad), max(0, y - pad)
            rx1, ry1 = min(w, x + bw + pad), min(h, y + bh + pad)
            box_area = (rx1 - rx0) * (ry1 - ry0)
            if box_area == 0:
                continue
            if cv2.countNonZero(red_mask[ry0:ry1, rx0:rx1]) < cfg["police_red_frac"] * box_area:
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
        """Red mask on the road with the cop's box zeroed out, so the cop's red body
        is never chased as a token."""
        red = cv2.bitwise_and(_build_color_mask(hsv, CONFIG["red"]), roi_mask)
        red = _fill_blobs(red)
        if self.cop_box is not None:
            x, y, bw, bh = self.cop_box
            pad = int(0.04 * w)  # pad to also drop the cop's red rim
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
            red[y0:y1, x0:x1] = 0
        return red

    def _escape_direction(self, cop_cx, w, center_x):
        """+1.0 (right) or -1.0 (left): dodge toward the side of the cop with more
        road, using the ROI's bottom half-width as the road extent."""
        bhw = CONFIG["roi_bot_half_w"] * w
        left_room = cop_cx - (center_x - bhw)
        right_room = (center_x + bhw) - cop_cx
        return 1.0 if right_room > left_room else -1.0

    def _cop_proximity(self, cop_cy, h):
        """0..1 weight: closer cop -> stronger avoidance bias."""
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
            cv2.putText(vis, f"POLICE:{mode}  steer={steer:+.2f}{remain}  #det={self._cop_detect_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Police Debug", vis)
            cv2.waitKey(1)
        except Exception:
            pass
