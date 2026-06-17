"""
police_car.py
=============
Police-car override for the driving agent.

The level occasionally spawns a stationary police car. Hitting it is an instant
game over, so while a cop is on screen this controller TAKES OVER from the main
policy: it steers the car to grab the nearest red token while hard-avoiding the
cop, then hands control straight back once the cop leaves the frame.

Detection (see _detect_cop) keys off the cop's livery -- it is the only thing on
screen that is strongly RED and strongly BLUE at once. The HSV bands for those
two colours live in PCFG below and were measured from the real sprite with
cop_hsv_sample.py; tune them live via the "Police Masks" window (debug=True).

This module owns only cop colours. It borrows the brain's CONFIG for shared,
non-cop concerns: road geometry (the ROI/asphalt mask) and token size/shape.
"""

import time
import cv2
import numpy as np

from agent_policy import (
    CONFIG,            # brain config: reused only for road geometry + token size
    _build_color_mask,
    _fill_blobs,
    _detect_tokens,
    _roi_polygon,
)


# =========================================================================
# CONFIG -- tune on real frames (HSV is OpenCV's: H 0-179, S/V 0-255)
# =========================================================================
PCFG = {
    # ---- Police-car detection (RED + BLUE together) --------------------
    # The cop sprite is a compact car that is strongly RED and strongly BLUE,
    # side by side. Nothing else on screen is both at once: the sky is blue-only,
    # red tokens and the red roadside are red-only. So we detect ONE connected
    # blob that contains BOTH colours. This is size-robust (works for a small or
    # distant sprite), unlike requiring a large colour-overlap area.
    # Both bands are OWNED HERE -- the cop detector does not borrow the brain's
    # token-red (that band is tuned for pale salmon orbs, a different target).
    # Anchored on the sampled sprite (cop_hsv_sample.py): blue H~125 / red H~0,
    # both fully saturated, V from ~68 up. S/V floors are opened from the sprite
    # values to survive the live JPEG feed + low-light dimming.
    # Calibrate against the live "Police Masks" window (set debug=True).
    "blue": [((108, 80, 45), (132, 255, 255))],
    # The cop's red half/lights: pure red on the 0/179 hue seam, so two bands.
    "red":  [((0, 80, 45),   (12, 255, 255)),     # red near hue 0
             ((165, 80, 45), (179, 255, 255))],   # hue wrap-around
    # px the blue|red union is closed by, so the red half and blue half merge
    # into ONE connected component even with a seam/gap between them.
    "cop_dilate": 11,
    # A blob must hold at least this much blue AND this much red to be the cop
    # (rejects a lone stray pixel of the other colour). Fraction of W*H each.
    "cop_min_color_frac": 0.00015,
    # Minimum size of the whole red+blue blob. Low so even a distant sprite still
    # qualifies; raise if scenery false-triggers. Fraction of W*H.
    "police_min_area_frac": 0.0006,
    # Vertical search band for the cop (fractions of H): below the sky, above the
    # player's own red+blue car at the very bottom. Wider than the token ROI so a
    # cop anywhere across the road is seen; the red+blue requirement keeps scenery
    # out, so the narrow trapezoid isn't needed here.
    "cop_search_top_frac": 0.28,
    "cop_search_bot_frac": 0.80,

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

    # ---- Red-token detection -------------------------------------------
    # The red token is a PALE, glossy salmon orb (washed-out low-saturation
    # centre, bright/high value) -- not the cop's solid deep red. Here red is
    # the GOAL we must not miss, so this range is MORE LENIENT than the brain's
    # hazard range (lower S-floor to catch the pale token; higher V-floor since
    # the orb is bright, which keeps dull reddish clutter out). The road-surface
    # gate + circularity still reject the red roadside, so leniency is safe.
    "token_red": [((0, 28, 90),   (12, 255, 255)),    # pale/salmon red
                  ((164, 28, 90), (179, 255, 255))],   # hue wrap-around
    "min_token_area_frac": CONFIG["min_token_area_frac"],
    "min_circularity":     CONFIG["min_circularity"],

    # ---- Tooling -------------------------------------------------------
    "debug": True,          # draw the live overlay window (tuning only)
    "police_duration": 10.0,  # cop lifetime in seconds (telemetry only; logic is presence-driven)
}


def _clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


class PoliceCarController:
    """Detects the stationary police car by its red+blue livery and, while it is
    on screen, takes over steering to grab the nearest red token while hard-
    avoiding the cop. When the cop is gone, the policy's output passes through
    unchanged."""

    def __init__(self, config=None):
        self.cfg = dict(PCFG)
        if config:
            self.cfg.update(config)
        self.police_active = False
        self.cop_pos = None        # (cx, cy) of the detected cop, or None
        self.cop_box = None        # (x, y, w, h) bounding box, or None
        self._prev_steer = 0.0
        self._first_seen_t = None  # timestamp of first sighting (telemetry only)
        self._dbg_blue = None      # last blue mask (diagnostic window)
        self._dbg_red = None       # last red mask (diagnostic window)

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

        # --- 1. Detect the cop (red+blue blob in the road search band) --
        self.cop_pos, self.cop_box = self._detect_cop(hsv, w, h)

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
    def _cop_search_mask(self, w, h):
        """Wide horizontal band over the road (below sky, above the player's car)
        used to look for the cop. Stored geometry, cheap to rebuild each frame."""
        band = np.zeros((h, w), np.uint8)
        y0 = int(self.cfg["cop_search_top_frac"] * h)
        y1 = int(self.cfg["cop_search_bot_frac"] * h)
        band[y0:y1, :] = 255
        return band

    def _detect_cop(self, hsv, w, h):
        """Return ((cx, cy), (x, y, bw, bh)) for the cop: the largest connected
        blob inside the search band that contains BOTH enough blue AND enough red
        px. Size-robust (no large-overlap requirement) so a small/distant sprite
        still registers. Requiring both colours rejects blue-only sky and red-only
        tokens/roadside. Returns (None, None) if nothing qualifies."""
        band = self._cop_search_mask(w, h)
        blue = cv2.bitwise_and(_build_color_mask(hsv, self.cfg["blue"]), band)
        red = cv2.bitwise_and(_build_color_mask(hsv, self.cfg["red"]), band)

        # Merge the red half and blue half into one component (close over the seam).
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.cfg["cop_dilate"], self.cfg["cop_dilate"]))
        combo = cv2.morphologyEx(cv2.bitwise_or(blue, red), cv2.MORPH_CLOSE, k)

        # Stash masks for the diagnostic window.
        self._dbg_blue, self._dbg_red = blue, red

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(combo, 8)
        min_area = self.cfg["police_min_area_frac"] * (w * h)
        min_color = self.cfg["cop_min_color_frac"] * (w * h)
        best, best_area = None, 0.0
        for i in range(1, n):                       # 0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area or area <= best_area:
                continue
            comp = labels == i
            if np.count_nonzero(blue[comp]) < min_color:
                continue
            if np.count_nonzero(red[comp]) < min_color:
                continue
            best, best_area = i, area
        if best is None:
            return None, None

        x = int(stats[best, cv2.CC_STAT_LEFT])
        y = int(stats[best, cv2.CC_STAT_TOP])
        bw = int(stats[best, cv2.CC_STAT_WIDTH])
        bh = int(stats[best, cv2.CC_STAT_HEIGHT])
        cx, cy = centroids[best]
        return (float(cx), float(cy)), (x, y, bw, bh)

    def _road_region(self, hsv, roi_mask):
        """Grey-asphalt mask, dilated to cover on-road tokens, ANDed with the ROI
        -- mirrors HeuristicPolicy._road_region using the brain's CONFIG. Tokens
        live on this grey surface; the RED road shoulder does not, so ANDing the
        red mask with this drops the roadside while keeping real tokens."""
        road = cv2.inRange(
            hsv,
            np.array((0, 0, CONFIG["road_val_min"]), np.uint8),
            np.array((179, CONFIG["road_sat_max"], 255), np.uint8),
        )
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (CONFIG["road_dilate"], CONFIG["road_dilate"]))
        road = cv2.dilate(road, k)          # grow over tokens sitting on the road
        return cv2.bitwise_and(road, roi_mask)

    def _red_token_mask(self, hsv, roi_mask, w, h):
        """Red tokens ON the grey road, with the cop's bounding box zeroed out so
        the cop's red body is never chased as a token. Restricting to the road
        surface (not just the ROI) is what excludes the red roadside/shoulder."""
        road_region = self._road_region(hsv, roi_mask)
        red = cv2.bitwise_and(_build_color_mask(hsv, self.cfg["token_red"]), road_region)
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
            # draw the cop search band so we can see where we look for the cop
            y0 = int(self.cfg["cop_search_top_frac"] * h)
            y1 = int(self.cfg["cop_search_bot_frac"] * h)
            cv2.rectangle(vis, (0, y0), (w - 1, y1), (0, 200, 200), 1)
            cv2.putText(vis, f"POLICE:{mode}  steer={steer:+.2f}{remain}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Police Debug", vis)

            # --- Mask diagnostic: see EXACTLY what blue/red the cop produces.
            # Blue mask tinted blue, red mask tinted red, over a dim frame. If the
            # cop appears here without lighting up BOTH colours, widen PCFG["blue"]
            # / PCFG["red"] until it does.
            if self._dbg_blue is not None and self._dbg_red is not None:
                masks = cv2.addWeighted(frame, 0.35, np.zeros_like(frame), 0, 0)
                masks[self._dbg_blue > 0] = (255, 0, 0)
                masks[self._dbg_red > 0] = (0, 0, 255)
                cv2.imshow("Police Masks (B=blue R=red)", masks)
            cv2.waitKey(1)
        except Exception:
            pass
