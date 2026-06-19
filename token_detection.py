"""
token_detection.py
===================
Perception component: detects green/red/yellow tokens (large semi-transparent
glossy spheres) on the road from a front-camera frame.

ROI strategy: the road is a perspective trapezoid (narrow near the horizon,
nearly full-width near the car), not a fixed-width band -- a fixed left/right
margin either lets grass through near the horizon or clips real lanes near the
car. So we crop top (sky/skyline/scoreboard) and bottom (player's own car),
then mask a trapezoid within that band before HSV thresholding.

roi_top_frac / trapezoid width fractions below are estimated from a real
screenshot -- tune against the debug overlay if the trapezoid doesn't track
the visible road edges.
"""

import cv2
import numpy as np

CONFIG = {
    "hsv": {
        # Green -- lime/spring/mint/cyan-green spheres. S/V lowered so semi-
        # transparent glowing spheres aren't missed; circularity filter (below)
        # removes grass, which shares the hue but has jagged edges.
        "green":  {"lower": (40, 40, 80),  "upper": (90, 255, 255)},
        # Red -- pinkish-red semi-transparent spheres, hue wraps at 0/180.
        # Low S floor catches pale/pastel reds; V>=80 keeps dark background out.
        "red":    [{"lower": (0, 20, 80),   "upper": (15, 160, 255)},
                   {"lower": (160, 20, 80), "upper": (180, 160, 255)}],
        # Yellow/gold -- kept below the green range; medium S, V>=80.
        "yellow": {"lower": (15, 80, 80),  "upper": (38, 255, 255)},
    },
    "roi_top_frac": 0.50,            # skip top half of frame (sky / city skyline / scoreboard); ASSUMED from screenshot, tune in-game
    "roi_bottom_frac": 0.85,         # skip bottom 15% of frame (player's own car)
    "roi_top_half_width_frac": 0.08,    # trapezoid half-width at the horizon (top of ROI), fraction of frame width; ASSUMED from screenshot
    "roi_bottom_half_width_frac": 0.48, # trapezoid half-width at the bottom of ROI (near the car), fraction of frame width
    "min_token_area": 40,
    "min_circularity": 0.55,   # 4*pi*area/perimeter^2, filters non-round blobs
}

_road_mask_cache = {}


def _hsv_mask(hsv, ranges):
    if isinstance(ranges, dict):
        ranges = [ranges]
    mask = None
    for r in ranges:
        m = cv2.inRange(hsv, np.array(r["lower"], np.uint8), np.array(r["upper"], np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def build_road_trapezoid_mask(roi_shape):
    """Trapezoid road mask local to the already top/bottom-cropped ROI: narrow
    at its top edge (horizon), wide at its bottom edge (near the car)."""
    h, w = roi_shape[:2]
    key = (w, h)
    if key in _road_mask_cache:
        return _road_mask_cache[key]

    cx = w / 2.0
    top_hw = w * CONFIG["roi_top_half_width_frac"]
    bot_hw = w * CONFIG["roi_bottom_half_width_frac"]
    pts = np.array([
        [cx - top_hw, 0],
        [cx + top_hw, 0],
        [cx + bot_hw, h],
        [cx - bot_hw, h],
    ], np.int32)

    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    _road_mask_cache[key] = mask
    return mask


def _detect_color_blobs(hsv_frame, roi_mask, color_key):
    """Return [(cx, cy, area), ...] for round blobs of `color_key`, local to hsv_frame."""
    mask = cv2.morphologyEx(_hsv_mask(hsv_frame, CONFIG["hsv"][color_key]),
                             cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.bitwise_and(mask, roi_mask)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < CONFIG["min_token_area"]:
            continue
        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < CONFIG["min_circularity"]:
            continue
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        blobs.append((M["m10"] / M["m00"], M["m01"] / M["m00"], area))
    return blobs


def detect_tokens(frame):
    """Detect green/red/yellow tokens in `frame`.

    Returns (green_blobs, red_blobs, yellow_blobs, roi_top, roi_bottom,
    top_half_width_px, bottom_half_width_px), with blob coordinates in
    full-frame space.
    """
    h, w = frame.shape[:2]
    roi_top = int(h * CONFIG["roi_top_frac"])
    roi_bottom = int(h * CONFIG["roi_bottom_frac"])
    roi = frame[roi_top:roi_bottom]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    road_mask = build_road_trapezoid_mask(roi.shape)
    top_half_width_px = int(roi.shape[1] * CONFIG["roi_top_half_width_frac"])
    bottom_half_width_px = int(roi.shape[1] * CONFIG["roi_bottom_half_width_frac"])

    def _detect(color_key):
        local = _detect_color_blobs(hsv, road_mask, color_key)
        return [(cx, cy + roi_top, area) for cx, cy, area in local]

    green_blobs = _detect("green")
    red_blobs = _detect("red")
    yellow_blobs = _detect("yellow")
    return (green_blobs, red_blobs, yellow_blobs, roi_top, roi_bottom,
            top_half_width_px, bottom_half_width_px)
