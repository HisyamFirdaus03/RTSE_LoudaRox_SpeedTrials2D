"""
Pure computer-vision detection routines for SpeedTrials2D.

Every function here takes a BGR numpy frame (and the CONFIG dict) and returns
plain data — no threading, no locks, no sockets. This makes each function
testable standalone against saved frames (see the __main__ block below).

NOTE: All thresholds/ranges in CONFIG are placeholders and MUST be calibrated
against the live simulator (see the plan's calibration checklist).
"""

import math
from collections import namedtuple

import cv2
import numpy as np

# ---------------------------------------------------------
# Tunable configuration — calibrate every value here against the live sim
# ---------------------------------------------------------
CONFIG = {
    # Working resolution for CV ops (the raw decoded frame may be larger;
    # downscale internally for speed before running heavy detections)
    'cv_width': 320,
    'cv_height': 240,

    # Regions of interest, expressed as fractions of the (possibly downscaled)
    # frame: (y_start, y_end, x_start, x_end)
    'road_roi': (0.45, 1.0, 0.0, 1.0),
    'token_roi': (0.25, 0.85, 0.0, 1.0),
    'rear_roi': (0.0, 0.7, 0.0, 1.0),

    # HSV color ranges as (lower, upper) tuples of (H, S, V).
    # Token ranges below are derived from the lab's reference token artwork
    # (glossy circular badges: mint-green, coral-red, gold/amber-yellow, and a
    # silver/gray badge that appears to be the "token type hidden" visual).
    # Ranges are intentionally widened on the low-S/high-V side to tolerate
    # the bright specular highlight on each badge — STILL NEEDS calibration
    # against actual in-game lighting/scale (this slide art != in-game render).
    'hsv_green_token':  ((45, 70, 110), (75, 255, 255)),    # mint/soft green
    'hsv_red_token_1':  ((0, 90, 110), (8, 255, 255)),      # coral/salmon red
    'hsv_red_token_2':  ((172, 90, 110), (180, 255, 255)),  # red hue wraparound
    'hsv_yellow_token': ((14, 110, 130), (28, 255, 255)),   # gold/amber
    'hsv_hidden_token': ((0, 0, 130), (180, 40, 220)),      # silver/gray (low sat, hue-agnostic)
    'hsv_road_lane':    ((0, 0, 150), (180, 60, 255)),
    'hsv_faster_car':   ((100, 80, 80), (130, 255, 255)),
    'hsv_police_car':   ((0, 0, 0), (180, 255, 60)),

    # Geometry / contour filters
    'min_token_contour_area': 80,
    'min_car_contour_area': 600,
    'token_min_circularity': 0.6,
    'car_min_aspect_ratio': 1.3,
    'car_max_aspect_ratio': 2.5,

    # Edge / line detection
    'canny_low': 50,
    'canny_high': 150,
    'hough_threshold': 40,
    'hough_min_line_len': 30,
    'hough_max_line_gap': 20,
    'min_slope': 0.3,            # ignore near-horizontal Hough segments

    # Discrete lane model — the simulator is a lane-snapping highway racer
    # (see the lab PDF: steering is *tapped* to move exactly one lane, not
    # held proportionally). `lane_boundary_fractions` are the x-positions
    # (as fractions of frame width) of the boundaries BETWEEN lanes, so
    # len(lane_boundary_fractions) == num_lanes - 1. With 3 lanes split
    # into even thirds: boundaries at 1/3 and 2/3.
    # STILL NEEDS calibration: confirm num_lanes against the live sim and
    # adjust boundaries to match where the lane-marking lines actually fall.
    'num_lanes': 3,
    'lane_boundary_fractions': [0.40, 0.60],
    'lane_index_debounce_frames': 3,

    # Brightness
    'brightness_threshold_v': 127.0,  # ~50% of 255
}

TokenDetection = namedtuple(
    'TokenDetection', ['color', 'lane', 'distance_estimate', 'cx', 'cy', 'area']
)
RearDetection = namedtuple(
    'RearDetection', ['faster_car', 'police_car', 'lane', 'distance_estimate']
)
EMPTY_REAR = RearDetection(faster_car=False, police_car=False, lane=None, distance_estimate=None)


# ---------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------
def _prep_frame(frame, cfg):
    """Downscale to the CV working resolution. Returns the resized BGR frame."""
    return cv2.resize(frame, (cfg['cv_width'], cfg['cv_height']))


def _crop_roi(frame, roi):
    """roi = (y_start, y_end, x_start, x_end) as fractions of frame size."""
    h, w = frame.shape[:2]
    y0, y1, x0, x1 = roi
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _classify_lane(cx_fraction, cfg):
    """
    Buckets an x-position (as a fraction of frame width) into a discrete lane
    index in [0, num_lanes-1], using the boundaries BETWEEN lanes from
    cfg['lane_boundary_fractions'] (sorted, len == num_lanes - 1).

    Index 0 is the leftmost lane. Used uniformly for the car's own lane,
    token lanes, and rear-car lanes so they all live in one coordinate system
    and can be compared directly (e.g. "is this token in my lane?").
    """
    boundaries = cfg['lane_boundary_fractions']
    lane = 0
    for boundary in boundaries:
        if cx_fraction >= boundary:
            lane += 1
        else:
            break
    return lane


def _mask_for_range(hsv, color_range):
    lower, upper = color_range
    return cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))


def _clean_mask(mask):
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


# ---------------------------------------------------------
# Discrete lane-position detection
#
# The simulator is a lane-snapping highway racer (per the lab PDF, steering
# is *tapped* to move exactly one lane — not held proportionally to "stay
# centered"). So perception's job here isn't "how far off-center am I"
# (a continuous correction) but "which discrete lane am I in right now"
# (an index that the policy/control layer compares against token/threat
# lanes to decide whether — and which way — to tap).
# ---------------------------------------------------------
_lane_index_history = []


def _lane_center_from_hough(roi_gray, cfg):
    """Returns the lane-center x-position as a fraction of ROI width, or None."""
    edges = cv2.Canny(roi_gray, cfg['canny_low'], cfg['canny_high'])
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=cfg['hough_threshold'],
        minLineLength=cfg['hough_min_line_len'],
        maxLineGap=cfg['hough_max_line_gap'],
    )
    if lines is None:
        return None

    h, w = roi_gray.shape[:2]
    left_xs, right_xs = [], []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < cfg['min_slope']:
            continue
        # x-intercept at the bottom of the ROI (y = h)
        x_at_bottom = x1 + (h - y1) / slope
        if slope < 0:
            left_xs.append(x_at_bottom)
        else:
            right_xs.append(x_at_bottom)

    if not left_xs and not right_xs:
        return None

    if left_xs and right_xs:
        lane_center = (np.mean(left_xs) + np.mean(right_xs)) / 2.0
    elif left_xs:
        lane_center = np.mean(left_xs) + w / 4.0
    else:
        lane_center = np.mean(right_xs) - w / 4.0

    return float(np.clip(lane_center / w, 0.0, 1.0))


def _lane_center_from_color_mask(roi_bgr, cfg):
    """Returns the road-surface centroid x-position as a fraction of ROI width, or None."""
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = _clean_mask(_mask_for_range(hsv, cfg['hsv_road_lane']))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < cfg['min_token_contour_area']:
        return None

    moments = cv2.moments(largest)
    if moments['m00'] == 0:
        return None

    cx = moments['m10'] / moments['m00']
    w = roi_bgr.shape[1]
    return float(np.clip(cx / w, 0.0, 1.0))


def detect_current_lane(frame, cfg=CONFIG):
    """
    Identifies which discrete lane the car currently occupies.

    Returns {'lane_index': int|None, 'lane_offset': float, 'confidence': float}:
      - lane_index: the bucketed lane in [0, num_lanes-1] the car appears
        centered in (debounced via majority-vote over recent frames to avoid
        flicker at lane boundaries), or None if not yet confidently known.
      - lane_offset: continuous in-lane centering signal in [-1, 1], kept only
        as a diagnostic (lane-snapping games auto-center within a lane once
        settled — this is NOT fed into steering).
      - confidence: fraction of the debounce window agreeing with lane_index
        (0.0 if lane_index is None).

    Tries Hough-line lane-marking detection first, falls back to a color-mask
    centroid of the road surface for plain unmarked roads.
    """
    small = _prep_frame(frame, cfg)
    roi_bgr = _crop_roi(small, cfg['road_roi'])
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    center_fraction = _lane_center_from_hough(roi_gray, cfg)
    if center_fraction is None:
        center_fraction = _lane_center_from_color_mask(roi_bgr, cfg)

    if center_fraction is None:
        _lane_index_history.clear()
        return {'lane_index': None, 'lane_offset': 0.0, 'confidence': 0.0}

    lane_offset = float(np.clip((center_fraction - 0.5) * 2.0, -1.0, 1.0))
    raw_index = _classify_lane(center_fraction, cfg)

    history = _lane_index_history
    history.append(raw_index)
    window = cfg['lane_index_debounce_frames']
    del history[:-window]

    counts = {}
    for idx in history:
        counts[idx] = counts.get(idx, 0) + 1
    majority_index, agree_count = max(counts.items(), key=lambda kv: kv[1])
    confidence = agree_count / len(history)

    lane_index = majority_index if len(history) >= window else None
    return {'lane_index': lane_index, 'lane_offset': lane_offset, 'confidence': confidence}


# ---------------------------------------------------------
# Token detection
# ---------------------------------------------------------
def _find_token_contours(hsv, color_range, cfg):
    mask = _clean_mask(_mask_for_range(hsv, color_range))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < cfg['min_token_contour_area']:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < cfg['token_min_circularity']:
            continue
        moments = cv2.moments(c)
        if moments['m00'] == 0:
            continue
        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']
        results.append((cx, cy, area))
    return results


def detect_tokens(frame, cfg=CONFIG):
    """
    Detects green/red/yellow/hidden tokens in the front camera frame via HSV
    color masking + circular-contour filtering. Returns a list of
    TokenDetection sorted closest-first (largest contour area / lowest in
    frame = closest).

    'hidden' corresponds to the silver/gray badge the lab's reference art
    shows alongside green/red/yellow — almost certainly the in-game visual
    used when the "next token type hidden" yellow effect is active (the
    agent can see *that* a token is there without knowing its real color).

    distance_estimate is a unitless proxy: larger value = closer.
    """
    small = _prep_frame(frame, cfg)
    roi_bgr = _crop_roi(small, cfg['token_roi'])
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = roi_bgr.shape[:2]

    color_ranges = {
        'green': [cfg['hsv_green_token']],
        'red': [cfg['hsv_red_token_1'], cfg['hsv_red_token_2']],
        'yellow': [cfg['hsv_yellow_token']],
        'hidden': [cfg['hsv_hidden_token']],
    }

    detections = []
    for color, ranges in color_ranges.items():
        for color_range in ranges:
            for cx, cy, area in _find_token_contours(hsv, color_range, cfg):
                lane = _classify_lane(cx / roi_w, cfg)
                distance_estimate = area * (1.0 + cy / roi_h)
                detections.append(TokenDetection(
                    color=color, lane=lane, distance_estimate=distance_estimate,
                    cx=cx, cy=cy, area=area,
                ))

    detections.sort(key=lambda d: d.distance_estimate, reverse=True)
    return detections


# ---------------------------------------------------------
# Rear-view event detection
# ---------------------------------------------------------
def detect_rear_events(back_frame, cfg=CONFIG):
    """
    Detects a faster car or police car approaching from behind via HSV color
    masking + car-shaped (wider-than-tall) contour filtering. Returns the
    single most significant detection (largest qualifying contour).

    Temporal tracking (e.g. "is it closing distance?") is intentionally left
    to the policy/state layer — this function is stateless per-frame.
    """
    if back_frame is None:
        return EMPTY_REAR

    small = _prep_frame(back_frame, cfg)
    roi_bgr = _crop_roi(small, cfg['rear_roi'])
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    roi_h, roi_w = roi_bgr.shape[:2]

    candidates = []
    for is_police, color_range in (
        (False, cfg['hsv_faster_car']),
        (True, cfg['hsv_police_car']),
    ):
        mask = _clean_mask(_mask_for_range(hsv, color_range))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < cfg['min_car_contour_area']:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if h == 0:
                continue
            aspect = w / h
            if not (cfg['car_min_aspect_ratio'] <= aspect <= cfg['car_max_aspect_ratio']):
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            candidates.append((area, is_police, cx, cy))

    if not candidates:
        return EMPTY_REAR

    area, is_police, cx, cy = max(candidates, key=lambda t: t[0])
    lane = _classify_lane(cx / roi_w, cfg)
    distance_estimate = area * (1.0 + cy / roi_h)

    return RearDetection(
        faster_car=not is_police,
        police_car=is_police,
        lane=lane,
        distance_estimate=distance_estimate,
    )


# ---------------------------------------------------------
# Brightness measurement
# ---------------------------------------------------------
def measure_brightness(frame, cfg=CONFIG):
    """Returns the mean V-channel value (0-255). Compare against
    cfg['brightness_threshold_v'] to detect the low-brightness event."""
    small = _prep_frame(frame, cfg)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


# ---------------------------------------------------------
# Standalone test harness — run against saved frames to tune CONFIG
# ---------------------------------------------------------
if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python perception.py <frame.png> [back_frame.png]")
        sys.exit(1)

    front = cv2.imread(sys.argv[1])
    if front is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(1)

    lane = detect_current_lane(front)
    print(f"Current lane: index={lane['lane_index']} offset={lane['lane_offset']:.3f} "
          f"confidence={lane['confidence']:.2f}")
    print(f"Brightness (V mean): {measure_brightness(front):.1f}")
    for t in detect_tokens(front):
        print(f"  Token: color={t.color} lane={t.lane} dist={t.distance_estimate:.1f}")

    if len(sys.argv) > 2:
        back = cv2.imread(sys.argv[2])
        if back is not None:
            rear = detect_rear_events(back)
            print(f"Rear: faster_car={rear.faster_car} police_car={rear.police_car} "
                  f"lane={rear.lane} dist={rear.distance_estimate}")

    cv2.imshow("front", front)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
