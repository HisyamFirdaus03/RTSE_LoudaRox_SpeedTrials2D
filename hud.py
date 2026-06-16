"""
hud.py  --  Module 1: read the score off the screen (the REWARD signal)
=======================================================================
To train with reinforcement learning, the agent needs a *reward*: a number that
says how good each moment was. In SpeedTrials2D the ground-truth score lives in
the on-screen counters (top-left): green / red / yellow tokens collected. We read
them straight from the pixels.

HOW (template-matching OCR):
  1. Crop the small box for each counter (fixed positions at 640x480).
  2. Isolate the digit pixels -- each counter is ONE bright colour, so a simple
     colour threshold gives a clean binary image of the digits.
  3. Split that into individual digit images (connected components, left->right).
  4. Match each digit against a tiny atlas of labelled digit pictures (0-9).
  5. Concatenate -> the integer.

The atlas (hud_digits/0.png .. 9.png) is built ONCE with the assisted labeller:
      python hud.py --build
This teaches a real ML habit: you label a little data, then the machine reads the
rest automatically.

Quick test on captured frames:
      python hud.py            # reads counts from ./frames/*.png

Everything works on a 640x480 frame. If a frame is a different size we resize to
640x480 first, so HUD positions stay valid.
"""

import os
import sys
import glob
import cv2
import numpy as np

HUD_W, HUD_H = 640, 480
ATLAS_DIR = "hud_digits"
DIGIT_SIZE = (18, 24)        # (w, h) canonical size every digit is scaled to

# Counter boxes at 640x480 (x0, y0, x1, y1), measured from real frames.
COUNTER_BOXES = {
    "green":  (34, 44, 95, 67),
    "red":    (34, 67, 95, 90),
    "yellow": (34, 90, 95, 113),
}

# Distance box (top-right, white digits). Used ONLY to harvest extra digit
# samples for the atlas -- not needed for the reward itself.
DISTANCE_BOX = (555, 76, 618, 98)

# HSV colour bands isolating each bright counter digit (OpenCV H 0-179).
COUNTER_HSV = {
    "green":  [((40, 120, 120), (85, 255, 255))],
    "red":    [((0, 120, 120), (10, 255, 255)), ((165, 120, 120), (179, 255, 255))],
    "yellow": [((20, 120, 120), (35, 255, 255))],
}


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------
def _ensure_size(frame):
    if frame.shape[1] != HUD_W or frame.shape[0] != HUD_H:
        frame = cv2.resize(frame, (HUD_W, HUD_H))
    return frame


def _digit_mask(frame, name):
    """Binary image (255 = digit pixel) for one counter box."""
    x0, y0, x1, y1 = COUNTER_BOXES[name]
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    mask = None
    for lo, hi in COUNTER_HSV[name]:
        m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def _white_digit_mask(frame):
    """Binary mask of the white distance digits (atlas-harvesting only)."""
    x0, y0, x1, y1 = DISTANCE_BOX
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, np.array((0, 0, 180), np.uint8),
                       np.array((179, 70, 255), np.uint8))


def _segment_digits(mask):
    """Split a binary digit-strip into individual digit crops, left to right.
    Returns a list of (x_left, crop) so callers can order them."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for i in range(1, n):                      # skip background label 0
        x, y, w, h, area = stats[i]
        if area < 6 or h < 6:                  # ignore speckle
            continue
        boxes.append((x, mask[y:y + h, x:x + w]))
    boxes.sort(key=lambda b: b[0])             # left -> right reading order
    return boxes


def _canon(crop):
    """Scale a digit crop to the canonical size for matching."""
    return cv2.resize(crop, DIGIT_SIZE, interpolation=cv2.INTER_NEAREST)


# --------------------------------------------------------------------------
# atlas (the labelled digit templates)
# --------------------------------------------------------------------------
def load_atlas(path=ATLAS_DIR):
    atlas = {}
    for d in range(10):
        p = os.path.join(path, f"{d}.png")
        if os.path.exists(p):
            atlas[d] = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    return atlas


def _match_digit(crop, atlas):
    """Return the digit (0-9) whose template best matches this crop."""
    c = _canon(crop).astype(np.float32)
    best_d, best_score = None, -1.0
    for d, tmpl in atlas.items():
        t = tmpl.astype(np.float32)
        # normalized correlation: 1.0 = identical, robust to brightness
        num = float((c * t).sum())
        den = float(np.linalg.norm(c) * np.linalg.norm(t)) + 1e-6
        score = num / den
        if score > best_score:
            best_score, best_d = score, d
    return best_d


def read_counts(frame, atlas=None):
    """Read (green, red, yellow) counts from a frame. Returns a dict; a value is
    None if its box was empty/unreadable (e.g. on the GAME OVER screen)."""
    frame = _ensure_size(frame)
    if atlas is None:
        atlas = load_atlas()
    out = {}
    for name in ("green", "red", "yellow"):
        mask = _digit_mask(frame, name)
        digits = _segment_digits(mask)
        if not digits or not atlas:
            out[name] = None
            continue
        value = 0
        for _, crop in digits:
            value = value * 10 + _match_digit(crop, atlas)
        out[name] = value
    return out


# --------------------------------------------------------------------------
# assisted labeller:  python hud.py --build
# --------------------------------------------------------------------------
def build_atlas(frames_glob="frames/*.png"):
    """Show digit crops harvested from captured frames; you press 0-9 to label
    each (or 's' to skip). Saves one template per digit to hud_digits/."""
    os.makedirs(ATLAS_DIR, exist_ok=True)
    have = set(load_atlas().keys())
    print(f"Building digit atlas in ./{ATLAS_DIR}/  (already have: {sorted(have)})")
    print("For each enlarged crop, press the matching DIGIT key, or 's' to skip, 'q' to quit.\n")

    files = sorted(glob.glob(frames_glob))
    if not files:
        print("No frames found -- run capture_frames.py first.")
        return

    for f in files:
        frame = _ensure_size(cv2.imread(f))
        # harvest from the 3 colour counters AND the white distance number
        sources = [_digit_mask(frame, n) for n in ("green", "red", "yellow")]
        sources.append(_white_digit_mask(frame))
        for mask in sources:
            for _, crop in _segment_digits(mask):
                if len(have) == 10:
                    print("All 10 digits captured. Done.")
                    return
                big = cv2.resize(_canon(crop), (180, 240), interpolation=cv2.INTER_NEAREST)
                cv2.imshow("label this digit (0-9, s=skip, q=quit)", big)
                k = cv2.waitKey(0) & 0xFF
                if k == ord("q"):
                    cv2.destroyAllWindows()
                    print(f"Stopped. Have digits: {sorted(have)}")
                    return
                if k == ord("s") or not (ord("0") <= k <= ord("9")):
                    continue
                d = k - ord("0")
                cv2.imwrite(os.path.join(ATLAS_DIR, f"{d}.png"), _canon(crop))
                have.add(d)
                print(f"  saved digit {d}   (have: {sorted(have)})")
    cv2.destroyAllWindows()
    print(f"Done. Digits captured: {sorted(have)}")


def _test_on_frames(frames_glob="frames/*.png"):
    atlas = load_atlas()
    if not atlas:
        print("No atlas yet. Run:  python hud.py --build")
        return
    for f in sorted(glob.glob(frames_glob))[:15]:
        counts = read_counts(cv2.imread(f), atlas)
        print(f"{os.path.basename(f)}: {counts}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--build":
        build_atlas()
    else:
        _test_on_frames()
