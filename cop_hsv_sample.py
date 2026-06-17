"""
cop_hsv_sample.py
=================
Data-driven HSV calibration for the police car -- "start from the basics".

police_car.py detects the cop with *guessed* HSV bands for its blue livery and
red lights/panels. This tool replaces the guess with measurement, sampling the
actual extracted sprite(s):

    python cop_hsv_sample.py asset/36254_cop1.png asset/36254_cop2.png

The sprites are PNGs with an alpha channel, so we sample EXACTLY the opaque
(non-transparent) pixels -- those pixels ARE the cop, no bounding box needed.
Pass several sprites and they're pooled so the bands cover every frame.

Output: copy-paste-ready HSV bands for
    PCFG["blue"]   (police_car.py)
    CONFIG["red"]  (agent_policy.py)
plus a magnified preview so you can confirm the blue/red buckets landed on the
livery and lights, not the windows or tyres.

WHY NOT the usual `inRange(guess) -> report min/max` trick:
    Filtering by a guessed range and then reporting that range's min/max is
    circular -- it can only echo back the bounds you fed in, and min/max is set
    by a single antialiased edge pixel. Here we isolate the cop by ALPHA, split
    pixels into red-ish vs blue-ish by hue, and report ROBUST PERCENTILES of the
    real distribution -- which finds the true spread even outside today's bands.

Flags:
    --pad N      percent trimmed from each tail (default 2 = keep 2nd..98th pct)
    --alpha N    min alpha to count a pixel as opaque cop (default 128)
    --no-preview skip the GUI window (e.g. headless / scripted use)
"""

import argparse
import sys

import cv2
import numpy as np


# OpenCV HSV: H 0-179, S 0-255, V 0-255.
# Coarse hue gates used ONLY to bucket opaque pixels into "blue-ish" vs
# "red-ish". Deliberately wide -- the measured percentiles do the real
# tightening; these just decide which bucket each pixel falls in.
BLUE_HUE = (85, 140)          # cyan -> blue -> violet
RED_HUE_LOW = (0, 18)         # red near hue 0
RED_HUE_HIGH = (160, 179)     # red wrapping past 179
# Below these a pixel is treated as dark/grey structure (windows, tyres, trim)
# rather than coloured livery, so it doesn't drag the S/V floors down.
MIN_S = 40
MIN_V = 40


def load_opaque_hsv(path, min_alpha):
    """Return the Nx3 HSV pixels of one sprite's opaque region."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read image: {path}")
    if img.shape[2] == 4:
        bgr = img[:, :, :3]
        opaque = img[:, :, 3] >= min_alpha
    else:                                  # no alpha -> use the whole image
        bgr = img
        opaque = np.ones(img.shape[:2], bool)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv[opaque], bgr, hsv, opaque


def percentile_sv(pixels, pad):
    """Robust S/V floor and ceiling over the central (100-2*pad)% of pixels."""
    lo, hi = pad, 100 - pad
    s_lo, v_lo = int(np.percentile(pixels[:, 1], lo)), int(np.percentile(pixels[:, 2], lo))
    s_hi, v_hi = int(np.percentile(pixels[:, 1], hi)), int(np.percentile(pixels[:, 2], hi))
    return s_lo, v_lo, s_hi, v_hi


def fmt(lo, hi):
    return f"(({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]}))"


def report(blue, red, pad):
    print(f"\nPooled coloured pixels: {len(blue)} blue-ish / {len(red)} red-ish "
          f"(percentile pad = {pad}%).\n")

    # ---- BLUE: one contiguous hue band -> single (lo, hi) ------------------
    if len(blue) < 20:
        print("WARNING: very few blue pixels -- check the sprite/alpha.\n")
    else:
        lo, hi = pad, 100 - pad
        s_lo, v_lo, s_hi, v_hi = percentile_sv(blue, pad)
        h_lo = int(np.percentile(blue[:, 0], lo))
        h_hi = int(np.percentile(blue[:, 0], hi))
        print("PCFG[\"blue\"]   (police_car.py):")
        print(f'    "blue": [{fmt((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi))}],\n')

    # ---- RED: hue wraps 0/179 -> up to two bands sharing one S/V floor -----
    if len(red) < 20:
        print("WARNING: very few red pixels -- try a clearer/larger sprite.\n")
        return
    s_lo, v_lo, s_hi, v_hi = percentile_sv(red, pad)
    rh = red[:, 0]
    low = rh[rh <= RED_HUE_LOW[1]]
    high = rh[rh >= RED_HUE_HIGH[0]]
    bands = []
    if len(low):
        h_hi = int(np.percentile(low, 100 - pad))
        bands.append(fmt((0, s_lo, v_lo), (h_hi, s_hi, v_hi)) + ",   # red near hue 0")
    if len(high):
        h_lo = int(np.percentile(high, pad))
        bands.append(fmt((h_lo, s_lo, v_lo), (179, s_hi, v_hi)) + ",   # red wrapping past 179")
    print("CONFIG[\"red\"]  (agent_policy.py):")
    print('    "red": [' + "\n            ".join(bands) + "],\n")


def bucket(hsv_pixels):
    """Split Nx3 HSV pixels into (blue-ish, red-ish), dropping dark/grey ones."""
    h, s, v = hsv_pixels[:, 0], hsv_pixels[:, 1], hsv_pixels[:, 2]
    coloured = (s >= MIN_S) & (v >= MIN_V)
    blue = hsv_pixels[coloured & (h >= BLUE_HUE[0]) & (h <= BLUE_HUE[1])]
    red = hsv_pixels[coloured & (
        ((h >= RED_HUE_LOW[0]) & (h <= RED_HUE_LOW[1])) |
        ((h >= RED_HUE_HIGH[0]) & (h <= RED_HUE_HIGH[1])))]
    return blue, red


def preview(paths, min_alpha):
    """Magnified strip: each sprite next to its blue/red bucket highlighting."""
    tiles = []
    for p in paths:
        _, bgr, hsv, opaque = load_opaque_hsv(p, min_alpha)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        coloured = opaque & (s >= MIN_S) & (v >= MIN_V)
        blue_m = coloured & (h >= BLUE_HUE[0]) & (h <= BLUE_HUE[1])
        red_m = coloured & (
            ((h >= RED_HUE_LOW[0]) & (h <= RED_HUE_LOW[1])) |
            ((h >= RED_HUE_HIGH[0]) & (h <= RED_HUE_HIGH[1])))
        vis = (bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis[~opaque] = (0, 0, 0)
        vis[blue_m] = (255, 0, 0)
        vis[red_m] = (0, 0, 255)
        tiles.append(np.hstack([bgr, np.full((bgr.shape[0], 4, 3), 60, np.uint8), vis]))
    strip = np.vstack([t for pair in
                       [(t, np.full((4, t.shape[1], 3), 60, np.uint8)) for t in tiles]
                       for t in pair][:-1])
    strip = cv2.resize(strip, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
    cv2.imshow("sprite | sampled (B=blue R=red) -- any key to close", strip)
    print("Preview open -- confirm blue=livery, red=lights/panels, then press a key.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Sample the cop sprite's true HSV bands.")
    ap.add_argument("images", nargs="+", help="cop sprite PNG(s) with alpha")
    ap.add_argument("--pad", type=int, default=2,
                    help="percent trimmed from each tail (default 2 = 2nd..98th pct)")
    ap.add_argument("--alpha", type=int, default=128,
                    help="min alpha to count a pixel as opaque cop (default 128)")
    ap.add_argument("--no-preview", action="store_true", help="skip the GUI window")
    args = ap.parse_args()

    pooled = np.vstack([load_opaque_hsv(p, args.alpha)[0] for p in args.images])
    blue, red = bucket(pooled)
    report(blue, red, args.pad)

    if not args.no_preview:
        try:
            preview(args.images, args.alpha)
        except cv2.error:
            print("(GUI unavailable -- skipping preview.)")


if __name__ == "__main__":
    main()
