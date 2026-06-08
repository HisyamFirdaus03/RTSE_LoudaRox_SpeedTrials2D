"""
auto_calibrate.py
=================
Hands-off HSV calibration for SpeedTrials2D tokens. No trackbars.

How it works: tokens are the only VIVID (high-saturation) blobs on the grey
road, and we know their rough hue identity (green~60, yellow~27, red~0/179).
This script watches the live front camera for a few seconds, collects the vivid
road pixels near each known hue, and derives a tight HSV range from percentiles.
It writes the result to `color_calibration.json`, which agent_policy.py loads
automatically -- so after running this once, sample_drive.py just works.

    1. Start the game so tokens are visible on the road.
    2. python auto_calibrate.py        (let some green/red/yellow tokens pass by)
    3. python sample_drive.py          (it auto-loads color_calibration.json)

Make sure NOTHING else is connected to the front camera (:8080) while this runs.
"""

import socket
import json
import time
import cv2
import numpy as np

from agent_policy import _roi_polygon, CONFIG

CAMERA_HOST = "127.0.0.1"
FRONT_CAMERA_PORT = 8080

CAPTURE_SECONDS = 8.0     # how long to watch the feed
VIVID_S = 70              # a "token" pixel is at least this saturated...
VIVID_V = 70              # ...and at least this bright (excludes grey road / shadow)
MIN_SAMPLES = 400         # need at least this many pixels to trust a colour
SUBSAMPLE = 3000          # cap pixels kept per frame per colour (bounds memory)

# Known hue identity of each token type (OpenCV H, 0-179). Red wraps 0/180.
BANDS = {
    "green":  [(33, 92)],
    "yellow": [(16, 34)],
    "red":    [(0, 13), (160, 179)],
}


def connect(port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((CAMERA_HOST, port))
            s.settimeout(2.0)
            print(f"Connected to front camera on :{port}")
            return s
        except ConnectionRefusedError:
            print(f"Camera not ready on :{port} -- is the game running? retrying...")
            time.sleep(1)


def recv_frame(sock):
    try:
        length_bytes = sock.recv(4)
        if not length_bytes or len(length_bytes) < 4:
            return None
        n = int.from_bytes(length_bytes, "little")
        buf = b""
        while len(buf) < n:
            pkt = sock.recv(n - len(buf))
            if not pkt:
                return None
            buf += pkt
        return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    except socket.timeout:
        return None


def _subsample(arr, n):
    if len(arr) <= n:
        return arr
    idx = np.random.choice(len(arr), n, replace=False)
    return arr[idx]


def _range_from_samples(samples, pad_h=6):
    """Percentile-based HSV box from a set of pixels (Nx3, uint8)."""
    if len(samples) < MIN_SAMPLES:
        return None
    H, S, V = samples[:, 0], samples[:, 1], samples[:, 2]
    h_lo = int(max(0,   np.percentile(H, 2)  - pad_h))
    h_hi = int(min(179, np.percentile(H, 98) + pad_h))
    s_lo = int(max(50,  np.percentile(S, 5)  - 15))
    v_lo = int(max(50,  np.percentile(V, 5)  - 15))
    return [[h_lo, s_lo, v_lo], [h_hi, 255, 255]]


def derive_ranges(pools):
    """Turn collected pixel pools into CONFIG-style ranges, with fallback."""
    out = {}
    for color, bands in BANDS.items():
        pool = pools[color]
        if not pool:
            print(f"  {color:6s}: no samples -> keeping default")
            out[color] = [[list(lo), list(hi)] for lo, hi in CONFIG[color]]
            continue
        samples = np.concatenate(pool, axis=0)

        if color == "red":
            # Red wraps hue 0/180 -> split low vs high and emit up to two ranges.
            low = samples[samples[:, 0] < 90]
            high = samples[samples[:, 0] >= 90]
            ranges = []
            r_low = _range_from_samples(low)
            r_high = _range_from_samples(high)
            if r_low:
                ranges.append(r_low)
            if r_high:
                ranges.append(r_high)
            if not ranges:
                print(f"  {color:6s}: too few samples -> keeping default")
                ranges = [[list(lo), list(hi)] for lo, hi in CONFIG[color]]
            else:
                print(f"  {color:6s}: {len(samples)} px -> {ranges}")
            out[color] = ranges
        else:
            r = _range_from_samples(samples)
            if r is None:
                print(f"  {color:6s}: too few samples ({len(samples)}) -> keeping default")
                out[color] = [[list(lo), list(hi)] for lo, hi in CONFIG[color]]
            else:
                print(f"  {color:6s}: {len(samples)} px -> {[r]}")
                out[color] = [r]
    return out


def main():
    sock = connect(FRONT_CAMERA_PORT)
    pools = {c: [] for c in BANDS}

    roi_poly = None
    print(f"\nCapturing for {CAPTURE_SECONDS:.0f}s -- drive around so green/red/yellow "
          f"tokens pass through view...\n")
    t_end = time.time() + CAPTURE_SECONDS
    frames = 0

    while time.time() < t_end:
        frame = recv_frame(sock)
        if frame is None:
            continue
        frames += 1
        h, w = frame.shape[:2]
        if roi_poly is None:
            roi_poly = _roi_polygon(w, h)
            roi_mask = np.zeros((h, w), np.uint8)
            cv2.fillPoly(roi_mask, [roi_poly], 255)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # vivid pixels inside the road ROI
        vivid = cv2.inRange(hsv, (0, VIVID_S, VIVID_V), (179, 255, 255))
        vivid = cv2.bitwise_and(vivid, roi_mask)

        for color, bands in BANDS.items():
            band_mask = None
            for lo_h, hi_h in bands:
                m = cv2.inRange(hsv, (lo_h, 0, 0), (hi_h, 255, 255))
                band_mask = m if band_mask is None else cv2.bitwise_or(band_mask, m)
            sel = cv2.bitwise_and(vivid, band_mask)
            px = hsv[sel > 0]
            if len(px):
                pools[color].append(_subsample(px, SUBSAMPLE))

        # live preview so you can see tokens are being captured
        prev = frame.copy()
        cv2.polylines(prev, [roi_poly], True, (255, 255, 255), 1)
        cv2.putText(prev, f"capturing... {int(t_end - time.time())}s",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("auto_calibrate (front)", prev)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    sock.close()
    print(f"\nCaptured {frames} frames. Deriving colour ranges:")
    ranges = derive_ranges(pools)

    with open("color_calibration.json", "w") as f:
        json.dump(ranges, f, indent=2)
    print("\nSaved -> color_calibration.json")
    print("agent_policy.py will load this automatically. Run sample_drive.py next.")

    # Final visual check: show the derived masks on a live-ish frame.
    print("Showing derived masks on the last frame. Press any key to close.")
    if roi_poly is not None and frames:
        last_hsv = hsv
        for color in BANDS:
            m = None
            for lo, hi in ranges[color]:
                mm = cv2.inRange(last_hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
                m = mm if m is None else cv2.bitwise_or(m, mm)
            cv2.imshow(f"mask: {color}", m)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
