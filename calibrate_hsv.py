"""
calibrate_hsv.py — interactive HSV range tuning aid for perception.CONFIG.

perception.CONFIG's HSV ranges (token colors, road-lane surface, threat cars)
are placeholders that need calibrating against the live simulator. This tool
shows a saved frame cropped to the same ROI / working resolution perception.py
uses, lets you drag H/S/V min/max trackbars, and prints a CONFIG-ready
((H, S, V), (H, S, V)) tuple you can paste straight into perception.CONFIG.

Usage:
    python calibrate_hsv.py <frame.png> [roi_name]

    roi_name: one of 'road_roi', 'token_roi', 'rear_roi' (default: full frame)

Capture frames to tune against by running sample_drive.py / test_communication.py
and saving a frame from the live camera windows with cv2.imwrite(...).

Controls:
    p — print the current ((H,S,V),(H,S,V)) tuple to the console
    q — quit
"""
import sys

import cv2
import numpy as np

from perception import CONFIG, _prep_frame, _crop_roi

WINDOW = "HSV Calibration  (p = print range, q = quit)"
TRACKBARS = (
    ('H min', 0, 179), ('H max', 179, 179),
    ('S min', 0, 255), ('S max', 255, 255),
    ('V min', 0, 255), ('V max', 255, 255),
)


def _nothing(_value):
    pass


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    frame = cv2.imread(sys.argv[1])
    if frame is None:
        print(f"Could not read {sys.argv[1]}")
        sys.exit(1)

    roi_name = sys.argv[2] if len(sys.argv) > 2 else None
    if roi_name is not None and roi_name not in CONFIG:
        print(f"Unknown ROI '{roi_name}'. Expected one of: road_roi, token_roi, rear_roi")
        sys.exit(1)

    # Crop to the exact same coordinate space detect_tokens/detect_rear_events/
    # detect_current_lane operate on, so calibrated values transfer directly.
    small = _prep_frame(frame, CONFIG)
    roi_bgr = _crop_roi(small, CONFIG[roi_name]) if roi_name else small

    cv2.namedWindow(WINDOW)
    for name, init, max_val in TRACKBARS:
        cv2.createTrackbar(name, WINDOW, init, max_val, _nothing)

    print("Drag the trackbars until the mask (right half) cleanly isolates your target.")
    print("Press 'p' to print the CONFIG-ready HSV range tuple, 'q' to quit.")

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    while True:
        lower = np.array([cv2.getTrackbarPos(name, WINDOW) for name in ('H min', 'S min', 'V min')], dtype=np.uint8)
        upper = np.array([cv2.getTrackbarPos(name, WINDOW) for name in ('H max', 'S max', 'V max')], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imshow(WINDOW, np.hstack([roi_bgr, mask_bgr]))

        key = cv2.waitKey(30) & 0xFF
        if key == ord('p'):
            print(f"((  {lower[0]}, {lower[1]}, {lower[2]} ), ( {upper[0]}, {upper[1]}, {upper[2]} ))")
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
