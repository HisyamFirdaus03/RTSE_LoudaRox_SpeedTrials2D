"""
debug_view.py
=============
Shared visualization helpers. Each decision/event component calls into this to
draw its own debug overlay window; disable per-component via that component's
own CONFIG["debug_overlay"] flag.
"""

import cv2
import numpy as np


def draw_dark_overlay(frame, brightness, steer, acceleration):
    try:
        vis = frame.copy()
        cv2.putText(vis, f"DARK brightness={brightness:.3f} steer={steer:+.2f} accel={acceleration:+.2f}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        cv2.imshow("Policy Debug", vis)
        cv2.waitKey(1)
    except Exception:
        pass


def draw_debug_overlay(frame, roi_top, roi_bottom, top_half_width_px, bottom_half_width_px,
                        green_blobs, red_blobs, yellow_blobs, target, hazard, steer, acceleration):
    try:
        vis = frame.copy()
        h, w = vis.shape[:2]
        cx = w // 2

        # Trapezoid road ROI outline (narrow at horizon, wide near the car).
        pts = np.array([
            [cx - top_half_width_px, roi_top],
            [cx + top_half_width_px, roi_top],
            [cx + bottom_half_width_px, roi_bottom],
            [cx - bottom_half_width_px, roi_bottom],
        ], np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 255), 1)

        colors = {"green": (0, 255, 0), "red": (0, 0, 255), "yellow": (0, 255, 255)}
        for name, blobs in (("green", green_blobs), ("red", red_blobs), ("yellow", yellow_blobs)):
            for bcx, bcy, _ in blobs:
                cv2.circle(vis, (int(bcx), int(bcy)), 8, colors[name], 2)

        if target is not None:
            cv2.drawMarker(vis, (int(target[0]), int(target[1])), (0, 255, 0),
                            cv2.MARKER_CROSS, 16, 2)
        if hazard is not None:
            cv2.drawMarker(vis, (int(hazard[0]), int(hazard[1])), (0, 0, 255),
                            cv2.MARKER_TILTED_CROSS, 16, 2)

        tip = int(cx + steer * w * 0.25)
        cv2.arrowedLine(vis, (cx, h - 5), (tip, h - 40), (255, 0, 255), 3, tipLength=0.3)
        cv2.putText(vis, f"steer={steer:+.2f} accel={acceleration:+.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Policy Debug", vis)
        cv2.waitKey(1)
    except Exception:
        pass
