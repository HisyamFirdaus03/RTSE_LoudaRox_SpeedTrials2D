"""
capture_frames.py
=================
Step 0 of the ML pipeline: collect real game frames to develop against.

Connects to the game cameras and saves frames as PNG so we can build + test
vision code (hud.py, the token detector, the chasing-car / golden-lane handlers)
offline, without the live game running every time.

    FRONT camera (:8080) -> ./frames/        (tokens, HUD, police, golden-lane text)
    BACK  camera (:8082) -> ./frames_back/    (chasing cars approaching from behind)

Usage:
    python capture_frames.py                  # both cameras (default)
    python capture_frames.py --camera front   # front only
    python capture_frames.py --camera back    # back only (chasing-car frames)
    python capture_frames.py --num 80 --interval 0.3

Tips:
    - Capture variety: dense token fields AND a GAME OVER screen (let a run end).
    - To tune a specific event, TRIGGER it in-game during the capture window
      (e.g. let a car chase you while recording the back camera).
    - Press 'q' in any preview window to stop early.

Nothing else may be connected to the same camera port while this runs.
"""

import os
import sys
import time
import socket
import argparse
import cv2
import numpy as np

CAMERA_HOST = "127.0.0.1"

# name -> (port, output directory)
CAMERAS = {
    "front": (8080, "frames"),
    "back":  (8082, "frames_back"),
}

NUM_FRAMES = 40          # how many to save per camera
INTERVAL = 5         # seconds between saves (spread them out for variety)


def connect(name, port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((CAMERA_HOST, port))
            s.settimeout(2.0)
            print(f"Connected to {name} camera on :{port}")
            return s
        except ConnectionRefusedError:
            print(f"{name.capitalize()} camera not ready on :{port} -- is the game running? retrying...")
            time.sleep(1)


def recv_frame(sock):
    """Read one length-prefixed JPEG (same protocol as sample_drive.py)."""
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


def main():
    ap = argparse.ArgumentParser(description="Capture front/back game-camera frames.")
    ap.add_argument("--camera", choices=["front", "back", "both"], default="both",
                    help="which camera(s) to record (default: both)")
    ap.add_argument("--num", type=int, default=NUM_FRAMES,
                    help="frames to save per camera")
    ap.add_argument("--interval", type=float, default=INTERVAL,
                    help="seconds between saves")
    args = ap.parse_args()

    names = ["front", "back"] if args.camera == "both" else [args.camera]

    # Per-camera state: socket, out dir, save counter, next-save time, last shape.
    cams = {}
    for name in names:
        port, out_dir = CAMERAS[name]
        os.makedirs(out_dir, exist_ok=True)
        cams[name] = {
            "sock": connect(name, port),
            "out": out_dir,
            "saved": 0,
            "next_save": time.time(),
            "shape": None,
        }

    print(f"\nSaving {args.num} frames per camera (every {args.interval}s) "
          f"to: {', '.join(CAMERAS[n][1] for n in names)}/")
    print("Tip: trigger the event you want while recording. Press 'q' to stop early.\n")

    try:
        while any(c["saved"] < args.num for c in cams.values()):
            for name, c in cams.items():
                if c["saved"] >= args.num:
                    continue
                frame = recv_frame(c["sock"])
                if frame is None:
                    continue
                c["shape"] = frame.shape
                now = time.time()
                if now >= c["next_save"]:
                    path = os.path.join(c["out"], f"frame_{c['saved']:03d}.png")
                    cv2.imwrite(path, frame)
                    c["saved"] += 1
                    c["next_save"] = now + args.interval
                    print(f"  [{name}] saved {path}  ({frame.shape[1]}x{frame.shape[0]})")
                cv2.imshow(f"capturing ({name})", frame)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        for c in cams.values():
            c["sock"].close()
        cv2.destroyAllWindows()

    print("\nDone.")
    for name, c in cams.items():
        msg = f"  [{name}] saved {c['saved']} frames to ./{c['out']}/"
        if c["shape"] is not None:
            msg += f"  (resolution {c['shape'][1]}x{c['shape'][0]})"
        print(msg)
    print("Remember the resolution -- HUD digit positions in hud.py assume 640x480.")


if __name__ == "__main__":
    main()
