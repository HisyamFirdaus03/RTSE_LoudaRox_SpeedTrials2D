"""
capture_frames.py
=================
Step 0 of the ML pipeline: collect real game frames to develop against.

Connects to the FRONT camera and saves frames to ./frames/ as PNG. We use these
to build + test hud.py (the reward reader) and features.py (the observation
extractor) offline, without needing the live game running every time.

    1. Start the game (and play a little, or let the heuristic drive).
    2. python capture_frames.py            # saves ~40 frames over ~20s
    3. Try to capture variety: dense token fields AND a GAME OVER screen
       (let the run end) so we can build the "episode done" detector too.

Nothing else may be connected to the front camera (:8080) while this runs.
"""

import os
import time
import socket
import cv2
import numpy as np

CAMERA_HOST = "127.0.0.1"
FRONT_CAMERA_PORT = 8080

OUT_DIR = "frames"
NUM_FRAMES = 40          # how many to save
INTERVAL = 0.5           # seconds between saves (spread them out for variety)


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
    os.makedirs(OUT_DIR, exist_ok=True)
    sock = connect(FRONT_CAMERA_PORT)

    print(f"Saving {NUM_FRAMES} frames to ./{OUT_DIR}/ (every {INTERVAL}s)...")
    print("Tip: let a run reach GAME OVER so we capture that screen too.\n")

    saved = 0
    next_save = time.time()
    last_shape = None
    while saved < NUM_FRAMES:
        frame = recv_frame(sock)
        if frame is None:
            continue
        last_shape = frame.shape
        now = time.time()
        if now >= next_save:
            path = os.path.join(OUT_DIR, f"frame_{saved:03d}.png")
            cv2.imwrite(path, frame)
            saved += 1
            next_save = now + INTERVAL
            print(f"  saved {path}  (frame size {frame.shape[1]}x{frame.shape[0]})")
        # keep the window responsive / let you see what's being captured
        cv2.imshow("capturing (front)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    sock.close()
    cv2.destroyAllWindows()
    print(f"\nDone. Saved {saved} frames to ./{OUT_DIR}/")
    if last_shape is not None:
        print(f"Frame resolution: {last_shape[1]}x{last_shape[0]} "
              f"(remember this -- HUD positions depend on it)")


if __name__ == "__main__":
    main()
