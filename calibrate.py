"""
calibrate.py
============
Live HSV tuner for SpeedTrials2D token colours. Run the game first, then run
this. It connects to the FRONT camera, shows the feed, and gives you six
trackbars (H/S/V min + max). Drag them until only one token colour stays white
in the "Mask" window, then copy the printed values into CONFIG in agent_policy.py.

    python calibrate.py

Keys:  p = print current HSV range   |   s = save current frame to frame.png   |   q = quit
"""

import socket
import time
import cv2
import numpy as np

CAMERA_HOST = "127.0.0.1"
FRONT_CAMERA_PORT = 8080


def connect(port):
    while True:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((CAMERA_HOST, port))
            print(f"Connected to camera on :{port}")
            return s
        except ConnectionRefusedError:
            print(f"Camera not ready on :{port}, retrying...")
            time.sleep(1)


def recv_frame(sock):
    """Read one length-prefixed JPEG. Returns None on timeout / no data so the
    caller can keep pumping the GUI event loop instead of freezing."""
    try:
        length_bytes = sock.recv(4)
    except socket.timeout:
        return None
    if not length_bytes or len(length_bytes) < 4:
        return None
    n = int.from_bytes(length_bytes, "little")
    buf = b""
    while len(buf) < n:
        try:
            pkt = sock.recv(n - len(buf))
        except socket.timeout:
            return None
        if not pkt:
            return None
        buf += pkt
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def _noop(_):
    pass


def main():
    sock = connect(FRONT_CAMERA_PORT)
    sock.settimeout(2.0)   # never block forever -> GUI stays responsive

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    for name, init in [("H min", 35), ("H max", 85),
                       ("S min", 80), ("S max", 255),
                       ("V min", 80), ("V max", 255)]:
        hi = 179 if name.startswith("H") else 255
        cv2.createTrackbar(name, "Controls", init, hi, _noop)

    last = None
    while True:
        frame = recv_frame(sock)
        if frame is not None:
            last = frame

        # Always render + pump the GUI event loop EVERY iteration, even if no
        # frame arrived this tick. This is what keeps the window responsive.
        if last is not None:
            hsv = cv2.cvtColor(last, cv2.COLOR_BGR2HSV)
            g = lambda n: cv2.getTrackbarPos(n, "Controls")
            lo = np.array([g("H min"), g("S min"), g("V min")], np.uint8)
            hi = np.array([g("H max"), g("S max"), g("V max")], np.uint8)
            mask = cv2.inRange(hsv, lo, hi)
            result = cv2.bitwise_and(last, last, mask=mask)
            cv2.imshow("Feed", last)
            cv2.imshow("Mask", mask)
            cv2.imshow("Result", result)
        else:
            # Waiting for the first frame -- show a placeholder so there's a
            # window to pump, and tell the user what's happening.
            placeholder = np.zeros((120, 480, 3), np.uint8)
            cv2.putText(placeholder, "Waiting for camera frames...", (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Feed", placeholder)
            lo = np.array([0, 0, 0]); hi = np.array([0, 0, 0])

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            print(f"(({lo[0]}, {lo[1]}, {lo[2]}), ({hi[0]}, {hi[1]}, {hi[2]})),")
        elif key == ord("s"):
            cv2.imwrite("frame.png", last)
            print("Saved frame.png")

    sock.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
