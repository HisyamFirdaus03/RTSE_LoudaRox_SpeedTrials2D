"""
feedback.py  --  measurable feedback on how the agent is doing
==============================================================
Reads the live score off the HUD (hud.py) and appends it to run_log.csv, so a
run produces NUMBERS we can inspect (green/red/yellow over time) instead of
eyeballing screenshots.

Needs the digit atlas first (one-time):
      python hud.py --build

It only writes a row when a count CHANGES, so run_log.csv stays a clean timeline
of "what the agent scored and when". Also prints each change to the console.
"""

import os
import csv
import time
import hud


class RunLogger:
    def __init__(self, path="run_log.csv"):
        self.path = path
        self.atlas = hud.load_atlas()
        self.last = None
        new_file = not os.path.exists(path)
        self._f = open(path, "a", newline="")
        self._w = csv.writer(self._f)
        if new_file:
            self._w.writerow(["time", "green", "red", "yellow", "net(g-r)"])
            self._f.flush()
        if not self.atlas:
            print("[feedback] No digit atlas yet -> run:  python hud.py --build")
            print("[feedback] (score logging will stay blank until the atlas exists)")
        else:
            print(f"[feedback] logging score -> {path}  (atlas digits: {sorted(self.atlas)})")

    def update(self, frame):
        """Call ~1x/sec with the latest front frame."""
        if frame is None or not self.atlas:
            return
        c = hud.read_counts(frame, self.atlas)
        key = (c.get("green"), c.get("red"), c.get("yellow"))
        if key == self.last or all(v is None for v in key):
            return                                  # nothing new / unreadable
        g, r, y = key
        net = (g - r) if (g is not None and r is not None) else ""
        self._w.writerow([round(time.time(), 1), g, r, y, net])
        self._f.flush()
        self.last = key
        print(f"[score] green={g} red={r} yellow={y}  net={net}")

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass
