"""
lanes.py
========
A 5-lane spatial model for a CAR-FIXED front camera (the camera follows the car,
so the car is ALWAYS at the horizontal center of its own frame; the road scrolls
underneath it). Used by the Golden Lane handler (EV5) and chasing-car evade
(EV3/EV4).

Because the car never moves within the frame, you cannot read your lane from the
car's pixel position. Instead we read it from where the ROAD sits relative to
frame-center:
  - center lane -> road is symmetric about center
  - leftmost lane (1) -> almost all road is to your RIGHT
  - rightmost lane (5) -> almost all road is to your LEFT

So each frame we detect the road's left/right edges in a near-field band (reusing
the same grey-asphalt threshold as agent_policy), split that span into 5 lanes,
and ask which fifth the frame-center falls in -> that's the current lane.

`LaneModel.update(frame)` refreshes the detected span; then:
    current_lane()      -> 1..5   the lane the car is in right now
    target_x(n)         -> px      where lane n currently sits in the frame
    steer_to_lane(n)    -> [-1,1]  steering to bring lane n under the car

VERIFY in-game (needs one captured multi-lane frame): the near-field band and the
road threshold, and that all 5 lanes are actually visible. If the road runs off
the frame edge the span is truncated -- tune `band_*` / `min_frac` then.
"""

import cv2
import numpy as np

from agent_policy import CONFIG

N_LANES = 5


class LaneModel:
    """Per-frame estimate of the road span and the car's current lane."""

    def __init__(self, band_lo=0.60, band_hi=0.80, min_frac=0.12):
        self.band_lo = band_lo      # near-field strip (fractions of H) used to find edges
        self.band_hi = band_hi
        self.min_frac = min_frac    # a column counts as "road" if this frac of the strip is road
        self.w = None
        self.span = None            # (left_x, right_x) in pixels, or None if not found
        self.detected = False

    # -- per-frame update ----------------------------------------------
    def update(self, frame):
        if frame is None:
            return self
        h, w = frame.shape[:2]
        self.w = w
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        road = cv2.inRange(
            hsv,
            np.array((0, 0, CONFIG["road_val_min"]), np.uint8),
            np.array((179, CONFIG["road_sat_max"], 255), np.uint8),
        )
        y0, y1 = int(h * self.band_lo), int(h * self.band_hi)
        strip = road[y0:y1, :]
        col_road = (strip > 0).sum(axis=0)              # road pixels per column
        flags = col_road > self.min_frac * (y1 - y0)
        # Take the LARGEST CONTIGUOUS run of road columns, not the absolute
        # leftmost/rightmost: the road is one wide block, while the grey
        # streetlight poles beside it are isolated thin runs that would
        # otherwise wreck the span. Small gaps (dashes/shadows) are bridged.
        run = self._longest_run(flags, max_gap=int(0.02 * w))
        if run is not None and (run[1] - run[0]) > 0.2 * w:
            self.span = (float(run[0]), float(run[1]))
            self.detected = True
        else:
            self.span = None                             # fall back to a centered default
            self.detected = False
        return self

    @staticmethod
    def _longest_run(flags, max_gap=12):
        """Longest run of True in `flags`, bridging gaps up to `max_gap`. Returns
        (start_idx, end_idx) or None."""
        idx = np.where(flags)[0]
        if len(idx) == 0:
            return None
        groups = []
        start = prev = idx[0]
        for x in idx[1:]:
            if x - prev <= max_gap:
                prev = x
            else:
                groups.append((start, prev))
                start = prev = x
        groups.append((start, prev))
        return max(groups, key=lambda g: g[1] - g[0])

    # -- queries -------------------------------------------------------
    def _span(self):
        if self.span is not None:
            return self.span
        w = self.w or 640                                # centered full-width default
        cx = w * CONFIG["center_x_frac"]
        bhw = w * CONFIG["roi_bot_half_w"]
        return cx - bhw, cx + bhw

    def _center_x(self):
        return (self.w or 640) * CONFIG["center_x_frac"]

    def lane_width(self):
        left, right = self._span()
        return (right - left) / N_LANES

    def lane_of_x(self, x):
        """Which lane (1..5) the pixel-x falls in, given the current road span."""
        left, _ = self._span()
        idx = int((x - left) // self.lane_width()) + 1
        return max(1, min(N_LANES, idx))

    def current_lane(self):
        """The lane the car occupies = the lane the frame-center sits in."""
        return self.lane_of_x(self._center_x())

    def target_x(self, n):
        """Where lane n currently sits in the frame (its center x)."""
        n = max(1, min(N_LANES, int(n)))
        left, _ = self._span()
        return left + (n - 0.5) * self.lane_width()

    def steer_to_lane(self, n, gain=2.2):
        """P-control steering in [-1,1] to bring lane n under the car. As we move,
        the road scrolls so lane n drifts toward center; when it's centered, we're
        in lane n."""
        err = (self.target_x(n) - self._center_x()) / (0.5 * (self.w or 640))
        return max(-1.0, min(1.0, gain * err))
