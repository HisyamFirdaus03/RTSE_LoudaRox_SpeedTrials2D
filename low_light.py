"""
low_light.py
============
Challenge 1 (Low Light) handler for SpeedTrials2D, kept completely separate from
the vision brain (`agent_policy.py`) and the networking / scheduler scaffold
(`sample_drive.py`).

Challenge 1: the screen brightness drops and all tokens become "unknown" once
during the first 10s of the game. The agent must detect the change and send
`acceleration_input = -1.0` to recover the light. While the light is off every
control sent is applied at -10% speed.

This component is a thin post-processing override that sits between the policy
and the control output:

    (steering, acceleration) from policy  ->  LowLightController.apply()  ->  (steering, acceleration)

While dark it overrides ONLY the acceleration (-> RECOVER_ACCEL); steering is
passed through untouched so the policy keeps steering the road. Once brightness
recovers, the policy's normal output flows through unchanged.
"""

import cv2
import numpy as np

from event_manager import Override

# Tunables -- calibrate on real frames if needed.
DARK_THRESHOLD = 0.15   # mean brightness (0..1) below this => light is OFF (matches old code)
RECOVER_ACCEL  = -1.0   # acceleration sent while dark to recover the light


def get_brightness(frame):
    """Mean grayscale brightness of a BGR frame, normalised to 0..1.
    Ported verbatim from sample_drive_old.py."""
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0


class LowLightController:
    """Detects the blackout via an absolute brightness threshold and, while dark,
    overrides acceleration to RECOVER_ACCEL so the game restores the light.
    Steering is passed through untouched (caller keeps the policy's steering)."""

    def __init__(self, dark_threshold=DARK_THRESHOLD, recover_accel=RECOVER_ACCEL):
        self.dark_threshold = dark_threshold
        self.recover_accel  = recover_accel
        self.is_dark = False
        self.brightness = 1.0

    def apply(self, front_frame, steering, acceleration):
        """Take the policy's (steering, acceleration); return possibly-overridden
        (steering, acceleration). While dark, acceleration -> recover_accel."""
        if front_frame is None:
            return steering, acceleration
        self.brightness = get_brightness(front_frame)
        self.is_dark = self.brightness < self.dark_threshold
        if self.is_dark:
            return steering, self.recover_accel   # steering unchanged; reverse to recover light
        return steering, acceleration

    # -- arbiter interface (used by EventManager) ------------------------
    def evaluate(self, front_frame, back_frame, ctx):
        """While dark, bid to override acceleration to recover the light (steering
        passes through). Priority 80: a collision dodge (100) still beats this, so
        we never brake into a car closing from behind; we brake once clear."""
        if front_frame is None:
            return None
        self.brightness = get_brightness(front_frame)
        self.is_dark = self.brightness < self.dark_threshold
        if self.is_dark:
            ctx.passed.add("darkness")   # sending accel=-1.0 while dark passes the event
            return Override(ctx.base_steer, self.recover_accel, 80, "DARK")
        return None
