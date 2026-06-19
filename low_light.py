"""
low_light.py
=============
EV1 / Challenge 1 (Darkness) event component.

Brightness drops and all tokens become "unknown" once early in the run. While
dark, the agent must send acceleration_input = -1.0 to recover the light;
steering can't trust stale token detections, so it decays toward straight
instead. The -10% speed penalty applied while dark is enforced by the game
itself, not simulated here.
"""

import cv2
import numpy as np

import debug_view

CONFIG = {
    "dark_threshold": 0.15,    # mean brightness (0..1) below which the light is "off"; ASSUMED, tune against a real blackout frame
    "recover_accel": -1.0,
    "steer_decay_alpha": 0.3,  # low-pass: new = alpha*0 + (1-alpha)*prev, matches navigation's smoothing
    "debug_overlay": True,
}


def get_brightness(frame):
    """Mean grayscale brightness of a BGR frame, normalized to 0..1."""
    return float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) / 255.0


def is_dark(brightness):
    return brightness < CONFIG["dark_threshold"]


def handle(frame, brightness, prev_steer):
    """While dark: decay steering toward straight, force accel=-1.0 to recover the light."""
    alpha = CONFIG["steer_decay_alpha"]
    steer = max(-1.0, min(1.0, (1 - alpha) * prev_steer))
    acceleration = CONFIG["recover_accel"]

    if CONFIG["debug_overlay"]:
        debug_view.draw_dark_overlay(frame, brightness, steer, acceleration)

    return steer, acceleration
