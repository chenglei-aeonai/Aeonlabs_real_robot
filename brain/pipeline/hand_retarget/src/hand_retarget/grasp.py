"""Grasp strengthening: closing the fingers a little more than the human did, over part of a demonstration.

The retargeted hand only reproduces the human's finger shape, which on the robot may
hold the object too loosely. A hand-set extra angle is added to the joint that closes
each finger at its base, between two video frames, and ramped in before the first and
out after the last so the fingers never close or open all of a sudden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hand_optimizer import MAX_JOINT_SPEED
from .robot import HAND_JOINT_NAMES, HAND_LOWER, HAND_UPPER

# The joint that closes each finger at its base, thumb to pinky.
GRASP_JOINTS = ("rj_dg_1_3", "rj_dg_2_2", "rj_dg_3_2", "rj_dg_4_2", "rj_dg_5_3")
GRASP_JOINT_INDEX = np.array([HAND_JOINT_NAMES.index(name) for name in GRASP_JOINTS])
DEFAULT_RAMP_FRAMES = 12  # half a second of a 24 fps video


@dataclass(frozen=True)
class GraspStrengthening:
    """How much more each finger closes, and over which video frames of the whole clip."""

    extra: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)  # rad, thumb to pinky
    first_frame: int = 0
    last_frame: int | None = None  # None: the last frame of the clip
    # The extra closure grows linearly over this many frames before the first frame, where it
    # is complete, and goes away over as many after the last.
    ramp_frames: int = DEFAULT_RAMP_FRAMES

    @property
    def active(self) -> bool:
        return any(self.extra)

    def envelope(self, num_frames: int) -> np.ndarray:
        """Per frame, the share (0 to 1) of the extra closure applied."""
        last = num_frames - 1 if self.last_frame is None else self.last_frame
        frames = np.arange(num_frames)
        if self.ramp_frames <= 0:
            return ((frames >= self.first_frame) & (frames <= last)).astype(float)
        ramp = self.ramp_frames
        return np.minimum(
            np.clip((frames - self.first_frame + ramp) / ramp, 0.0, 1.0), np.clip((last + ramp - frames) / ramp, 0.0, 1.0)
        )


def strengthen_grasp(hand_q: np.ndarray, grasp: GraspStrengthening, fps: float) -> np.ndarray:
    """Hand angles (T, 20) with the extra closure added, inside the joint limits and the joint speed limit."""
    if not grasp.active:
        return hand_q
    out = hand_q.copy()
    out[:, GRASP_JOINT_INDEX] += grasp.envelope(len(hand_q))[:, None] * np.asarray(grasp.extra)
    out = np.clip(out, HAND_LOWER, HAND_UPPER)
    max_step = MAX_JOINT_SPEED / fps
    for i in range(1, len(out)):
        out[i] = np.clip(out[i], out[i - 1] - max_step, out[i - 1] + max_step)
    return out
