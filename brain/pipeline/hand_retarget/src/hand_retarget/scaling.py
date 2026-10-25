"""Hand scaling: where the robot's fingertips should be, given a skeleton in the palm frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .robot import RobotKinematics
from .skeleton import FINGER_BASES, FINGERTIPS, HumanDemonstration

ScalingMode = Literal["global", "per_finger"]


@dataclass(frozen=True)
class HandScaling:
    """Scale factors from one human demonstration's hand to the DG5F, thumb to pinky."""

    per_finger: np.ndarray  # (5,)
    overall: float
    robot_finger_bases: np.ndarray  # (5, 3) in the robot palm frame
    robot_hand_anchor: np.ndarray  # (3,) in the robot palm frame

    @classmethod
    def from_demonstration(cls, demo: HumanDemonstration, robot: RobotKinematics) -> "HandScaling":
        human = demo.finger_lengths()
        return cls(
            per_finger=robot.finger_lengths / human,
            overall=float(robot.finger_lengths.sum() / human.sum()),
            robot_finger_bases=robot.finger_bases_in_palm,
            robot_hand_anchor=robot.hand_anchor_in_palm,
        )

    def fingertip_targets(self, skeleton_in_palm: np.ndarray, mode: ScalingMode) -> np.ndarray:
        """Fingertip targets (5, 3) in the robot palm frame, thumb to pinky.

        ``skeleton_in_palm`` is the skeleton in the human palm frame (origin at the hand
        anchor), whose axes match the robot palm's.

        - global: every fingertip keeps its offset from the hand anchor, scaled by one factor.
          Touching fingertips stay touching.
        - per_finger: each fingertip keeps its offset from its own finger base, scaled by
          that finger's factor, and is hung from the robot's finger base. Fits each
          finger's reach better but can pull touching fingertips apart.
        """
        tips = skeleton_in_palm[list(FINGERTIPS)]
        if mode == "global":
            return self.robot_hand_anchor + self.overall * tips
        if mode == "per_finger":
            bases = skeleton_in_palm[list(FINGER_BASES)]
            return self.robot_finger_bases + self.per_finger[:, None] * (tips - bases)
        raise ValueError(f"unknown scaling mode {mode!r}")
