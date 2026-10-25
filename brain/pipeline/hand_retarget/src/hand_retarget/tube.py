"""The tube the human picks up: its size, and where it lies on the table when the demonstration starts.

The video gives no object pose, so the tube is put where the demonstrated grasp says it
must be: at the grasp frame (the hand closed around it), under the centre of the grasp,
lying on the table along the knuckle row. A hand-set correction moves it from there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import Replay
from .retargeters import human_joint_angles
from .skeleton import FINGERTIPS, hand_anchor, palm_frame, to_frame

# The grasp frame is the first frame where the hand is this closed, relative to its most closed.
GRASP_CLOSURE_FRACTION = 0.9


@dataclass(frozen=True)
class TubeSpec:
    """Estimated from the video (the A4 sheet as a ruler) and a typical pop tube."""

    length: float = 0.30  # m, end to end
    radius: float = 0.025  # m
    mass: float = 0.04  # kg
    friction: float = 1.0  # rubber


@dataclass
class TubeAdjustment:
    """What the user adds on top of the automatic tube pose."""

    x: float = 0.0  # m, world axes
    y: float = 0.0
    yaw: float = 0.0  # rad, about the vertical


@dataclass
class TubePlacement:
    spec: TubeSpec
    position: np.ndarray  # (3,) world, centre of the tube at the start
    yaw: float  # rad, heading of the tube's axis in the world
    grasp_frame: int
    grasp_centre_path: np.ndarray  # (T, 3) world, where the human grasp holds the tube over time
    goal: np.ndarray  # (3,) world, where the human leaves the tube: the grasp centre in the last frame


def hand_closure(skeleton: np.ndarray) -> float:
    """Mean MCP + PIP flexion (rad) of the four long fingers."""
    angles = human_joint_angles(to_frame(palm_frame(skeleton), skeleton))
    return float(angles[1:, 1:3].sum(axis=1).mean())


def grasp_frame(skeletons: np.ndarray) -> int:
    closure = np.array([hand_closure(s) for s in skeletons])
    return int(np.argmax(closure >= GRASP_CLOSURE_FRACTION * closure.max()))


def grasp_centre(skeleton: np.ndarray) -> np.ndarray:
    """Centre of a power grasp: the mean of the thumb, index, middle and ring tips and the knuckle row."""
    return np.vstack([skeleton[list(FINGERTIPS[:4])], hand_anchor(skeleton)]).mean(axis=0)


def place_tube(
    replay: Replay, table_top_z: float, adjustment: TubeAdjustment = TubeAdjustment(), spec: TubeSpec = TubeSpec()
) -> TubePlacement:
    skeletons = replay.stack("skeleton_world")
    frame = grasp_frame(skeletons)
    centres = np.array([grasp_centre(s) for s in skeletons])
    across = palm_frame(skeletons[frame])[:3, 1]  # knuckle row, pinky to index
    yaw = float(np.arctan2(across[1], across[0])) + adjustment.yaw
    position = np.array(
        [centres[frame, 0] + adjustment.x, centres[frame, 1] + adjustment.y, table_top_z + spec.radius]
    )
    return TubePlacement(
        spec=spec,
        position=position,
        yaw=yaw,
        grasp_frame=frame,
        grasp_centre_path=centres,
        goal=np.array([centres[-1, 0], centres[-1, 1], table_top_z + spec.radius]),
    )
