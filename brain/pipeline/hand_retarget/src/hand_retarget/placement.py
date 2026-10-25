"""Demonstration placement: the one rigid transform from the demonstration frame to the robot's world."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grasp import DEFAULT_RAMP_FRAMES, GraspStrengthening
from .robot import HOME_ARM_Q, RobotKinematics
from .skeleton import HumanDemonstration, hand_anchor, palm_frame


@dataclass
class PlacementAdjustment:
    """What the user adds on top of the default placement, about the first frame's hand anchor."""

    x: float = 0.0  # metres, world axes
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0  # radians, about world z, then y, then x
    pitch: float = 0.0
    roll: float = 0.0


def _rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return rz @ ry @ rx


def _heading(direction: np.ndarray) -> float:
    return float(np.arctan2(direction[1], direction[0]))


def default_placement(demo: HumanDemonstration, robot: RobotKinematics) -> np.ndarray:
    """Place the first frame's hand where the robot's hand is at home.

    The first frame's hand anchor goes onto the robot's home hand anchor, and the human
    fingers' level heading is turned onto the robot's.
    """
    first = demo.skeletons[0]
    home_palm = robot.palm_pose(HOME_ARM_Q)
    robot_anchor = home_palm[:3, :3] @ robot.hand_anchor_in_palm + home_palm[:3, 3]
    yaw = _heading(home_palm[:3, 2]) - _heading(palm_frame(first)[:3, 2])
    placement = np.eye(4)
    placement[:3, :3] = _rotation(yaw, 0.0, 0.0)
    anchor = placement[:3, :3] @ hand_anchor(first)
    placement[:3, 3] = robot_anchor - anchor
    return placement


def adjusted_placement(default: np.ndarray, first_skeleton: np.ndarray, adjustment: PlacementAdjustment) -> np.ndarray:
    """Apply the user's adjustment to the default placement, rotating about the first frame's hand anchor."""
    pivot = default[:3, :3] @ hand_anchor(first_skeleton) + default[:3, 3]
    rotation = _rotation(adjustment.yaw, adjustment.pitch, adjustment.roll)
    adjust = np.eye(4)
    adjust[:3, :3] = rotation
    adjust[:3, 3] = pivot - rotation @ pivot + np.array([adjustment.x, adjustment.y, adjustment.z])
    return adjust @ default


def apply(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform points (..., 3) by a 4x4."""
    return points @ transform[:3, :3].T + transform[:3, 3]


@dataclass
class SavedPlacement:
    """A human demonstration's own starting placement and table height, tuned by hand in the viewer."""

    adjustment: PlacementAdjustment = field(default_factory=PlacementAdjustment)
    table_offset: float = 0.0  # m, table top above simtoolreal's height
    tube_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)  # m, m, rad: added to the automatic tube pose
    first_frame: int = 0  # the robot demonstration keeps the video frames from this one...
    last_frame: int | None = None  # ...to this one; None is the last of the clip
    grasp: GraspStrengthening = field(default_factory=GraspStrengthening)


def placement_file(csv_path: str | Path) -> Path:
    """Where a demonstration's saved placement lives: next to its CSV."""
    csv_path = Path(csv_path)
    return csv_path.with_name(csv_path.name.removesuffix("_world_joints.csv") + "_placement.json")


def load_placement(csv_path: str | Path) -> SavedPlacement:
    path = placement_file(csv_path)
    if not path.exists():
        return SavedPlacement()
    v = json.loads(path.read_text())
    return SavedPlacement(
        adjustment=PlacementAdjustment(
            x=v["x_cm"] / 100, y=v["y_cm"] / 100, z=v["z_cm"] / 100,
            yaw=np.deg2rad(v["yaw_deg"]), pitch=np.deg2rad(v["pitch_deg"]), roll=np.deg2rad(v["roll_deg"]),
        ),
        table_offset=v.get("table_cm", 0.0) / 100,
        tube_offset=(v.get("tube_x_cm", 0.0) / 100, v.get("tube_y_cm", 0.0) / 100, np.deg2rad(v.get("tube_yaw_deg", 0.0))),
        first_frame=v.get("first_frame", 0),
        last_frame=v.get("last_frame"),
        grasp=GraspStrengthening(
            extra=tuple(float(np.deg2rad(a)) for a in v.get("grasp_extra_deg", (0.0,) * 5)),
            first_frame=v.get("grasp_first_frame", 0),
            last_frame=v.get("grasp_last_frame"),
            ramp_frames=v.get("grasp_ramp_frames", DEFAULT_RAMP_FRAMES),
        ),
    )


def save_placement(csv_path: str | Path, saved: SavedPlacement) -> Path:
    a = saved.adjustment
    values = {
        "x_cm": round(a.x * 100, 3), "y_cm": round(a.y * 100, 3), "z_cm": round(a.z * 100, 3),
        "yaw_deg": round(float(np.degrees(a.yaw)), 3), "pitch_deg": round(float(np.degrees(a.pitch)), 3),
        "roll_deg": round(float(np.degrees(a.roll)), 3), "table_cm": round(saved.table_offset * 100, 3),
        "tube_x_cm": round(saved.tube_offset[0] * 100, 3), "tube_y_cm": round(saved.tube_offset[1] * 100, 3),
        "tube_yaw_deg": round(float(np.degrees(saved.tube_offset[2])), 3),
    }
    if saved.first_frame:
        values["first_frame"] = saved.first_frame
    if saved.last_frame is not None:
        values["last_frame"] = saved.last_frame
    if saved.grasp.active:
        values["grasp_extra_deg"] = [round(float(np.degrees(a)), 3) for a in saved.grasp.extra]  # thumb to pinky
        values["grasp_first_frame"] = saved.grasp.first_frame
        values["grasp_ramp_frames"] = saved.grasp.ramp_frames
        if saved.grasp.last_frame is not None:
            values["grasp_last_frame"] = saved.grasp.last_frame
    path = placement_file(csv_path)
    path.write_text(json.dumps(values, indent=2) + "\n")
    return path
