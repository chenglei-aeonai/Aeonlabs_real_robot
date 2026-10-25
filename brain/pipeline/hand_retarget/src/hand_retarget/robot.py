"""The UR5e + right DG5F: its URDF, scene placement and kinematics.

The URDF, meshes, home pose and scene numbers are copied from simtoolreal_newton
(``assets/ur5e_right_dg5f.urdf``, ``cfg/simtoolreal_config.py``), so a replay here
lines up with that repo's simulation and robot demonstrations.
"""

from __future__ import annotations

import re
import tempfile
import xml.etree.ElementTree as ET
from functools import cached_property
from pathlib import Path

import numpy as np
import pinocchio as pin

ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"
ROBOT_URDF = ASSETS_DIR / "ur5e_right_dg5f.urdf"

ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
# Finger 1 is the thumb, 5 the pinky; joint 1 is the one nearest the palm.
HAND_JOINT_NAMES = tuple(f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5))
PALM_LINK = "rl_dg_palm"
FINGERTIP_LINKS = tuple(f"rl_dg_{finger}_tip" for finger in range(1, 6))

# The conservative limits the real hand's ROS bridge enforces (same values as the
# MANUS teleop controller); tighter than the URDF on the thumb.
HAND_LOWER = np.array(
    [
        -0.3839724354, -3.1415926536, 0.0, 0.0,
        -0.4188790205, 0.0, 0.0, 0.0,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.0174532925, -0.4188790205, 0.0, 0.0,
    ]
)
HAND_UPPER = np.array(
    [
        0.8901179185, 0.0, 1.5707963268, 1.5707963268,
        0.6108652382, 2.0071286398, 1.5707963268, 1.5707963268,
        0.6108652382, 1.9547687622, 1.5707963268, 1.5707963268,
        0.4188790205, 1.9024088847, 1.5707963268, 1.5707963268,
        1.0471975512, 0.6108652382, 1.5707963268, 1.5707963268,
    ]
)

# First frame of simtoolreal's robot demonstration: palm down, 16 cm above the table.
HOME_ARM_Q = np.array([-1.5708, -1.05, 1.95, -0.9, 1.571, -2.618])
HOME_HAND_Q = np.array(
    [
        0.382, -0.195, 0.037, 0.033,
        -0.199, 0.058, 0.0, 0.246,
        0.136, 0.391, 0.019, 0.058,
        0.213, 0.393, 0.042, 0.031,
        0.211, 0.321, 0.339, 0.026,
    ]
)

# Scene, in the world frame of simtoolreal's simulation.
ROBOT_BASE_POSITION = np.array([0.0, 0.6, 0.55])
TABLE_SIZE = np.array([0.75, 0.75, 0.3])
TABLE_CENTER_XY = np.array([0.0, 0.0])
TABLE_TOP_Z = ROBOT_BASE_POSITION[2] - 0.035  # simtoolreal: the surface is 3.5 cm below the robot base

# Joints whose position stands for the human MCP (or thumb CMC) of each finger,
# thumb to pinky. The pinky's first two joints are opposition and spread, so its
# flexing MCP is joint 3.
FINGER_BASE_JOINTS = ("rj_dg_1_2", "rj_dg_2_2", "rj_dg_3_2", "rj_dg_4_2", "rj_dg_5_3")
# The joints after the base, down to the tip, per finger.
FINGER_CHAIN_JOINTS = (
    ("rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4"),
    ("rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4"),
    ("rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4"),
    ("rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4"),
    ("rj_dg_5_3", "rj_dg_5_4"),
)


def _se3(transform: np.ndarray) -> pin.SE3:
    return pin.SE3(transform[:3, :3].copy(), transform[:3, 3].copy())


def _matrix(placement: pin.SE3) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = placement.rotation
    out[:3, 3] = placement.translation
    return out


def write_hand_urdf(directory: Path | None = None) -> Path:
    """Write the DG5F alone, rooted at the palm, with the real hand's joint limits.

    dex-retargeting optimises a hand with a fixed base, and the floating-hand view
    draws it at the palm target. Mesh paths are made absolute so the file can live
    anywhere.
    """
    tree = ET.parse(ROBOT_URDF)
    root = tree.getroot()
    children: dict[str, list[ET.Element]] = {}
    for joint in root.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    keep_links, keep_joints, stack = {PALM_LINK}, [], [PALM_LINK]
    while stack:
        for joint in children.get(stack.pop(), []):
            child = joint.find("child").get("link")
            keep_links.add(child)
            keep_joints.append(joint)
            stack.append(child)
    limits = dict(zip(HAND_JOINT_NAMES, zip(HAND_LOWER, HAND_UPPER)))
    for joint in keep_joints:
        name = joint.get("name")
        if name in limits:
            joint.find("limit").set("lower", repr(float(limits[name][0])))
            joint.find("limit").set("upper", repr(float(limits[name][1])))
    for element in list(root):
        if element.tag == "link" and element.get("name") not in keep_links:
            root.remove(element)
        elif element.tag == "joint" and element not in keep_joints:
            root.remove(element)
    root.set("name", "dg5f_right_hand")
    text = ET.tostring(root, encoding="unicode")
    text = re.sub(
        r'filename="([^"]+)"',
        lambda m: f'filename="{(ASSETS_DIR / m.group(1)).resolve()}"',
        text,
    )
    directory = Path(directory or tempfile.mkdtemp(prefix="hand_retarget_"))
    path = directory / "dg5f_right_hand.urdf"
    path.write_text(text, encoding="utf-8")
    return path


class RobotKinematics:
    """Forward kinematics of the UR5e + DG5F and damped least-squares IK for the palm.

    Positions and poses are in the world frame; the robot base sits at
    ``ROBOT_BASE_POSITION`` with no rotation, as in simtoolreal.
    """

    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(ROBOT_URDF))
        self.data = self.model.createData()
        self.base = pin.SE3(np.eye(3), ROBOT_BASE_POSITION.copy())
        self._q_index = {
            name: self.model.joints[self.model.getJointId(name)].idx_q
            for name in ARM_JOINT_NAMES + HAND_JOINT_NAMES
        }
        self.arm_idx = np.array([self._q_index[n] for n in ARM_JOINT_NAMES])
        self.hand_idx = np.array([self._q_index[n] for n in HAND_JOINT_NAMES])
        self.palm_frame = self.model.getFrameId(PALM_LINK)
        self.tip_frames = [self.model.getFrameId(n) for n in FINGERTIP_LINKS]

    def full_q(self, arm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        q[self.arm_idx] = arm_q
        q[self.hand_idx] = hand_q
        return q

    def _forward(self, arm_q: np.ndarray, hand_q: np.ndarray) -> None:
        pin.framesForwardKinematics(self.model, self.data, self.full_q(arm_q, hand_q))

    def palm_pose(self, arm_q: np.ndarray, hand_q: np.ndarray | None = None) -> np.ndarray:
        """World pose (4x4) of ``rl_dg_palm``."""
        self._forward(arm_q, HOME_HAND_Q if hand_q is None else hand_q)
        return _matrix(self.base * self.data.oMf[self.palm_frame])

    def fingertips(self, arm_q: np.ndarray, hand_q: np.ndarray) -> np.ndarray:
        """World positions (5, 3) of the fingertips, thumb to pinky."""
        self._forward(arm_q, hand_q)
        return np.stack([(self.base * self.data.oMf[f]).translation for f in self.tip_frames])

    def hand_points_in_palm(self, hand_q: np.ndarray, joint_names) -> np.ndarray:
        """Positions of the given hand joints' origins, in the palm frame."""
        self._forward(HOME_ARM_Q, hand_q)
        palm_inv = self.data.oMf[self.palm_frame].inverse()
        return np.stack(
            [(palm_inv * self.data.oMi[self.model.getJointId(n)]).translation for n in joint_names]
        )

    def fingertips_in_palm(self, hand_q: np.ndarray) -> np.ndarray:
        """Fingertip positions (5, 3) in the palm frame, thumb to pinky."""
        self._forward(HOME_ARM_Q, hand_q)
        palm_inv = self.data.oMf[self.palm_frame].inverse()
        return np.stack([(palm_inv * self.data.oMf[f]).translation for f in self.tip_frames])

    @cached_property
    def finger_bases_in_palm(self) -> np.ndarray:
        """Where each finger's base joint sits in the palm frame (hand at zero), thumb to pinky."""
        return self.hand_points_in_palm(np.zeros(20), FINGER_BASE_JOINTS)

    @cached_property
    def hand_anchor_in_palm(self) -> np.ndarray:
        """Centre of the knuckle row: the mean of the four long fingers' MCP joints."""
        return self.finger_bases_in_palm[1:].mean(axis=0)

    @cached_property
    def finger_lengths(self) -> np.ndarray:
        """Length of each finger from its base joint to its tip (hand at zero), thumb to pinky."""
        tips = self.fingertips_in_palm(np.zeros(20))
        lengths = []
        for finger, joints in enumerate(FINGER_CHAIN_JOINTS):
            points = np.vstack([self.hand_points_in_palm(np.zeros(20), joints), tips[finger]])
            lengths.append(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        return np.array(lengths)

    def solve_palm_ik(
        self,
        target_palm_world: np.ndarray,
        arm_q_init: np.ndarray,
        iterations: int = 100,
        damping: float = 1e-3,
        tolerance: float = 1e-5,
    ) -> tuple[np.ndarray, float, float]:
        """Arm angles that put the palm at ``target_palm_world``; the hand does not affect the palm.

        Returns the arm angles and the remaining position (m) and rotation (rad) errors;
        when the target is out of reach the result is the best effort, not an exception.
        """
        target = self.base.inverse() * _se3(target_palm_world)
        q = self.full_q(arm_q_init, HOME_HAND_Q)
        lower = self.model.lowerPositionLimit[self.arm_idx]
        upper = self.model.upperPositionLimit[self.arm_idx]
        for _ in range(iterations):
            pin.framesForwardKinematics(self.model, self.data, q)
            current = self.data.oMf[self.palm_frame]
            error = pin.log6(current.actInv(target)).vector
            if np.linalg.norm(error) < tolerance:
                break
            jacobian = pin.computeFrameJacobian(
                self.model, self.data, q, self.palm_frame, pin.ReferenceFrame.LOCAL
            )[:, self.arm_idx]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping * np.eye(6), error)
            q[self.arm_idx] = np.clip(q[self.arm_idx] + step, lower, upper)
        pin.framesForwardKinematics(self.model, self.data, q)
        residual = self.data.oMf[self.palm_frame].actInv(target)
        return (
            q[self.arm_idx].copy(),
            float(np.linalg.norm(residual.translation)),
            float(np.linalg.norm(pin.log3(residual.rotation))),
        )
