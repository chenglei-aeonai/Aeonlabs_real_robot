"""The three retargeting methods. Each turns one skeleton into the DG5F's 20 joint angles.

Every retargeter receives the skeleton already expressed in the human palm frame, whose
axes match ``rl_dg_palm`` (x out of the palm side, y toward the thumb, z along the
fingers) and whose origin is the hand anchor. It may keep state from the previous frame.

Fingertip IK and vector retargeting each come with two optimizers:

- ``dex``: dex-retargeting, which only sees the fingertips, with DIP coupled to PIP and a
  joint speed limit added around it. Adjacent fingers can still pass through each other.
- ``own``: our HandOptimizer, which also follows every phalanx's direction and keeps
  adjacent fingers apart.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol

import numpy as np
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor
from dex_retargeting.optimizer import DexPilotOptimizer, PositionOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting

from .hand_optimizer import MAX_JOINT_SPEED, DexPilotVectors, HandOptimizer, fingertip_cost, human_phalanx_directions
from .robot import FINGERTIP_LINKS, HAND_JOINT_NAMES, HAND_LOWER, HAND_UPPER, PALM_LINK, write_hand_urdf
from .scaling import HandScaling, ScalingMode
from .skeleton import FINGER_CHAINS, WRIST

METHODS = ("fingertip_ik", "joint_mapping", "vector")
OPTIMIZERS = ("own", "dex")
DEFAULT_SCALING: dict[str, ScalingMode] = {"fingertip_ik": "global", "vector": "per_finger"}


class Retargeter(Protocol):
    name: str

    def reset(self) -> None: ...

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        """DG5F joint angles (20,), in ``HAND_JOINT_NAMES`` order."""
        ...


# -- dex-retargeting -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _hand_urdf() -> str:
    return str(write_hand_urdf())


# dex-retargeting's own stopping tolerances (1e-5, 1e-6) are sized for other objective
# scales; with fingertip errors in metres they stop the solver centimetres short.
SOLVER_TOLERANCE = 1e-9
# Weight of the pull toward the previous frame's angles: dex-retargeting's default. Lower
# values let a finger jump between equally good solutions from one frame to the next.
PREVIOUS_FRAME_WEIGHT = 4e-3
# With only the fingertip to follow, a long finger's three flexing joints are one too
# many; DIP follows PIP one to one, as it roughly does on a human finger.
DIP_FOLLOWS_PIP = (("rj_dg_2_3", "rj_dg_2_4"), ("rj_dg_3_3", "rj_dg_3_4"), ("rj_dg_4_3", "rj_dg_4_4"))
DEX_TARGET_JOINTS = [n for n in HAND_JOINT_NAMES if n not in {dip for _, dip in DIP_FOLLOWS_PIP}]


class _DexRetargeter:
    def __init__(self, optimizer, rate_hz: float):
        optimizer.opt.set_ftol_abs(SOLVER_TOLERANCE)
        optimizer.set_kinematic_adaptor(
            MimicJointKinematicAdaptor(
                optimizer.robot,
                target_joint_names=DEX_TARGET_JOINTS,
                source_joint_names=[pip for pip, _ in DIP_FOLLOWS_PIP],
                mimic_joint_names=[dip for _, dip in DIP_FOLLOWS_PIP],
                multipliers=[1.0] * len(DIP_FOLLOWS_PIP),
                offsets=[0.0] * len(DIP_FOLLOWS_PIP),
            )
        )
        self._optimizer = optimizer
        self._max_step = MAX_JOINT_SPEED / rate_hz
        self.reset()

    def reset(self) -> None:
        self._retargeting = SeqRetargeting(self._optimizer, has_joint_limits=True)
        self._last: np.ndarray | None = None

    def _solve(self, reference: np.ndarray) -> np.ndarray:
        # The solver may overshoot a limit by a milliradian; the real hand may not. Nor may
        # it move a joint faster than its speed limit, so the step is clamped and the
        # clamped angles become the next frame's starting point.
        q = np.clip(np.asarray(self._retargeting.retarget(reference), dtype=np.float64), HAND_LOWER, HAND_UPPER)
        if self._last is not None:
            q = np.clip(q, self._last - self._max_step, self._last + self._max_step)
            self._retargeting.last_qpos = q[self._optimizer.idx_pin2target].astype(np.float32)
        self._last = q
        return q.copy()


class DexFingertipIK(_DexRetargeter):
    """Put the robot fingertips at the (scaled) human fingertip positions."""

    name = "fingertip_ik"

    def __init__(self, scaling: HandScaling, scaling_mode: ScalingMode, rate_hz: float):
        self.scaling, self.scaling_mode = scaling, scaling_mode
        optimizer = PositionOptimizer(
            RobotWrapper(_hand_urdf()),
            DEX_TARGET_JOINTS,
            target_link_names=list(FINGERTIP_LINKS),
            target_link_human_indices=np.arange(5),
            norm_delta=PREVIOUS_FRAME_WEIGHT,
        )
        super().__init__(optimizer, rate_hz)

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        return self._solve(self.scaling.fingertip_targets(skeleton_in_palm, self.scaling_mode))


class DexVector(_DexRetargeter):
    """DexPilot: match palm-to-fingertip and fingertip-to-fingertip vectors, snapping near pinches shut.

    When a human thumb-finger pair comes within ``project_dist`` it is pulled into contact
    on the robot, and released once it is farther than ``escape_dist``.
    """

    name = "vector"

    def __init__(self, scaling: HandScaling, scaling_mode: ScalingMode, rate_hz: float):
        self.scaling, self.scaling_mode = scaling, scaling_mode
        optimizer = DexPilotOptimizer(
            RobotWrapper(_hand_urdf()),
            DEX_TARGET_JOINTS,
            finger_tip_link_names=list(FINGERTIP_LINKS),
            wrist_link_name=PALM_LINK,
            norm_delta=PREVIOUS_FRAME_WEIGHT,
        )
        origin, task = DexPilotOptimizer.generate_link_indices(5)
        self._origin, self._task = np.array(origin), np.array(task)
        super().__init__(optimizer, rate_hz)

    def reset(self) -> None:
        self._optimizer.projected[:] = False
        super().reset()

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        tips = self.scaling.fingertip_targets(skeleton_in_palm, self.scaling_mode)
        # Link 0 is the palm origin; links 1-5 are the fingertips, thumb first.
        points = np.vstack([np.zeros(3), tips])
        return self._solve(points[self._task] - points[self._origin])


# -- our optimizer -------------------------------------------------------------------

class OwnFingertipIK:
    """Fingertip positions, plus phalanx directions, finger collisions and the speed limit."""

    name = "fingertip_ik"

    def __init__(self, scaling: HandScaling, scaling_mode: ScalingMode, rate_hz: float):
        self.scaling, self.scaling_mode = scaling, scaling_mode
        self.optimizer = HandOptimizer(rate_hz)

    def reset(self) -> None:
        self.optimizer.reset()

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        targets = self.scaling.fingertip_targets(skeleton_in_palm, self.scaling_mode)
        term = fingertip_cost(targets, self.optimizer.model.tip_rows, self.optimizer.weights)
        return self.optimizer.solve(human_phalanx_directions(skeleton_in_palm), term)


class OwnVector:
    """DexPilot's vectors and pinch rule, plus phalanx directions, finger collisions and the speed limit."""

    name = "vector"

    def __init__(self, scaling: HandScaling, scaling_mode: ScalingMode, rate_hz: float):
        self.scaling, self.scaling_mode = scaling, scaling_mode
        self.optimizer = HandOptimizer(rate_hz)
        self.vectors = DexPilotVectors()

    def reset(self) -> None:
        self.optimizer.reset()
        self.vectors.reset()

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        tips = self.scaling.fingertip_targets(skeleton_in_palm, self.scaling_mode)
        term = self.vectors.cost(tips, self.optimizer.model.tip_rows, self.optimizer.weights)
        return self.optimizer.solve(human_phalanx_directions(skeleton_in_palm), term)


# -- direct joint mapping ------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.arccos(np.clip(_unit(a) @ _unit(b), -1.0, 1.0)))


def human_joint_angles(skeleton_in_palm: np.ndarray) -> np.ndarray:
    """Human hand angles (5, 4) in radians, thumb to pinky, from a skeleton in the palm frame.

    Thumb: [in-plane angle of the metacarpal, from pointing across the palm (+y) toward the
    fingers (+z); palmar angle of the metacarpal, out of the palm plane toward the palm side
    (+x); MCP flexion; IP flexion].
    Long fingers: [spread toward the thumb; MCP flexion toward the palm; PIP flexion;
    DIP flexion].
    """
    s = skeleton_in_palm
    x, y, z = np.eye(3)
    angles = np.zeros((5, 4))

    cmc, mcp, ip, tip = (s[i] for i in FINGER_CHAINS[0])
    metacarpal, proximal, distal = mcp - cmc, ip - mcp, tip - ip
    angles[0] = [
        np.arctan2(metacarpal @ z, metacarpal @ y),
        np.arcsin(np.clip(_unit(metacarpal) @ x, -1.0, 1.0)),
        _angle(metacarpal, proximal),
        _angle(proximal, distal),
    ]
    for finger in range(1, 5):
        mcp, pip, dip, tip = (s[i] for i in FINGER_CHAINS[finger])
        metacarpal, proximal, middle, distal = mcp - s[WRIST], pip - mcp, dip - pip, tip - dip
        elevation = lambda v: np.arcsin(np.clip(_unit(v) @ x, -1.0, 1.0))  # noqa: E731
        # Spread comes before flexion, as on the DG5F (joint 1 turns about the palm
        # normal, joint 2 bends toward the palm): it is the proximal phalanx's heading
        # within the palm plane. That heading is undefined when the finger points into
        # the palm, so it fades out once less than half the phalanx lies in the plane.
        in_plane = np.hypot(proximal @ y, proximal @ z) / np.linalg.norm(proximal)
        angles[finger] = [
            np.arctan2(proximal @ y, proximal @ z) * min(1.0, in_plane / 0.5),
            elevation(proximal) - elevation(metacarpal),
            _angle(proximal, middle),
            _angle(middle, distal),
        ]
    return angles


class JointMapping:
    """Direct joint mapping, following the MANUS glove teleop controller's DG5F mapping.

    Like ``manus_right_to_dg5f`` in the teleop repo (manus_dg5f_teleop_controller.py):
    the thumb's two CMC axes go to DG5F thumb joints 1 and 2 (swapped relative to
    spread/stretch, with a -7 degree offset on joint 1); spread drives joint 1 of index,
    middle and ring with the sign flipped to the DG5F's axis; index, middle and ring
    flexion is multiplied by ``flex_gain``; pinky opposition is held fixed; pinky joint
    2 is driven by the pinky's spread with a signed gain and a floor; pinky joints 3 and
    4 take the PIP and DIP flexion. The human angles are measured geometrically here
    rather than by a glove, so the offsets that depended on the glove's own zero (the
    -45 degrees on thumb joint 2) are not carried over.
    """

    name = "joint_mapping"

    def __init__(
        self,
        flex_gain: float = 1.15,
        thumb_j1_offset_deg: float = -7.0,
        thumb_j2_offset_deg: float = 0.0,
        pinky_opposition_deg: float = 0.0,
        pinky_j2_gain: float = -3.0,
        pinky_j2_min_deg: float = 5.0,
    ):
        self.flex_gain = flex_gain
        self.thumb_j1_offset = np.deg2rad(thumb_j1_offset_deg)
        self.thumb_j2_offset = np.deg2rad(thumb_j2_offset_deg)
        self.pinky_opposition = np.deg2rad(pinky_opposition_deg)
        self.pinky_j2_gain = pinky_j2_gain
        self.pinky_j2_min = np.deg2rad(pinky_j2_min_deg)

    def reset(self) -> None:
        pass

    def retarget(self, skeleton_in_palm: np.ndarray) -> np.ndarray:
        a = human_joint_angles(skeleton_in_palm)
        q = np.zeros((5, 4))
        # Thumb: in-plane angle -> joint 1 (rotation about the palm normal); palmar angle
        # -> joint 2, whose negative direction swings the thumb toward the palm.
        q[0] = [a[0, 0] + self.thumb_j1_offset, -a[0, 1] + self.thumb_j2_offset, a[0, 2], a[0, 3]]
        for finger in (1, 2, 3):
            # The DG5F spread joint's positive direction is away from the thumb.
            q[finger] = [-a[finger, 0], *(self.flex_gain * a[finger, 1:])]
        # Pinky: opposition, spread (positive away from the thumb), MCP, PIP. The gain is
        # negative for the same axis flip as the other fingers. The DG5F pinky has two
        # flexing joints; as in the teleop they take PIP and DIP.
        pinky_spread = max(self.pinky_j2_gain * a[4, 0], self.pinky_j2_min)
        q[4] = [self.pinky_opposition, pinky_spread, a[4, 2], a[4, 3]]
        return np.clip(q.reshape(20), HAND_LOWER, HAND_UPPER)


def make_retargeter(
    method: str,
    scaling: HandScaling,
    scaling_mode: ScalingMode | None = None,
    optimizer: str = "own",
    rate_hz: float = 24.0,
) -> Retargeter:
    if method == "joint_mapping":
        return JointMapping()
    classes = {
        ("fingertip_ik", "own"): OwnFingertipIK,
        ("fingertip_ik", "dex"): DexFingertipIK,
        ("vector", "own"): OwnVector,
        ("vector", "dex"): DexVector,
    }
    if (method, optimizer) not in classes:
        raise ValueError(f"unknown method {method!r} or optimizer {optimizer!r}; choose from {METHODS}, {OPTIMIZERS}")
    return classes[method, optimizer](scaling, scaling_mode or DEFAULT_SCALING[method], rate_hz)
