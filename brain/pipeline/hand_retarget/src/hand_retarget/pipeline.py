"""From a human demonstration to robot joint angles, one frame at a time, and the robot demonstration export."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pinocchio as pin

from .collisions import FingerCollisionChecker
from .grasp import GraspStrengthening, strengthen_grasp
from .hand_optimizer import MAX_JOINT_SPEED
from .placement import PlacementAdjustment, adjusted_placement, apply, default_placement
from .retargeters import DEFAULT_SCALING, make_retargeter
from .robot import HOME_ARM_Q, TABLE_TOP_Z, RobotKinematics, write_hand_urdf
from .scaling import HandScaling, ScalingMode
from .skeleton import HumanDemonstration, OneEuroFilter, palm_frame, to_frame

if TYPE_CHECKING:
    from .tube import TubePlacement

ROBOT_RATE_HZ = 60.0


@dataclass
class Settings:
    method: str = "vector"
    optimizer: str = "own"  # fingertip IK and vector: "own" or "dex" (dex-retargeting)
    scaling_mode: ScalingMode | None = None  # None: the method's default
    filter: bool = False
    floating_hand: bool = False
    adjustment: PlacementAdjustment = field(default_factory=PlacementAdjustment)
    grasp: GraspStrengthening = field(default_factory=GraspStrengthening)

    def hand_key(self) -> tuple:
        """The settings the retargeter's joint angles depend on; the placement and the grasp
        strengthening, applied afterwards, are not among them."""
        return (self.method, self.optimizer, self.scaling_mode, self.filter)


@dataclass
class HandFrame:
    """One frame of hand retargeting, which works in the palm frame and so does not depend on the placement."""

    skeleton: np.ndarray  # (21, 3) in the demonstration frame, after the filter
    skeleton_in_palm: np.ndarray  # (21, 3)
    hand_q: np.ndarray  # (20,)
    fingertip_targets_in_palm: np.ndarray  # (5, 3) per-finger-scaled human fingertips: the fingertip error's reference
    finger_penetration: dict[tuple[int, int], float]  # finger pair (0 = thumb) -> how deep their meshes overlap, m


@dataclass
class FrameResult(HandFrame):
    skeleton_world: np.ndarray = None  # (21, 3)
    palm_target: np.ndarray = None  # (4, 4) where the palm should be
    palm_pose: np.ndarray = None  # (4, 4) where the palm is (equal to the target for the floating hand)
    arm_q: np.ndarray = None  # (6,), NaN for the floating hand
    fingertips: np.ndarray = None  # (5, 3) robot fingertips, world
    fingertip_targets: np.ndarray = None  # (5, 3) the reference fingertips, world
    fingertip_error: np.ndarray = None  # (5,) metres
    palm_position_error: float = 0.0
    palm_rotation_error: float = 0.0


@lru_cache(maxsize=1)
def _collision_checker() -> FingerCollisionChecker:
    return FingerCollisionChecker(str(write_hand_urdf()))


class HandStage:
    """The fingers: skeleton (demonstration frame) -> DG5F joint angles, one frame at a time."""

    def __init__(self, demo: HumanDemonstration, robot: RobotKinematics, settings: Settings):
        self.scaling = HandScaling.from_demonstration(demo, robot)
        self.retargeter = make_retargeter(
            settings.method, self.scaling, settings.scaling_mode, settings.optimizer, demo.fps
        )
        self.filter = OneEuroFilter(demo.fps) if settings.filter else None
        self.collisions = _collision_checker()
        self.max_step = MAX_JOINT_SPEED / demo.fps
        self.reset()

    def reset(self) -> None:
        self.retargeter.reset()
        if self.filter:
            self.filter.reset()
        self._hand_q: np.ndarray | None = None

    def step(self, skeleton: np.ndarray) -> HandFrame:
        if self.filter:
            skeleton = self.filter(skeleton)
        skeleton_in_palm = to_frame(palm_frame(skeleton), skeleton)
        hand_q = self.retargeter.retarget(skeleton_in_palm)
        if self._hand_q is not None:
            # The real hand cannot move a joint faster than this, whatever the method asks.
            hand_q = np.clip(hand_q, self._hand_q - self.max_step, self._hand_q + self.max_step)
        self._hand_q = hand_q
        return HandFrame(
            skeleton=skeleton,
            skeleton_in_palm=skeleton_in_palm,
            hand_q=hand_q,
            fingertip_targets_in_palm=self.scaling.fingertip_targets(skeleton_in_palm, "per_finger"),
            finger_penetration=self.collisions.penetrations(hand_q),
        )


class ArmStage:
    """The placement and the arm: puts the palm where the placed human palm is, one frame at a time."""

    def __init__(self, robot: RobotKinematics, placement: np.ndarray, floating_hand: bool, rate_hz: float,
                 initial_arm_q: np.ndarray | None = None, palm_offset: np.ndarray | None = None):
        self.robot = robot
        self.placement = placement
        self.floating_hand = floating_hand
        self.initial_arm_q = HOME_ARM_Q.copy() if initial_arm_q is None else np.asarray(initial_arm_q,dtype=float).copy()
        if self.initial_arm_q.shape!=(6,) or not np.isfinite(self.initial_arm_q).all():
            raise ValueError("Arm seed must contain six finite joint angles.")
        self.palm_offset = np.eye(4) if palm_offset is None else np.asarray(palm_offset,dtype=float).copy()
        if palm_offset is None:
            self.palm_offset[:3,3] = -robot.hand_anchor_in_palm
        # The UR5e's joint speed limits, per frame: when the human wrist turns faster, the
        # arm lags behind and the palm error says by how much.
        self.max_step = robot.model.velocityLimit[robot.arm_idx] / rate_hz
        self.reset()

    def reset(self) -> None:
        self._arm_q = self.initial_arm_q.copy()
        self._first = True

    def step(self, hand: HandFrame) -> FrameResult:
        skeleton_world = apply(self.placement, hand.skeleton)
        human_palm = palm_frame(skeleton_world)
        palm_target = human_palm @ self.palm_offset
        if self.floating_hand:
            arm_q = np.full(6, np.nan)
            palm_pose, position_error, rotation_error = palm_target, 0.0, 0.0
        else:
            arm_q, _, _ = self.robot.solve_palm_ik(palm_target, self._arm_q)
            if not self._first:  # the robot is brought to the first pose slowly, before the replay
                arm_q = np.clip(arm_q, self._arm_q - self.max_step, self._arm_q + self.max_step)
            self._arm_q, self._first = arm_q, False
            palm_pose = self.robot.palm_pose(arm_q)
            position_error = float(np.linalg.norm(palm_pose[:3, 3] - palm_target[:3, 3]))
            rotation_error = float(np.linalg.norm(pin.log3(palm_target[:3, :3].T @ palm_pose[:3, :3])))
        fingertips = apply(palm_pose, self.robot.fingertips_in_palm(hand.hand_q))
        targets = apply(palm_target, hand.fingertip_targets_in_palm)
        return FrameResult(
            **vars(hand),
            skeleton_world=skeleton_world,
            palm_target=palm_target,
            palm_pose=palm_pose,
            arm_q=arm_q,
            fingertips=fingertips,
            fingertip_targets=targets,
            fingertip_error=np.linalg.norm(fingertips - targets, axis=1),
            palm_position_error=position_error,
            palm_rotation_error=rotation_error,
        )


class FrameRetargeter:
    """Hand retargeting for one human demonstration, called once per skeleton, in order."""

    def __init__(self, demo: HumanDemonstration, robot: RobotKinematics, settings: Settings):
        self.hand = HandStage(demo, robot, settings)
        self.placement = adjusted_placement(default_placement(demo, robot), demo.skeletons[0], settings.adjustment)
        self.arm = ArmStage(robot, self.placement, settings.floating_hand, demo.fps)

    def reset(self) -> None:
        self.hand.reset()
        self.arm.reset()

    def step(self, skeleton: np.ndarray) -> FrameResult:
        """Retarget one skeleton given in the demonstration frame."""
        return self.arm.step(self.hand.step(skeleton))


@dataclass
class Replay:
    """A whole human demonstration retargeted with one set of settings."""

    demo: HumanDemonstration
    settings: Settings
    placement: np.ndarray
    frames: list[FrameResult]

    def stack(self, attribute: str) -> np.ndarray:
        return np.stack([getattr(frame, attribute) for frame in self.frames])

    def cut(self, first_frame: int = 0, last_frame: int | None = None) -> "Replay":
        """Only the video frames from ``first_frame`` to ``last_frame``, both kept.

        The whole demonstration is retargeted first and cut afterwards, so the kept
        frames are the same ones the viewer shows, whatever the range.
        """
        last_frame = self.demo.num_frames - 1 if last_frame is None else min(last_frame, self.demo.num_frames - 1)
        if not 0 <= first_frame < last_frame:
            raise ValueError(f"cannot keep frames {first_frame} to {last_frame} of {self.demo.num_frames}")
        keep = slice(first_frame, last_frame + 1)
        return replace(self, demo=replace(self.demo, skeletons=self.demo.skeletons[keep]), frames=self.frames[keep])

    def worst_finger_penetration(self) -> np.ndarray:
        """Per frame, the deepest overlap (m) between any two fingers' meshes; 0 when none touch."""
        return np.array([max(frame.finger_penetration.values(), default=0.0) for frame in self.frames])


def retarget_hands(demo: HumanDemonstration, robot: RobotKinematics, settings: Settings) -> list[HandFrame]:
    """The slow half: the fingers, for every frame. Reusable across placements."""
    stage = HandStage(demo, robot, settings)
    return [stage.step(skeleton) for skeleton in demo.skeletons]


def strengthened_hands(hands: list[HandFrame], grasp: GraspStrengthening, fps: float) -> list[HandFrame]:
    """The hand frames with the grasp strengthening applied; the frames it leaves alone are returned as they are."""
    if not grasp.active:
        return hands
    hand_q = strengthen_grasp(np.stack([hand.hand_q for hand in hands]), grasp, fps)
    collisions = _collision_checker()
    return [
        hand if np.array_equal(q, hand.hand_q)
        else replace(hand, hand_q=q, finger_penetration=collisions.penetrations(q))
        for hand, q in zip(hands, hand_q)
    ]


def retarget_demonstration(
    demo: HumanDemonstration,
    robot: RobotKinematics,
    settings: Settings,
    hands: list[HandFrame] | None = None,
) -> Replay:
    """Retarget a whole demonstration; pass ``hands`` from an earlier run with the same
    ``settings.hand_key()`` to only redo the grasp strengthening, the placement and the arm."""
    if hands is None:
        hands = retarget_hands(demo, robot, settings)
    hands = strengthened_hands(hands, settings.grasp, demo.fps)
    placement = adjusted_placement(default_placement(demo, robot), demo.skeletons[0], settings.adjustment)
    arm = ArmStage(robot, placement, settings.floating_hand, demo.fps)
    return Replay(demo=demo, settings=settings, placement=placement, frames=[arm.step(hand) for hand in hands])


def export_robot_demonstration(
    replay: Replay, path: str | Path, table_top_z: float = TABLE_TOP_Z, tube: "TubePlacement | None" = None
) -> Path:
    """Write the replay as a robot demonstration, in simtoolreal's layout, resampled to 60 Hz.

    The joint keys are simtoolreal's (``arm_q``, ``arm_dq``, ``hand_q_measured``,
    ``hand_dq_measured``), so its tools read the file as they read a recording; here the
    hand angles are the retargeted commands, not measurements (with the grasp strengthening,
    if any: ``grasp_extra`` in rad, thumb to pinky, and its first, last and ramp video frames). The settings that made the
    file are stored beside them, and ``table_top_z`` is the world height of the table
    top the replay was judged against. With ``tube``, the tube's size, starting pose, the
    path of the human grasp that carries it and where the human leaves it are stored too.
    """
    source_t = np.arange(replay.demo.num_frames) / replay.demo.fps
    t = np.arange(0.0, source_t[-1] + 1e-9, 1.0 / ROBOT_RATE_HZ)
    resample = lambda values: np.stack(  # noqa: E731
        [np.interp(t, source_t, values[:, j]) for j in range(values.shape[1])], axis=1
    )
    arm_q, hand_q = resample(replay.stack("arm_q")), resample(replay.stack("hand_q"))
    s = replay.settings
    a = s.adjustment
    tube_keys = {}
    if tube is not None:
        tube_keys = dict(
            tube_length=tube.spec.length,
            tube_radius=tube.spec.radius,
            tube_mass=tube.spec.mass,
            tube_friction=tube.spec.friction,
            tube_position=tube.position,
            tube_yaw=tube.yaw,
            tube_goal=tube.goal,
            tube_grasp_time=tube.grasp_frame / replay.demo.fps,
            grasp_centre_path=resample(tube.grasp_centre_path),
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **tube_keys,
        timestamp=t,
        monotonic_timestamp=t,
        arm_q=arm_q,
        arm_dq=np.gradient(arm_q, t, axis=0),
        hand_q_measured=hand_q,
        hand_dq_measured=np.gradient(hand_q, t, axis=0),
        source=replay.demo.name,
        method=s.method,
        optimizer=s.optimizer if s.method != "joint_mapping" else "",
        scaling_mode=(s.scaling_mode or DEFAULT_SCALING.get(s.method, "")),
        filter=s.filter,
        floating_hand=s.floating_hand,
        placement_adjustment=np.array([a.x, a.y, a.z, a.yaw, a.pitch, a.roll]),  # m, m, m, rad, rad, rad
        placement=replay.placement,
        grasp_extra=np.array(s.grasp.extra),
        grasp_frames=np.array(
            [s.grasp.first_frame, -1 if s.grasp.last_frame is None else s.grasp.last_frame, s.grasp.ramp_frames]
        ),
        table_top_z=table_top_z,
    )
    return path
