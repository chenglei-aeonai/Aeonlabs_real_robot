"""Bridge H3's decoded camera-space predictions to this repository's retargeter."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .skeleton import HumanDemonstration, KEYPOINT_NAMES, FINGER_CHAINS


def rigid_transforms(value: np.ndarray, frames: int) -> np.ndarray:
    """Accept one fixed transform or one camera-to-demonstration transform per frame."""
    value = np.asarray(value, dtype=float)
    if value.shape == (4, 4):
        value = np.broadcast_to(value, (frames, 4, 4)).copy()
    if value.shape != (frames, 4, 4) or not np.isfinite(value).all():
        raise ValueError("camera transforms must be finite (4,4) or (T,4,4)")
    if not np.allclose(value[:, 3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("invalid homogeneous transform")
    rotation = value[:, :3, :3]
    if (not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-6)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-6)):
        raise ValueError("camera rotation must be proper orthonormal (no scale/reflection)")
    return value


def fixed_camera_transform(pitch_down_deg: float) -> np.ndarray:
    """Optical x-right/y-down/z-forward -> level x-forward/y-left/z-up."""
    if not np.isfinite(pitch_down_deg) or not -89 < pitch_down_deg < 89:
        raise ValueError("camera pitch must be finite and between -89 and 89 degrees")
    p = np.deg2rad(pitch_down_deg)
    level = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0],
                   [-np.sin(p), 0, np.cos(p)]])
    result = np.eye(4)
    result[:3, :3] = ry @ level
    return result


def load_h3_actions(path: Path, seconds: float, camera_to_demo: np.ndarray, *,
                    center_origin: bool = True
                    ) -> tuple[HumanDemonstration, dict]:
    """Keep right hand (index 1); H3 and the retargeter use the same dpl/MANO order.

    No torch pickle loading, hand mirroring, guessed joint permutation, or
    per-frame wrist centering. All displacement relative to frame zero survives.
    """
    if not np.isfinite(seconds) or not 4 <= seconds <= 5:
        raise ValueError("this pipeline supports 4–5 seconds")
    with np.load(path, allow_pickle=False) as data:
        joints = np.asarray(data["joints"], dtype=float)
        fps = float(data["fps"])
        K = np.asarray(data["K"], dtype=float)
    if joints.ndim != 4 or joints.shape[1:] != (2, 21, 3):
        raise ValueError("expected H3 joints shape (T,2,21,3), hands [left,right]")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid action fps")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("invalid camera intrinsics K")
    count = round(seconds * fps)
    if not np.isclose(count / fps, seconds):
        raise ValueError("duration must contain an integer number of video frames")
    if len(joints) < count:
        raise ValueError(f"need {count} frames for {seconds}s, received {len(joints)}")
    # Per-frame camera motion must cover the complete source, or the retained prefix.
    camera_to_demo = np.asarray(camera_to_demo)
    if camera_to_demo.shape == (len(joints), 4, 4):
        camera_to_demo = camera_to_demo[:count]
    transforms = rigid_transforms(camera_to_demo, count)
    right = joints[:count, 1]
    if not np.isfinite(right).all():
        raise ValueError("right-hand prediction has missing/non-finite keypoints")
    if np.any(right[..., 2] <= 0):
        raise ValueError("right-hand prediction is missing or behind the optical camera")
    for chain in FINGER_CHAINS:
        if np.any(np.linalg.norm(np.diff(right[:, chain], axis=1), axis=-1) < 1e-6):
            raise ValueError("right-hand prediction contains collapsed finger bones")
    z = right[:, [1, 4, 7, 10]].mean(1) - right[:, 0]
    across = right[:, 1] - right[:, 7]
    if np.any(np.linalg.norm(np.cross(z, across), axis=-1) < 1e-8):
        raise ValueError("right-hand palm frame is degenerate")
    skeletons = np.einsum("tij,tkj->tki", transforms[:, :3, :3], right) + transforms[:, None, :3, 3]
    origin = skeletons[0, 0].copy()
    # A calibrated demonstration frame has a meaningful absolute origin.
    # Only home-pose experiments may discard the initial wrist translation.
    removed_origin = origin if center_origin else np.zeros(3)
    skeletons -= removed_origin
    demo = HumanDemonstration(path.stem, fps, skeletons,
                              camera_position=transforms[0, :3, 3] - removed_origin)
    return demo, {
        "source_frames": len(joints), "video_frames": count, "fps": fps,
        "duration_seconds": count / fps, "hand": "right", "hand_index": 1,
        "joint_order": "dpl_mano_21", "units": "metres", "K": K.tolist(),
        "origin_before_centering": origin.tolist(),
        "origin_removed": removed_origin.tolist(),
        "origin_centered": center_origin,
        "camera_to_demo": transforms.tolist(),
    }


def write_world_csv(demo: HumanDemonstration, path: Path) -> None:
    """Write the existing viewer/exporter's uncorrected input schema."""
    with path.open("w", newline="") as stream:
        stream.write(f"# {demo.name}: {demo.num_frames} frames at {demo.fps:g} fps\n")
        writer = csv.writer(stream)
        writer.writerow(["frame", "time_s", "hand", "joint", "name",
                         "raw_x_m", "raw_y_m", "raw_z_m"])
        for t, skeleton in enumerate(demo.skeletons):
            for j, xyz in enumerate(skeleton):
                writer.writerow([t, t / demo.fps, "right", j, KEYPOINT_NAMES[j], *xyz])


def retarget_actions(demo: HumanDemonstration, out: Path, method: str = "vector",
                     robot_from_demo: np.ndarray | None = None) -> dict:
    """Run the upstream optimizer and exporter; hold last pose to video end time."""
    from dataclasses import replace
    from .pipeline import (Settings, ArmStage, Replay, retarget_hands,
                           retarget_demonstration, export_robot_demonstration)
    from .robot import RobotKinematics, ARM_JOINT_NAMES, HAND_JOINT_NAMES

    robot, settings = RobotKinematics(), Settings(method=method)
    if robot_from_demo is None:
        replay = retarget_demonstration(demo, robot, settings)
    else:
        placement = rigid_transforms(robot_from_demo, 1)[0]
        hands = retarget_hands(demo, robot, settings)
        arm = ArmStage(robot, placement, False, demo.fps)
        replay = Replay(demo, settings, placement, [arm.step(h) for h in hands])
    # A 96-frame, 24fps video covers [0,4); the last observation is at 95/24.
    # Explicit zero-order hold closes that final interval without inventing motion.
    extended = replace(demo, skeletons=np.concatenate([demo.skeletons, demo.skeletons[-1:]]))
    replay = replace(replay, demo=extended, frames=[*replay.frames, replay.frames[-1]])
    for key in ("arm_q", "hand_q", "fingertip_error", "palm_position_error", "palm_rotation_error"):
        if not np.isfinite(replay.stack(key)).all():
            raise ValueError(f"retargeting produced non-finite {key}")
    path = export_robot_demonstration(replay, out / "robot_actions.npz")
    return {
        "robot_actions": str(path), "robot": "UR5e + Tesollo DG5F right",
        "method": method, "optimizer": settings.optimizer,
        "arm_joint_names": list(ARM_JOINT_NAMES), "hand_joint_names": list(HAND_JOINT_NAMES),
        "joint_units": "radians", "rate_hz": 60,
        "duration_seconds": demo.num_frames / demo.fps,
        "last_pose_hold_seconds": 1 / demo.fps,
        "placement_mode": "calibrated" if robot_from_demo is not None else "upstream_home_pose",
        "placement": replay.placement.tolist(),
        "mean_fingertip_error_m": float(replay.stack("fingertip_error").mean()),
        "max_palm_position_error_m": float(replay.stack("palm_position_error").max()),
        "max_palm_rotation_error_rad": float(replay.stack("palm_rotation_error").max()),
        "max_finger_penetration_m": float(replay.worst_finger_penetration().max()),
        "hardware_executed": False,
    }
