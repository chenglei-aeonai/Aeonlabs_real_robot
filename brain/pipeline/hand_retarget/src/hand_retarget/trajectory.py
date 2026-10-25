"""Portable exports matching real_robot/replay_real.py's existing controller protocol."""
import csv
import json
from pathlib import Path

import numpy as np


def export_trajectory(out: Path, metadata: dict) -> dict:
    from .robot import ARM_JOINT_NAMES, HAND_JOINT_NAMES, HAND_LOWER, HAND_UPPER

    with np.load(out / "robot_actions.npz", allow_pickle=False) as data:
        t = data["timestamp"]
        arm, hand = data["arm_q"], data["hand_q_measured"]
        velocity = np.concatenate([data["arm_dq"], data["hand_dq_measured"]], axis=1)
    q = np.concatenate([arm, hand], axis=1)
    if (len(t) < 2 or q.shape != (len(t), 26) or velocity.shape != q.shape
            or not all(np.isfinite(a).all() for a in (t, q, velocity))
            or not np.isclose(t[0], 0) or np.any(np.diff(t) <= 0)):
        raise ValueError("invalid robot trajectory dimensions, values or timestamps")
    if (np.any(hand < HAND_LOWER - 1e-6) or np.any(hand > HAND_UPPER + 1e-6)
            or np.any(np.abs(arm) > [2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])):
        raise ValueError("trajectory exceeds existing real-robot joint limits")
    peak_speed = float(np.max(np.abs(np.diff(q, axis=0) / np.diff(t)[:, None])))
    if peak_speed > np.pi * 1.05:
        raise ValueError("trajectory exceeds existing replay controller speed limit")
    names = list(ARM_JOINT_NAMES + HAND_JOINT_NAMES)
    with (out / "robot_actions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", *[f"{n}_rad" for n in names],
                         *[f"{n}_rad_s" for n in names]])
        writer.writerows(np.column_stack([t, q, velocity]))
    trajectory = {
        "schema": "ur5e-dg5f-trajectory-v1", "joint_names": names,
        "position_units": "rad", "velocity_units": "rad/s", "time_units": "s",
        "rate_hz": 60, "duration_seconds": float(t[-1]),
        "placement_mode": metadata["placement_mode"], "placement": metadata["placement"],
        "hardware_executed": False,
        "points": [{"time_from_start": float(time), "positions": positions.tolist(),
                    "velocities": speeds.tolist()}
                   for time, positions, speeds in zip(t, q, velocity)],
    }
    (out / "trajectory.json").write_text(json.dumps(trajectory, allow_nan=False))
    commands = {
        "schema": "simtoolreal-controller-export-v1", "hardware_executed": False,
        "arm": {"transport": "ZMQ JSON", "payload": {"time": t.tolist(), "path": arm.tolist()}},
        "hand": {"transport": "UDP JSON", "rate_hz": 60,
                 "scheduled_messages": [
                     {"time_from_start": float(time),
                      "payload": {"type": "hand_target", "sequence": i + 1,
                                  "positions": positions.tolist()}}
                     for i, (time, positions) in enumerate(zip(t, hand))]},
        "replay_entrypoint": "real_robot/replay_real.py robot_actions.npz",
        "note": "Use the existing replay driver for synchronized dispatch and start-pose handling.",
    }
    (out / "controller_commands.json").write_text(json.dumps(commands, allow_nan=False))
    return {"samples": len(t), "peak_joint_speed_rad_s": peak_speed,
            "files": ["robot_actions.npz", "robot_actions.csv", "trajectory.json",
                      "controller_commands.json"]}
