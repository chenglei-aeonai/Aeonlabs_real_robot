#!/usr/bin/env python3
"""Compare saved hand feedback through the actual replay readers, without sockets.

Only reads recorded files and writes diagnostic JSON. No robot objects are
initialized and no target is sent. Use the pipeline Python environment.
"""
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


class RecordedPacket:
    """Minimal in-memory replacement for recvfrom; never opens a socket."""

    def __init__(self, packet):
        self.payload = json.dumps(packet).encode()

    def recvfrom(self, _size):
        if self.payload is None:
            raise BlockingIOError
        payload, self.payload = self.payload, None
        return payload, ("recording", 0)


def readers():
    spec = importlib.util.spec_from_file_location("offline_replay", ROOT / "replay_real.py")
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    path = ROOT / "brain/pipeline/hand_retarget/src/hand_retarget/robot_rollout.py"
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body
                if getattr(node, "name", None) in ("SafetyStop", "check_finger_velocity")]
    replay = next(node for node in tree.body if getattr(node, "name", None) == "replay")
    selected.append(next(node for node in replay.body
                         if isinstance(node, ast.ClassDef) and node.name == "Hand"))
    namespace = dict(np=np, json=json, time=time, transport=legacy, armed=True, native=False, fixed_pose=False,
                     feedback=[], HAND_JOINT_NAMES=legacy.HAND_JOINTS)
    # Execute the source definitions themselves, not a duplicate of the guard.
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return legacy.Hand, namespace["Hand"], namespace["SafetyStop"]


def feed(reader, packet):
    instance = object.__new__(reader)  # Deliberately bypass socket-opening __init__.
    instance.state = RecordedPacket(packet)
    instance.q = instance.q_at = None
    instance.poll()
    return instance.q


def inspect(log, ros, legacy, guarded, safety_stop):
    with np.load(log, allow_pickle=False) as data:
        feedback = data["hand_feedback"]
        samples = data["samples"]
        columns = data["hand_feedback_columns"].astype(str)
    assert feedback.ndim == 2 and feedback.shape[1] == 61 and len(feedback)
    assert samples.ndim == 2 and samples.shape[1] == 53 and len(samples) > 1
    names = [name.split(":", 1)[1] for name in columns[1:21]]
    expected = [f"rj_dg_{finger}_{joint}" for finger in range(1, 6) for joint in range(1, 5)]
    assert names == expected, "Unexpected saved joint ordering"
    first_stop = None
    accepted = 0
    for index, row in enumerate(feedback):
        packet = dict(type="hand_state", positions=row[1:21].tolist(),
                      velocities=row[21:41].tolist(), currents_ma=row[41:61].tolist(),
                      measured_age_s=0.)
        assert np.array_equal(feed(legacy, packet), row[1:21])
        accepted += 1
        try:
            feed(guarded, packet)
        except safety_stop as error:
            if first_stop is None:
                joint = int(np.argmax(np.abs(row[21:41])))
                first_stop = dict(packet_index=index, joint=names[joint],
                                  velocity_rad_s=float(row[21 + joint]),
                                  velocity_rpm=float(row[21 + joint] * 30 / np.pi),
                                  monotonic_s=float(row[0]), message=str(error))
    target_rate = np.abs(np.diff(samples[:, 33:53], axis=0)) / np.diff(samples[:, 0])[:, None]
    result = dict(log=str(log.resolve()), packets=len(feedback),
                  legacy_reader_accepted=accepted, guarded_reader_first_stop=first_stop,
                  last_logged_elapsed_s=float(samples[-1, 0]),
                  initial_hand_error_rad=float(np.abs(samples[0, 7:27] - samples[0, 33:53]).max()),
                  max_logged_hand_error_rad=float(np.abs(samples[:, 7:27] - samples[:, 33:53]).max()),
                  max_logged_hand_target_secant_rad_s=float(target_rate.max()),
                  velocity_quantization_residual_rpm=float(np.abs(
                      feedback[:, 21:41] * 30 / np.pi - np.rint(feedback[:, 21:41] * 30 / np.pi)).max()),
                  minimum_udp_read_gap_s=float(np.diff(feedback[:, 0]).min()))
    if first_stop and ros:
        joint_name = first_stop["joint"]
        event_time = first_stop["monotonic_s"]
        observations = []
        for line in ros.read_text().splitlines():
            snapshot = json.loads(line)
            state = snapshot["joints"]
            stamp = state["received_monotonic_s"]
            if not event_time - .4 < stamp < event_time + .1:
                continue
            index = state["names"].index(joint_name)  # ROS order differs from UDP.
            if observations and observations[-1]["received_monotonic_s"] == stamp:
                continue
            pid = next(item for item in snapshot["pid"]["joints"] if item["name"] == joint_name)
            observations.append(dict(received_monotonic_s=stamp,
                                     position_rad=state["position"][index],
                                     velocity_rad_s=state["velocity"][index],
                                     pid_reference_rad=pid["reference"],
                                     pid_feedback_rad=pid["feedback"],
                                     pid_error_rad=pid["error"], pid_output=pid["output"]))
        if len(observations) > 1:
            # Independently timed ROS samples nearest the recorded trigger velocity.
            index = min(range(1, len(observations)), key=lambda i:
                        abs(observations[i]["velocity_rad_s"] - first_stop["velocity_rad_s"]))
            a, b = observations[index - 1:index + 1]
            dt = b["received_monotonic_s"] - a["received_monotonic_s"]
            delta = b["position_rad"] - a["position_rad"]
            result["ros_trigger_interval"] = dict(dt_s=dt, delta_rad=delta,
                                                  mean_encoder_velocity_rad_s=delta / dt,
                                                  start=a, end=b)
        result["ros_event_observations"] = observations
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--ros", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inspect(args.log, args.ros, *readers())
    result["limitations"] = [
        "Reconstructed packets contain saved position/velocity/current and synthetic fresh age=0; not a raw wire capture.",
        "Only actual reader/guard definitions are exercised; this is not a whole replay or a hardware test.",
        "UDP timestamps are queue drain times; do not derive instantaneous velocity from adjacent UDP receive times.",
        "ROS timestamps are local receipt times, not hardware acquisition times; encoder slopes are approximate.",
        "PID, joint and UDP samples are asynchronous; firmware velocity filter/latency is not established.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ("ros_event_observations", "limitations")}, indent=2))


if __name__ == "__main__":
    main()
