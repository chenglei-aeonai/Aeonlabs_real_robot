"""replay_real.py against the fake arm and hand: it homes, replays, logs, and stops on faults."""

import json
import socket
import threading

import numpy as np
import pytest

import replay_real
from fake_robot import HOME_ARM_Q, FakeArm, FakeHand


def free_ports(n):
    ports = []
    for _ in range(n):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        ports.append(s.getsockname()[1])
        s.close()
    return ports


@pytest.fixture
def robot(tmp_path):
    """Fake arm and hand on free ports, and the replay_real arguments that reach them."""
    arm_cmd, arm_state, hand_cmd, hand_state = free_ports(4)
    config = tmp_path / "controller.json"
    config.write_text(json.dumps({"socket_port": arm_cmd, "publisher_port": arm_state}))
    made = []

    def start(frozen_arm=False):
        arm = FakeArm(arm_cmd, arm_state, start_q=HOME_ARM_Q + 0.1, frozen=frozen_arm)
        hand = FakeHand(hand_cmd, hand_state)
        for part in (arm, hand):
            part.start()
            made.append(part)
        args = ["--controller-config", str(config), "--hand-command-port", str(hand_cmd),
                "--hand-state-port", str(hand_state), "--log-dir", str(tmp_path / "logs")]
        return arm, hand, args

    yield start
    for part in made:
        part.close()


def demonstration(tmp_path, seconds=1.5):
    t = np.arange(0.0, seconds, 1 / 60)
    arm = HOME_ARM_Q + np.outer(np.sin(np.pi * t / seconds), [0.2, -0.1, 0.1, 0.0, 0.15, 0.2])
    hand = np.clip(np.outer(t / seconds, np.full(20, 0.6)), replay_real.HAND_LOWER, replay_real.HAND_UPPER)
    path = tmp_path / "demo.npz"
    np.savez(path, timestamp=t, arm_q=arm, hand_q_measured=hand, table_top_z=0.515)
    return path


def test_dry_run_moves_nothing(tmp_path, robot):
    arm, hand, args = robot()
    assert replay_real.main([str(demonstration(tmp_path))] + args) == 0
    assert arm.path is None and hand.received == 0


def test_full_replay_follows_the_plan_and_logs(tmp_path, robot, monkeypatch):
    arm, hand, args = robot()
    monkeypatch.setattr(replay_real, "HOME_MIN_DURATION", 1.0)  # keep the test short
    assert replay_real.main([str(demonstration(tmp_path)), "--send", "--yes"] + args) == 0
    logs = list((tmp_path / "logs").glob("*.npz"))
    assert len(logs) == 1
    log = np.load(logs[0])
    assert np.abs(log["arm_q"] - log["arm_target"]).max() < np.deg2rad(3)
    assert np.abs(log["hand_q"] - log["hand_target"]).max() < 0.2


def test_a_stuck_arm_is_stopped(tmp_path, robot):
    arm, hand, args = robot(frozen_arm=True)
    # The frozen arm never reaches the start pose: homing fails and the arm is told to stop.
    assert replay_real.main([str(demonstration(tmp_path)), "--send", "--yes", "--arm-only"] + args[:2]) == 2
    assert arm.stopped_count >= 1


def test_a_demonstration_faster_than_the_robot_is_refused(tmp_path):
    path = demonstration(tmp_path)
    d = dict(np.load(path))
    d["arm_q"] = d["arm_q"].copy()
    d["arm_q"][10:, 0] += 0.3  # a 0.3 rad jump in one 60 Hz step
    np.savez(path, **d)
    with pytest.raises(ValueError, match="speed"):
        replay_real.Demonstration(path, 1.0)
