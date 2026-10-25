#!/usr/bin/env python3
"""Replay a robot demonstration (hand_retarget's export) on the real UR5e + DG5F.

Talks to the deployment that already runs on the ur5 PC, unmodified:

- arm: ``impedance_controller pc_ur_new.json`` over ZMQ JSON (commands on
  ``socket_port``, state on ``publisher_port``). The whole arm trajectory is sent
  once as a timed path; the controller follows it with velocity feed-forward
  (``speedJ(v_spline + p * (q_spline - Q))``) and holds the last point at the end.
- hand: ``dg5f_policy_ros_bridge.py`` over localhost UDP (targets on 5562, state
  on 5563), streamed at the demonstration's 60 Hz.

Without ``--send`` it only connects, checks and prints: nothing moves. With it,
each motion waits for the operator: type SEND to arm, then Enter to move the arm
to the start pose (a slow spline), Enter to bring the hand to its start pose, and
Enter to replay. Any fault, and Ctrl-C, stops the arm and holds the hand where it
is. Keep the e-stop in hand.

Dependencies: numpy and pyzmq (both in simtoolreal_real's .venv on the ur5 PC).
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

ARM_DOF, HAND_DOF = 6, 20
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
HAND_JOINTS = tuple(f"rj_dg_{f}_{j}" for f in range(1, 6) for j in range(1, 5))

# The right hand's limits, as enforced by dg5f_policy_ros_bridge.py (rad).
HAND_LOWER = np.array([
    -0.3839724354, -3.1415926536, 0.0, 0.0,
    -0.4188790205, 0.0, 0.0, 0.0,
    -0.6108652382, 0.0, 0.0, 0.0,
    -0.6108652382, 0.0, 0.0, 0.0,
    -0.0174532925, -0.4188790205, 0.0, 0.0,
])
HAND_UPPER = np.array([
    0.8901179185, 0.0, 1.5707963268, 1.5707963268,
    0.6108652382, 2.0071286398, 1.5707963268, 1.5707963268,
    0.6108652382, 1.9547687622, 1.5707963268, 1.5707963268,
    0.4188790205, 1.9024088847, 1.5707963268, 1.5707963268,
    1.0471975512, 0.6108652382, 1.5707963268, 1.5707963268,
])
ARM_LIMIT = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])
MAX_JOINT_SPEED = np.pi  # rad/s, UR5e and DG5F

# Safety thresholds (the same as the existing policy and teleop deployments).
STATE_TIMEOUT = 0.25  # s without a fresh state -> stop
HOME_MAX_DISTANCE = 1.9  # rad: refuse to home from farther than this
HOME_SPEED = 0.15  # rad/s for the move to the start pose
HOME_MIN_DURATION = 5.0  # s
HOME_TOLERANCE = 0.05  # rad
HAND_RAMP_SPEED = 0.5  # rad/s for the hand's move to its start pose
HAND_START_TOLERANCE = 0.1  # rad
ARM_TRACKING_LIMIT = np.deg2rad(10.0)  # rad between the planned and the measured arm
RATE_HZ = 60.0

# World (simtoolreal) -> UR controller base: the base link sits at ROBOT_BASE, and the
# controller's `base` frame is turned 180 degrees about z from it.
ROBOT_BASE = np.array([0.0, 0.6, 0.55])
WORLD_TO_UR_BASE = np.diag([-1.0, -1.0, 1.0])


class Abort(RuntimeError):
    """A safety check failed: stop the robot."""


# -- the demonstration --------------------------------------------------------------

class Demonstration:
    def __init__(self, path: Path, speed: float):
        d = np.load(path)
        self.name = path.stem
        self.t = np.asarray(d["timestamp"], dtype=np.float64) / speed
        self.arm = np.asarray(d["arm_q"], dtype=np.float64)
        hand_key = "hand_q_measured" if "hand_q_measured" in d.files else "hand_q"
        self.hand = np.asarray(d[hand_key], dtype=np.float64)
        self.speed = speed
        self.tube = None
        if "tube_position" in d.files:
            self.tube = (np.asarray(d["tube_position"]), float(d["tube_yaw"]), float(d["tube_length"]),
                         float(d["tube_radius"]))
        self.table_top_z = float(d["table_top_z"]) if "table_top_z" in d.files else None
        self.check()

    def check(self) -> None:
        n = len(self.t)
        if self.arm.shape != (n, ARM_DOF) or self.hand.shape != (n, HAND_DOF):
            raise ValueError(f"shapes: t {self.t.shape}, arm {self.arm.shape}, hand {self.hand.shape}")
        if not (np.all(np.isfinite(self.arm)) and np.all(np.isfinite(self.hand)) and np.all(np.isfinite(self.t))):
            raise ValueError("the demonstration contains NaN or inf")
        if np.any(np.diff(self.t) <= 0):
            raise ValueError("timestamps are not strictly increasing")
        if np.any(np.abs(self.arm) > ARM_LIMIT):
            raise ValueError("arm angles outside the UR5e's joint limits")
        if np.any(self.hand < HAND_LOWER - 1e-6) or np.any(self.hand > HAND_UPPER + 1e-6):
            raise ValueError("hand angles outside the DG5F's joint limits")
        for name, q in (("arm", self.arm), ("hand", self.hand)):
            speed = np.abs(np.diff(q, axis=0)).max(0) / np.diff(self.t).min()
            if speed.max() > MAX_JOINT_SPEED * 1.05:
                raise ValueError(f"{name} joint speed {speed.max():.2f} rad/s over the {MAX_JOINT_SPEED:.2f} limit")

    def at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Arm and hand targets at time t (s since the replay started), linear between rows."""
        t = min(max(t, 0.0), self.t[-1])
        arm = np.array([np.interp(t, self.t, self.arm[:, j]) for j in range(ARM_DOF)])
        hand = np.array([np.interp(t, self.t, self.hand[:, j]) for j in range(HAND_DOF)])
        return arm, hand

    def describe(self) -> str:
        lines = [
            f"{self.name}: {len(self.t)} rows, {self.t[-1]:.2f} s at speed {self.speed:g}",
            "  start arm (deg): " + ", ".join(f"{n} {v:.1f}" for n, v in zip(ARM_JOINTS, np.degrees(self.arm[0]))),
        ]
        if self.table_top_z is not None:
            lines.append(f"  table top in the plan: {100 * (self.table_top_z - ROBOT_BASE[2]):+.1f} cm from the robot base")
        if self.tube is not None:
            position, yaw, length, radius = self.tube
            ur = WORLD_TO_UR_BASE @ (position - ROBOT_BASE)
            heading = np.degrees(yaw + np.pi) % 180.0
            lines.append(
                f"  tube ({100 * length:.0f} x {200 * radius:.0f} cm): centre at x {100 * ur[0]:.1f} cm, "
                f"y {100 * ur[1]:.1f} cm in the UR base frame, axis {heading:.0f} deg from the base x axis"
            )
        return "\n".join(lines)


# -- the arm controller -------------------------------------------------------------

class Arm:
    def __init__(self, command_port: int, state_port: int):
        self.context = zmq.Context()
        self.command = self.context.socket(zmq.PUB)
        self.command.setsockopt(zmq.LINGER, 500)
        self.command.bind(f"tcp://*:{command_port}")
        self.state = self.context.socket(zmq.SUB)
        self.state.setsockopt(zmq.CONFLATE, 1)
        self.state.setsockopt_string(zmq.SUBSCRIBE, "")
        self.state.connect(f"tcp://127.0.0.1:{state_port}")
        self.q: Optional[np.ndarray] = None
        self.q_at: Optional[float] = None
        time.sleep(1.0)  # let the controller's subscriber connect before the first command

    def poll(self) -> None:
        while True:
            try:
                message = self.state.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            q = np.asarray(message.get("Q", []), dtype=np.float64) if isinstance(message, dict) else np.empty(0)
            if q.shape[0] >= ARM_DOF and np.all(np.isfinite(q[:ARM_DOF])):
                self.q, self.q_at = q[:ARM_DOF].copy(), time.monotonic()

    def fresh(self) -> np.ndarray:
        self.poll()
        if self.q is None or time.monotonic() - self.q_at > STATE_TIMEOUT:
            raise Abort("arm state missing or stale: is impedance_controller running?")
        return self.q.copy()

    def wait(self, timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if self.q is not None:
                return self.q.copy()
            time.sleep(0.01)
        raise Abort("no arm state: start ./impedance_controller pc_ur_new.json and put the UR in Remote Control")

    def send_path(self, times: np.ndarray, path: np.ndarray) -> None:
        self.command.send_json({"time": np.asarray(times).tolist(), "path": np.asarray(path).tolist()})

    def stop(self) -> None:
        for _ in range(3):
            self.command.send_json({"stop": True})
            time.sleep(0.01)

    def close(self) -> None:
        self.command.close()
        self.state.close()
        self.context.term()


# -- the hand bridge ----------------------------------------------------------------

class Hand:
    def __init__(self, command_port: int, state_port: int):
        self.state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state.setblocking(False)
        self.state.bind(("127.0.0.1", state_port))
        self.command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.endpoint = ("127.0.0.1", command_port)
        self.sequence = 0
        self.q: Optional[np.ndarray] = None
        self.q_at: Optional[float] = None

    def poll(self) -> None:
        while True:
            try:
                payload, _ = self.state.recvfrom(65535)
            except BlockingIOError:
                return
            try:
                message = json.loads(payload)
            except ValueError:
                continue
            q = np.asarray(message.get("positions", []), dtype=np.float64)
            if message.get("type") == "hand_state" and q.shape == (HAND_DOF,) and np.all(np.isfinite(q)):
                self.q, self.q_at = q, time.monotonic()

    def fresh(self) -> np.ndarray:
        self.poll()
        if self.q is None or time.monotonic() - self.q_at > STATE_TIMEOUT:
            raise Abort("hand state missing or stale: is dg5f_policy_ros_bridge.py running?")
        return self.q.copy()

    def wait(self, timeout: float = 5.0) -> np.ndarray:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.poll()
            if self.q is not None:
                return self.q.copy()
            time.sleep(0.01)
        raise Abort("no hand state: start the DG5F driver and dg5f_policy_ros_bridge.py")

    def send(self, target: np.ndarray) -> None:
        target = np.clip(np.asarray(target, dtype=np.float64), HAND_LOWER, HAND_UPPER)
        self.sequence += 1
        message = {"type": "hand_target", "sequence": self.sequence, "positions": target.tolist()}
        self.command.sendto(json.dumps(message, separators=(",", ":")).encode(), self.endpoint)

    def hold_measured(self, seconds: float = 0.5) -> None:
        """Stream the measured position, so the bridge brakes instead of chasing an old target."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            if self.q is not None:
                self.send(self.q)
            time.sleep(0.01)

    def close(self) -> None:
        self.state.close()
        self.command.close()


# -- the replay ---------------------------------------------------------------------

def ask(prompt: str, auto: bool) -> None:
    if auto:
        print(prompt + " [--yes]")
        return
    input(prompt)


def home_arm(arm: Arm, target: np.ndarray, auto: bool) -> None:
    q = arm.fresh()
    distance = np.abs(target - q).max()
    print(f"Arm: largest joint move to the start pose {np.degrees(distance):.1f} deg")
    if distance > HOME_MAX_DISTANCE:
        raise Abort(f"the start pose is {distance:.2f} rad away (limit {HOME_MAX_DISTANCE}): bring the arm closer first")
    duration = max(HOME_MIN_DURATION, distance / HOME_SPEED)
    ask(f"Enter: move the arm to the start pose in {duration:.1f} s", auto)
    q = arm.fresh()
    arm.send_path([0.0, duration / 2, duration], [q, (q + target) / 2, target])
    deadline = time.monotonic() + duration + 1.5
    while time.monotonic() < deadline:
        arm.fresh()
        time.sleep(0.02)
    error = np.abs(arm.fresh() - target).max()
    if error > HOME_TOLERANCE:
        raise Abort(f"the arm stopped {np.degrees(error):.1f} deg from the start pose")
    print(f"Arm at the start pose (within {np.degrees(error):.1f} deg)")


def home_hand(hand: Hand, target: np.ndarray, auto: bool) -> None:
    q = hand.fresh()
    distance = np.abs(target - q).max()
    ask(f"Enter: move the hand to its start pose (largest joint move {np.degrees(distance):.0f} deg)", auto)
    streamed = hand.fresh()
    step = HAND_RAMP_SPEED / RATE_HZ
    deadline = time.monotonic() + distance / HAND_RAMP_SPEED + 3.0
    while time.monotonic() < deadline:
        streamed = streamed + np.clip(target - streamed, -step, step)
        hand.send(streamed)
        if np.abs(streamed - target).max() < 1e-9 and np.abs(hand.fresh() - target).max() < HAND_START_TOLERANCE:
            print("Hand at its start pose")
            return
        time.sleep(1.0 / RATE_HZ)
    raise Abort(f"the hand stopped {np.degrees(np.abs(hand.fresh() - target).max()):.0f} deg from its start pose")


def replay(demo: Demonstration, arm: Optional[Arm], hand: Optional[Hand], hand_error_limit: float,
           auto: bool) -> dict:
    ask(f"Enter: replay {demo.t[-1]:.1f} s" + (" (arm" if arm else " (") + (" + hand)" if hand else ")"), auto)
    log = {k: [] for k in ("t", "arm_q", "arm_target", "hand_q", "hand_target")}
    if arm is not None:
        path = demo.arm.copy()
        path[0] = arm.fresh()  # start the spline exactly where the arm is (within the homing tolerance)
        arm.send_path(demo.t - demo.t[0], path)
    start = time.monotonic()
    next_tick = start
    while True:
        t = time.monotonic() - start
        arm_target, hand_target = demo.at(t)
        if hand is not None:
            hand.send(hand_target)
        arm_q = arm.fresh() if arm is not None else np.full(ARM_DOF, np.nan)
        hand_q = hand.fresh() if hand is not None else np.full(HAND_DOF, np.nan)
        if arm is not None and np.abs(arm_q - arm_target).max() > ARM_TRACKING_LIMIT:
            j = int(np.argmax(np.abs(arm_q - arm_target)))
            raise Abort(f"arm {ARM_JOINTS[j]} is {np.degrees(abs(arm_q[j] - arm_target[j])):.1f} deg off the plan at {t:.2f} s")
        if hand is not None and np.abs(hand_q - hand_target).max() > hand_error_limit:
            j = int(np.argmax(np.abs(hand_q - hand_target)))
            raise Abort(f"hand {HAND_JOINTS[j]} is {np.degrees(abs(hand_q[j] - hand_target[j])):.0f} deg off the plan at {t:.2f} s")
        for key, value in zip(log, (t, arm_q, arm_target, hand_q, hand_target)):
            log[key].append(value)
        if t >= demo.t[-1]:
            break
        next_tick += 1.0 / RATE_HZ
        time.sleep(max(0.0, next_tick - time.monotonic()))
    print("Replay finished.")
    if hand is not None:
        # Keep squeezing: holding the *measured* position would relax the grasp and drop the object.
        print("The hand keeps its last target. Enter: finish (the hand then holds its measured position)")
        final = demo.hand[-1]
        if auto:
            hand.send(final)
        else:
            import select

            while not select.select([sys.stdin], [], [], 1.0 / RATE_HZ)[0]:
                hand.send(final)
                hand.fresh()
            sys.stdin.readline()
    return {k: np.array(v) for k, v in log.items()}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("demonstration", type=Path, help="robot demonstration .npz from hand_retarget")
    parser.add_argument("--send", action="store_true", help="actually move the robot (default: check and print only)")
    parser.add_argument("--speed", type=float, default=1.0, help="time scaling, e.g. 0.25 for a quarter speed")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--arm-only", action="store_true", help="leave the hand alone (no bridge needed)")
    group.add_argument("--hand-only", action="store_true", help="leave the arm alone (no arm controller needed)")
    parser.add_argument("--hand-error-limit-deg", type=float, default=35.0,
                        help="stop when a finger is this far from its target (contact makes some error normal)")
    parser.add_argument("--controller-config", type=Path, default=None,
                        help="impedance_controller's JSON (for socket_port / publisher_port); default 5555 / 5556")
    parser.add_argument("--hand-command-port", type=int, default=5562)
    parser.add_argument("--hand-state-port", type=int, default=5563)
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parent / "logs")
    parser.add_argument("--yes", action="store_true", help=argparse.SUPPRESS)  # tests only: no prompts
    args = parser.parse_args(argv)
    if not 0.0 < args.speed <= 1.0:
        parser.error("--speed must be in (0, 1]")

    demo = Demonstration(args.demonstration, args.speed)
    print(demo.describe())
    command_port, state_port = 5555, 5556
    if args.controller_config:
        config = json.loads(args.controller_config.read_text())
        command_port, state_port = int(config["socket_port"]), int(config["publisher_port"])

    arm = None if args.hand_only else Arm(command_port, state_port)
    hand = None if args.arm_only else Hand(args.hand_command_port, args.hand_state_port)
    armed = False  # nothing is ever sent before the operator arms the robot
    try:
        if arm is not None:
            q = arm.wait()
            print(f"Arm connected; start pose is {np.degrees(np.abs(demo.arm[0] - q).max()):.1f} deg away (largest joint)")
        if hand is not None:
            q = hand.wait()
            print(f"Hand connected; start pose is {np.degrees(np.abs(demo.hand[0] - q).max()):.0f} deg away (largest joint)")
        if not args.send:
            print("Dry run: nothing was sent. Add --send to move the robot.")
            return 0
        if not args.yes and input("Type SEND to arm the robot: ").strip() != "SEND":
            print("Not armed.")
            return 1
        armed = True
        if arm is not None:
            home_arm(arm, demo.arm[0], args.yes)
        if hand is not None:
            home_hand(hand, demo.hand[0], args.yes)
        log = replay(demo, arm, hand, np.deg2rad(args.hand_error_limit_deg), args.yes)
        args.log_dir.mkdir(parents=True, exist_ok=True)
        path = args.log_dir / f"{demo.name}_{time.strftime('%Y%m%d_%H%M%S')}.npz"
        np.savez(path, speed=args.speed, **log)
        print(f"Log: {path}")
        return 0
    except (Abort, KeyboardInterrupt) as error:
        print(f"\nSTOP: {error or 'Ctrl-C'}")
        if armed and arm is not None:
            arm.stop()
        if armed and hand is not None:
            hand.hold_measured()
        return 2
    finally:
        if hand is not None:
            if armed:
                hand.hold_measured(0.1)
            hand.close()
        if arm is not None:
            arm.close()


if __name__ == "__main__":
    sys.exit(main())
