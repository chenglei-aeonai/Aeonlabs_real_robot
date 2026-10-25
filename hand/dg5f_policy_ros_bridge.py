#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Optional

import numpy as np
import rclpy
from control_msgs.msg import MultiDOFCommand
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


HAND_DOF = 20
DEFAULT_CURRENT_WARNING_MA = 170.0
DEFAULT_CURRENT_WARNING_CLEAR_MA = 130.0
DEFAULT_CURRENT_WARNING_PERIOD_S = 1.0
ANSI_BOLD_BRIGHT_YELLOW = "\033[1;93m"
ANSI_RESET = "\033[0m"
JOINT_NAMES = [
    f"rj_dg_{finger}_{joint}"
    for finger in range(1, 6)
    for joint in range(1, 5)
]
LEFT_TO_RIGHT_SIGN = np.array(
    [
        -1,
        -1,
        -1,
        -1,
        -1,
        +1,
        +1,
        +1,
        -1,
        +1,
        +1,
        +1,
        -1,
        +1,
        +1,
        +1,
        -1,
        -1,
        +1,
        +1,
    ],
    dtype=np.float64,
)
LEFT_LOWER = np.array(
    [
        -0.8901179185, 0.0, -1.5707963268, -1.5707963268,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.6108652382, 0.0, 0.0, 0.0,
        -0.4188790205, 0.0, 0.0, 0.0,
        -1.0471975512, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)
LEFT_UPPER = np.array(
    [
        0.3839724354, 3.1415926536, 0.0, 0.0,
        0.4188790205, 2.0071286398, 1.5707963268, 1.5707963268,
        0.6108652382, 1.9547687622, 1.5707963268, 1.5707963268,
        0.6108652382, 1.9024088847, 1.5707963268, 1.5707963268,
        0.0174532925, 0.4188790205, 1.5707963268, 1.5707963268,
    ],
    dtype=np.float64,
)
RIGHT_LOWER = np.minimum(
    LEFT_TO_RIGHT_SIGN * LEFT_LOWER,
    LEFT_TO_RIGHT_SIGN * LEFT_UPPER,
)
RIGHT_UPPER = np.maximum(
    LEFT_TO_RIGHT_SIGN * LEFT_LOWER,
    LEFT_TO_RIGHT_SIGN * LEFT_UPPER,
)


def warning_text(message: str) -> str:
    if (
        (sys.stdout.isatty() or sys.stderr.isatty())
        and "NO_COLOR" not in os.environ
    ):
        return f"{ANSI_BOLD_BRIGHT_YELLOW}{message}{ANSI_RESET}"
    return message


class Dg5fPolicyBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("dg5f_policy_bridge")
        self.args = args
        self.state_endpoint = (args.policy_address, args.state_port)
        self.state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setblocking(False)
        self.command_socket.bind((args.command_address, args.command_port))

        command_type = (
            MultiDOFCommand
            if args.controller_mode == "pid"
            else JointTrajectory
        )
        self.publisher = self.create_publisher(command_type, args.command_topic, 10)
        self.subscription = self.create_subscription(
            JointState,
            args.joint_state_topic,
            self._on_joint_state,
            qos_profile_sensor_data,
        )
        self.create_timer(1.0 / args.bridge_hz, self._tick)

        self.measured_q: Optional[np.ndarray] = None
        self.measured_qd: Optional[np.ndarray] = None
        self.measured_current_ma: Optional[np.ndarray] = None
        self.current_limit_active = np.zeros(HAND_DOF, dtype=bool)
        self.last_current_warning_at = 0.0
        self.last_state_at: Optional[float] = None
        self.last_command_at: Optional[float] = None
        self.last_target: Optional[np.ndarray] = None
        self.command_active = False
        self.state_sequence = 0
        self.watchdog_announced = False
        self.last_command_log_at = 0.0
        self.last_logged_measured: Optional[np.ndarray] = None
        self.no_subscriber_announced = False

        self.get_logger().info(
            f"State -> udp://{args.policy_address}:{args.state_port}; "
            f"commands <- udp://{args.command_address}:{args.command_port}"
        )
        self.get_logger().info(
            f"ROS state: {args.joint_state_topic}; command: {args.command_topic} "
            f"({args.controller_mode} mode)"
        )
        self.get_logger().info(
            "Motor-current warning: "
            f"activate above {args.current_warning_ma:g} mA, "
            f"clear at or below {args.current_warning_clear_ma:g} mA"
        )

    def close(self) -> None:
        self.state_socket.close()
        self.command_socket.close()

    def _on_joint_state(self, message: JointState) -> None:
        indices = {name: index for index, name in enumerate(message.name)}
        if any(name not in indices for name in JOINT_NAMES):
            return
        q = np.array(
            [message.position[indices[name]] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        if len(message.velocity) == len(message.name):
            qd = np.array(
                [message.velocity[indices[name]] for name in JOINT_NAMES],
                dtype=np.float64,
            )
        else:
            qd = np.zeros(HAND_DOF, dtype=np.float64)
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(qd)):
            return
        self.measured_q = q
        self.measured_qd = qd
        if len(message.effort) == len(message.name):
            current_ma = np.array(
                [message.effort[indices[name]] for name in JOINT_NAMES],
                dtype=np.float64,
            )
            if np.all(np.isfinite(current_ma)):
                self.measured_current_ma = current_ma
                self._update_current_warning(current_ma)
        self.last_state_at = time.monotonic()

    def _update_current_warning(self, current_ma: np.ndarray) -> None:
        absolute_current = np.abs(current_ma)
        previous_active = self.current_limit_active.copy()
        self.current_limit_active[
            absolute_current > self.args.current_warning_ma
        ] = True
        self.current_limit_active[
            absolute_current <= self.args.current_warning_clear_ma
        ] = False

        now = time.monotonic()
        newly_active = self.current_limit_active & ~previous_active
        if np.any(self.current_limit_active) and (
            np.any(newly_active)
            or now - self.last_current_warning_at
            >= self.args.current_warning_period
        ):
            details = "; ".join(
                (
                    f"{JOINT_NAMES[index]}="
                    f"{current_ma[index]:.0f} mA"
                )
                for index in np.flatnonzero(self.current_limit_active)
            )
            self.get_logger().warning(
                warning_text("Motor current limiter active: " + details)
            )
            self.last_current_warning_at = now

        if np.any(previous_active) and not np.any(self.current_limit_active):
            self.get_logger().info(
                "Motor currents returned below the limiter-clear threshold."
            )

    def _send_state(self) -> None:
        if self.measured_q is None or self.measured_qd is None:
            return
        self.state_sequence += 1
        payload = {
            "type": "hand_state",
            "sequence": self.state_sequence,
            "measured_age_s": time.monotonic() - self.last_state_at,
            "positions": self.measured_q.tolist(),
            "velocities": self.measured_qd.tolist(),
        }
        if self.measured_current_ma is not None:
            payload["currents_ma"] = self.measured_current_ma.tolist()
        self.state_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode(),
            self.state_endpoint,
        )

    def _receive_latest_command(self) -> Optional[np.ndarray]:
        latest = None
        while True:
            try:
                payload, _ = self.command_socket.recvfrom(65535)
            except BlockingIOError:
                return latest
            try:
                message = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if message.get("type") != "hand_target":
                continue
            target = np.asarray(message.get("positions"), dtype=np.float64)
            if target.shape != (HAND_DOF,) or not np.all(np.isfinite(target)):
                self.get_logger().warning(
                    warning_text("Rejected malformed hand target.")
                )
                continue
            latest = target

    def _safe_target(self, requested: np.ndarray) -> np.ndarray:
        reference = self.last_target
        if reference is None:
            reference = self.measured_q
        if reference is None:
            raise RuntimeError("Cannot command before receiving /joint_states.")

        # Usually this envelope is exactly [RIGHT_LOWER, RIGHT_UPPER].  If the
        # measured hand starts outside those conservative command limits,
        # extend the envelope only as far as that measured/reference value.
        # This permits a monotonic recovery toward the valid range without a
        # discontinuous jump to the boundary and never permits moving farther
        # outside than the starting state.
        recovery_lower = np.minimum(RIGHT_LOWER, reference)
        recovery_upper = np.maximum(RIGHT_UPPER, reference)
        requested = np.clip(requested, recovery_lower, recovery_upper)
        delta = np.clip(
            requested - reference,
            -self.args.max_command_step_rad,
            self.args.max_command_step_rad,
        )
        return reference + delta

    def publish_target(self, target: np.ndarray) -> None:
        if self.args.controller_mode == "pid":
            message = MultiDOFCommand()
            message.dof_names = JOINT_NAMES
            message.values = np.asarray(target, dtype=float).tolist()
            message.values_dot = [0.0] * HAND_DOF
            self.publisher.publish(message)
            return
        message = JointTrajectory()
        message.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = np.asarray(target, dtype=float).tolist()
        seconds = self.args.trajectory_duration
        whole_seconds = int(seconds)
        point.time_from_start.sec = whole_seconds
        point.time_from_start.nanosec = int(
            round((seconds - whole_seconds) * 1e9)
        )
        message.points = [point]
        self.publisher.publish(message)

    def publish_measured_hold(self) -> bool:
        if self.measured_q is None:
            return False
        self.publish_target(self.measured_q)
        self.last_target = self.measured_q.copy()
        return True

    def _tick(self) -> None:
        self._send_state()
        command = self._receive_latest_command()
        now = time.monotonic()

        if command is not None:
            if (
                self.last_state_at is None
                or now - self.last_state_at > self.args.state_timeout
            ):
                self.get_logger().error(
                    "Rejected command: /joint_states is stale."
                )
                return
            target = self._safe_target(command)
            self.last_target = target
            self.last_command_at = now
            self.command_active = True
            self.watchdog_announced = False
            if now - self.last_command_log_at >= 1.0:
                absolute_error = np.abs(target - self.measured_q)
                worst_index = int(np.argmax(absolute_error))
                max_error = float(absolute_error[worst_index])
                measured_motion = (
                    0.0
                    if self.last_logged_measured is None
                    else float(
                        np.max(np.abs(self.measured_q - self.last_logged_measured))
                    )
                )
                self.get_logger().info(
                    "Policy target accepted: "
                    f"max |target-measured|={max_error:.3f} rad on "
                    f"{JOINT_NAMES[worst_index]} "
                    f"(target={target[worst_index]:.3f}, "
                    f"measured={self.measured_q[worst_index]:.3f}); "
                    f"measured motion={measured_motion:.3f} rad; "
                    f"target[0:4]={np.round(target[:4], 3).tolist()}; "
                    f"measured[0:4]={np.round(self.measured_q[:4], 3).tolist()}"
                )
                self.last_logged_measured = self.measured_q.copy()
                self.last_command_log_at = now

        if (
            self.command_active
            and self.last_command_at is not None
            and now - self.last_command_at > self.args.command_timeout
        ):
            self.publish_measured_hold()
            self.command_active = False
            if not self.watchdog_announced:
                self.get_logger().warning(
                    warning_text(
                        "Policy command watchdog expired; "
                        "holding measured position."
                    )
                )
                self.watchdog_announced = True
            return

        # The Tesollo PID examples stream the complete reference continuously.
        # Republish the last accepted target at bridge_hz instead of emitting a
        # single ROS message for each (potentially much slower) policy packet.
        if self.command_active and self.last_target is not None:
            subscriber_count = self.publisher.get_subscription_count()
            if subscriber_count == 0:
                if not self.no_subscriber_announced:
                    self.get_logger().error(
                        "No ROS controller subscribes to the command topic; "
                        "the hand cannot move. Check the PID driver launch and "
                        "controller mode/topic."
                    )
                    self.no_subscriber_announced = True
            else:
                self.no_subscriber_announced = False
                self.publish_target(self.last_target)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Bridge DG5F ROS 2 joint states/trajectories to the hand-only "
            "SimToolReal policy process over localhost UDP."
        )
    )
    parser.add_argument("--policy-address", default="127.0.0.1")
    parser.add_argument("--state-port", type=int, default=5563)
    parser.add_argument("--command-address", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=5562)
    parser.add_argument(
        "--joint-state-topic",
        default="/dg5f_right/joint_states",
        help="DG5F joint-state topic (default: /dg5f_right/joint_states).",
    )
    parser.add_argument(
        "--controller-mode",
        choices=("pid", "trajectory"),
        default="pid",
        help=(
            "ROS command interface. 'pid' is the safe streaming default; "
            "'trajectory' retains the legacy spline-controller path."
        ),
    )
    parser.add_argument(
        "--command-topic",
        default=None,
        help=(
            "Override the command topic. Defaults to "
            "/dg5f_right/rj_dg_pospid/reference in pid mode or the namespaced "
            "joint_trajectory topic in trajectory mode."
        ),
    )
    parser.add_argument("--bridge-hz", type=float, default=100.0)
    parser.add_argument("--command-timeout", type=float, default=0.25)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument("--max-command-step-rad", type=float, default=0.12)
    parser.add_argument(
        "--current-warning-ma",
        type=float,
        default=DEFAULT_CURRENT_WARNING_MA,
    )
    parser.add_argument(
        "--current-warning-clear-ma",
        type=float,
        default=DEFAULT_CURRENT_WARNING_CLEAR_MA,
    )
    parser.add_argument(
        "--current-warning-period",
        type=float,
        default=DEFAULT_CURRENT_WARNING_PERIOD_S,
    )
    parser.add_argument(
        "--trajectory-duration",
        type=float,
        default=0.1,
        help=(
            "Positive JointTrajectory time_from_start. Zero-time points can "
            "already be expired when the controller receives them "
            "(default: 0.1 s)."
        ),
    )
    args, ros_args = parser.parse_known_args()
    if args.command_topic is None:
        args.command_topic = (
            "/dg5f_right/rj_dg_pospid/reference"
            if args.controller_mode == "pid"
            else "/dg5f_right/dg5f_right_controller/joint_trajectory"
        )
    for name in (
        "bridge_hz",
        "command_timeout",
        "state_timeout",
        "max_command_step_rad",
        "current_warning_ma",
        "current_warning_clear_ma",
        "current_warning_period",
    ):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.current_warning_clear_ma >= args.current_warning_ma:
        parser.error(
            "--current-warning-clear-ma must be less than "
            "--current-warning-ma"
        )
    if args.trajectory_duration < 0.0:
        parser.error("--trajectory-duration must be non-negative")
    return args, ros_args


def main() -> None:
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = Dg5fPolicyBridge(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C: holding measured hand position.")
    finally:
        deadline = time.monotonic() + 0.5
        while rclpy.ok() and time.monotonic() < deadline:
            node.publish_measured_hold()
            rclpy.spin_once(node, timeout_sec=0.01)
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
