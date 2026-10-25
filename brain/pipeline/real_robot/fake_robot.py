"""Stand-ins for impedance_controller and dg5f_policy_ros_bridge, speaking the same protocol.

For trying replay_real.py without a robot:

    python fake_robot.py            # in one terminal
    python replay_real.py demo.npz --send

The fake arm follows timed paths (linear between points, feed-forward plus the same
proportional term as the real controller) and stops on {"stop": true}; the fake hand
moves each joint toward its target at a limited speed. Neither simulates contact.
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time

import numpy as np
import zmq

HOME_ARM_Q = np.array([-1.5708, -1.05, 1.95, -0.9, 1.571, -2.618])


class FakeArm(threading.Thread):
    def __init__(self, command_port=5555, state_port=5556, rate_hz=500.0, p_gain=3.0, start_q=HOME_ARM_Q, frozen=False):
        super().__init__(daemon=True)
        self.context = zmq.Context()
        self.commands = self.context.socket(zmq.SUB)
        self.commands.setsockopt_string(zmq.SUBSCRIBE, "")
        self.commands.connect(f"tcp://127.0.0.1:{command_port}")
        self.state = self.context.socket(zmq.PUB)
        self.state.bind(f"tcp://*:{state_port}")
        self.rate_hz, self.p_gain = rate_hz, p_gain
        self.q = np.array(start_q, dtype=float)
        self.frozen = frozen  # a stuck arm, to test the tracking check
        self.path = None  # (times, points, start)
        self.target = None
        self.target_at = 0.
        self.watchdog_s = .2
        self.stopped_count = 0
        self.running = True

    def run(self):
        dt = 1.0 / self.rate_hz
        while self.running:
            while True:
                try:
                    message = self.commands.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                if message.get("stop"):
                    self.path, self.stopped_count = None, self.stopped_count + 1
                    self.target = None
                elif "path" in message:
                    self.path = (np.array(message["time"]), np.array(message["path"]), time.monotonic())
                    self.target = None
                elif "target_q" in message:
                    self.target = np.asarray(message['target_q'], float)
                    self.target_at = time.monotonic()
                    self.path = None
            before = self.q.copy()
            if self.target is not None and time.monotonic() - self.target_at > self.watchdog_s:
                self.target = None
            if self.target is not None and not self.frozen:
                self.q += dt * self.p_gain * (self.target - self.q)
            if self.path is not None and not self.frozen:
                times, points, start = self.path
                t = time.monotonic() - start
                ref = np.array([np.interp(t, times, points[:, j]) for j in range(6)])
                ref_next = np.array([np.interp(t + dt, times, points[:, j]) for j in range(6)])
                self.q = self.q + dt * ((ref_next - ref) / dt + self.p_gain * (ref - self.q))
            self.state.send_json({"Q": self.q.tolist(), "Qd": ((self.q-before)/dt).tolist(),
                                  "timestamp_ms": time.time()*1000,
                                  "joint_target_watchdog_s": self.watchdog_s,
                                  "robot_state_age_s": 0., "rtde_connected": True,
                                  "protective_stopped": False, "emergency_stopped": False})
            time.sleep(dt)

    def close(self):
        self.running = False
        self.join(1.0)
        self.commands.close()
        self.state.close()
        self.context.term()


class FakeHand(threading.Thread):
    def __init__(self, command_port=5562, state_port=5563, rate_hz=100.0, max_speed=np.pi):
        super().__init__(daemon=True)
        self.commands = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.commands.bind(("127.0.0.1", command_port))
        self.commands.setblocking(False)
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.state_endpoint = ("127.0.0.1", state_port)
        self.rate_hz, self.max_speed = rate_hz, max_speed
        self.q = np.zeros(20)
        self.target = None
        self.received = 0
        self.running = True

    def run(self):
        dt = 1.0 / self.rate_hz
        sequence = 0
        while self.running:
            while True:
                try:
                    payload, _ = self.commands.recvfrom(65535)
                except BlockingIOError:
                    break
                message = json.loads(payload)
                if message.get("type") == "hand_target":
                    self.target, self.received = np.array(message["positions"]), self.received + 1
            before = self.q.copy()
            if self.target is not None:
                step = self.max_speed * dt
                self.q = self.q + np.clip(self.target - self.q, -step, step)
            sequence += 1
            state = {"type": "hand_state", "sequence": sequence, "positions": self.q.tolist(),
                     "velocities": ((self.q-before)/dt).tolist(), "measured_age_s": 0.}
            self.out.sendto(json.dumps(state).encode(), self.state_endpoint)
            time.sleep(dt)

    def close(self):
        self.running = False
        self.join(1.0)
        self.commands.close()
        self.out.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-arm", action="store_true")
    parser.add_argument("--no-hand", action="store_true")
    args = parser.parse_args()
    parts = ([] if args.no_arm else [FakeArm()]) + ([] if args.no_hand else [FakeHand()])
    for part in parts:
        part.start()
    print("Fake " + " and ".join(type(p).__name__.removeprefix("Fake").lower() for p in parts) + " running; Ctrl-C to quit")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        for part in parts:
            part.close()


if __name__ == "__main__":
    main()
