#!/usr/bin/env python3
"""Level the DG5F palm with base XY, lifting tool0 first; guarded arm-only motion."""
import argparse
import json
from pathlib import Path
import signal
import socket
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import zmq

from hand_retarget.robot_rollout import (
    ROOT, ARM_LIMIT, RobotSurfaceModel, surface_clearance, check_finger_velocity, digest,
)
from hand_retarget.action_timing import retime_continuous, check_continuous


class Feedback:
    def __init__(self):
        config = json.loads((ROOT / 'arm/pc_ur_new.json').read_text())
        self.context = zmq.Context()
        self.arm = self.context.socket(zmq.SUB)
        self.arm.setsockopt(zmq.CONFLATE, 1)
        self.arm.setsockopt_string(zmq.SUBSCRIBE, '')
        self.arm.connect(f"tcp://127.0.0.1:{config['publisher_port']}")
        self.command = self.context.socket(zmq.PUB)
        self.command.setsockopt(zmq.LINGER, 500)
        self.command.bind(f"tcp://*:{config['socket_port']}")
        self.hand = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.hand.bind(('127.0.0.1', 5563))
        self.hand.setblocking(False)
        self.state = self.fingers = None
        self.arm_at = self.hand_at = 0.

    def read(self):
        while True:
            try:
                self.state = self.arm.recv_json(zmq.NOBLOCK)
                self.arm_at = time.monotonic()
            except zmq.Again:
                break
        while True:
            try:
                self.fingers = json.loads(self.hand.recvfrom(65535)[0])
                self.hand_at = time.monotonic()
            except BlockingIOError:
                break
        now = time.monotonic()
        if self.state is None or self.fingers is None:
            return None
        if max(now - self.arm_at, now - self.hand_at) > .25:
            raise RuntimeError('Feedback stale')
        if not -.05 <= time.time() - self.state.get('timestamp_ms', 0) / 1000 <= .25:
            raise RuntimeError('Arm producer timestamp stale')
        if any(self.state.get(k) for k in ('is_truncated', 'protective_stopped', 'emergency_stopped')):
            raise RuntimeError('Arm reports stop/fault')
        if not 0 < self.state.get('joint_target_watchdog_s', 0) <= .25:
            raise RuntimeError('Arm watchdog unavailable')
        if self.state.get('calibration_guard') is not None:
            raise RuntimeError('Use normal arm controller, not calibration controller')
        if not 0 <= self.fingers.get('measured_age_s', float('inf')) <= .25:
            raise RuntimeError('Finger measurement stale')
        q = np.asarray(self.state['Q'], float)
        qd = np.asarray(self.state['Qd'], float)
        h = np.asarray(self.fingers['positions'], float)
        hd = np.asarray(self.fingers['velocities'], float)
        if q.shape != (6,) or qd.shape != (6,) or h.shape != (20,) or not np.isfinite(np.r_[q, qd, h]).all():
            raise RuntimeError('Invalid joint state')
        if np.max(np.abs(qd)) > .3:
            raise RuntimeError('Measured arm overspeed')
        check_finger_velocity(hd)
        return q, h, qd

    def close(self):
        self.hand.close()
        self.arm.close()
        self.command.close()
        self.context.term()


def plan(model, q0, h0, normal, offset):
    frames = model.transforms(q0, h0)
    tool = frames['tool0']
    palm_normal = frames['rl_dg_palm'][:3, 0]  # Palm lies in local YZ.
    down = np.array([0., 0., -1.])
    angle = np.arccos(np.clip(palm_normal @ down, -1, 1))
    if angle > np.deg2rad(60):
        raise RuntimeError('Level adjustment exceeds 60 degrees; inspect orientation first')
    axis = np.cross(palm_normal, down)
    delta = np.zeros(3) if angle < 1e-8 else axis / np.linalg.norm(axis) * angle
    qs = [q0.copy()]
    for lift, fraction in [(x, 0.) for x in np.linspace(0, .01, 5)[1:]] + [(.01, x) for x in np.linspace(0, 1, 21)[1:]]:
        target = tool.copy()
        target[2, 3] += lift
        target[:3, :3] = Rotation.from_rotvec(delta * fraction).as_matrix() @ tool[:3, :3]
        def residual(q):
            actual = model.transforms(q, h0)['tool0']
            return np.r_[actual[:3, 3] - target[:3, 3],
                         .15 * Rotation.from_matrix(target[:3, :3].T @ actual[:3, :3]).as_rotvec()]
        result = least_squares(residual, qs[-1], bounds=(-ARM_LIMIT+1e-6, ARM_LIMIT-1e-6),
                               max_nfev=100, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        if np.linalg.norm(residual(result.x)) > 1e-5 or np.max(np.abs(result.x-qs[-1])) > .08:
            raise RuntimeError('Level path IK invalid or discontinuous')
        qs.append(result.x.copy())
    qs = np.asarray(qs)
    hs = np.repeat(h0[None], len(qs), axis=0)
    times = retime_continuous(qs, hs, np.linspace(0, 12, len(qs)), 1.)
    curves = check_continuous(qs, hs, times, 1.)
    clearance = float('inf')
    min_z = float('inf')
    for at in np.linspace(0, times[-1], 1001):
        q = curves[0](at)
        pose = model.transforms(q, h0)['tool0']
        clearance = min(clearance, surface_clearance(model, q, h0, normal, offset))
        min_z = min(min_z, pose[2, 3])
        if pose[2, 3] < tool[2, 3] - 1e-5 or np.linalg.norm(pose[:2, 3]-tool[:2, 3]) > .002:
            raise RuntimeError('Planned wrist leaves fixed-XY, non-descending-height corridor')
    if clearance < .03:
        raise RuntimeError('Level plan has less than 30 mm table clearance')
    return qs, times, curves[0], {'initial_tilt_deg': float(np.degrees(angle)),
        'wrist_lift_m': .01, 'duration_s': float(times[-1]), 'minimum_table_clearance_m': clearance,
        'planned_lowest_wrist_delta_z_m': float(min_z-tool[2, 3]), 'initial_tool_pose': tool.tolist()}


def checked_plane(folder, model_sha256):
    report = json.loads((folder / 'manifest.json').read_text())
    path = folder / 'robot_rollout.npz'
    if (not report.get('execution_ready') or digest(path) != report.get('rollout_sha256')
            or model_sha256 != report.get('collision_model_sha256')):
        raise ValueError('Calibration rollout or collision model is not validated.')
    with np.load(path, allow_pickle=False) as data:
        normal, offset = data['plane_normal'].copy(), float(data['plane_offset'])
    if (normal.shape != (3,) or not np.isfinite(np.r_[normal, offset]).all()
            or not np.isclose(np.linalg.norm(normal), 1., atol=1e-6)):
        raise ValueError('Invalid normalized table plane.')
    return normal, offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--calibration-rollout', type=Path, required=True,
                        help='Validated rollout folder containing the captured table plane')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    model = RobotSurfaceModel()
    normal, offset = checked_plane(args.calibration_rollout, model.sha256)
    feedback = Feedback()
    sent = False
    records = []
    report = {'completed': False, 'hand_commands': False}
    def stop_signal(*_):
        raise KeyboardInterrupt('Operator stopped leveling')
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        deadline = time.monotonic() + 5
        while feedback.read() is None:
            if time.monotonic() > deadline:
                raise RuntimeError('Feedback unavailable')
            time.sleep(.01)
        q0, h0, qd = feedback.read()
        if np.max(np.abs(qd)) > .01:
            raise RuntimeError('Start requires stationary arm')
        qs, times, curve, details = plan(model, q0, h0, normal, offset)
        report.update(details)
        np.savez_compressed(args.out/'plan.npz', arm_q=qs, timestamp=times, hand_q=h0,
                            plane_normal=normal, plane_offset=offset)
        print(json.dumps(report, indent=2), flush=True)
        if not args.send:
            return
        if input('Type SEND to lift wrist 1 cm and level palm: ').strip() != 'SEND':
            return
        q, h, _ = feedback.read()
        if np.max(np.abs(q-q0)) > .003 or np.max(np.abs(h-h0)) > .015:
            raise RuntimeError('Starting pose changed; plan again')
        z0 = report['initial_tool_pose'][2][3]
        measured_tcp_start = np.asarray(feedback.state['pos'][:3], float)
        start = time.monotonic()
        last = -1
        while True:
            elapsed = time.monotonic() - start
            target = curve(min(elapsed, times[-1]))
            q, h, _ = feedback.read()
            pose = model.transforms(q, h)['tool0']
            tcp = np.asarray(feedback.state['pos'][:3], float)
            if np.max(np.abs(q-target)) > .03 or np.max(np.abs(h-h0)) > .03:
                raise RuntimeError('Tracking error or unexpected finger motion')
            if pose[2, 3] < z0-.002 or tcp[2] < measured_tcp_start[2]-.002:
                raise RuntimeError('Measured wrist dropped below starting height tolerance')
            if surface_clearance(model, q, h, normal, offset) < .03:
                raise RuntimeError('Measured table clearance violation')
            feedback.command.send_json({'target_q': target.tolist()})
            sent = True
            palm = model.transforms(q, h)['rl_dg_palm']
            tilt = np.degrees(np.arccos(np.clip(-palm[2, 0], -1, 1)))
            records.append(np.r_[elapsed, q, target, tcp, tilt])
            if int(elapsed) != last:
                last = int(elapsed)
                print(f'{elapsed:.1f}s: palm tilt {tilt:.2f} deg; measured TCP height change {(tcp[2]-measured_tcp_start[2])*1000:.2f} mm', flush=True)
            if elapsed >= times[-1]+2:
                if tilt > 1 or np.max(np.abs(q-qs[-1])) > .01:
                    raise RuntimeError('Level target not reached')
                report.update(completed=True, final_tilt_deg=float(tilt),
                              measured_tcp_displacement_m=(tcp-measured_tcp_start).tolist())
                break
            time.sleep(1/60)
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if sent:
            for _ in range(3):
                feedback.command.send_json({'stop': True})
                time.sleep(.01)
            stopped = False
            deadline = time.monotonic()+1.5
            while time.monotonic() < deadline:
                try:
                    if np.max(np.abs(feedback.read()[2])) < .01:
                        stopped = True
                        break
                except Exception:
                    pass
                time.sleep(.02)
            report['stop_confirmed'] = stopped
            print('ARM STOP CONFIRMED' if stopped else 'STOP UNCONFIRMED: USE E-STOP', flush=True)
        np.savez_compressed(args.out/'execution.npz', samples=np.asarray(records))
        (args.out/'report.json').write_text(json.dumps(report, indent=2))
        feedback.close()


if __name__ == '__main__':
    main()
