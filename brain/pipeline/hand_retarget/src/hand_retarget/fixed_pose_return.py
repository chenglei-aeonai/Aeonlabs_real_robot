"""Prepare an absolute return to a saved, successfully measured levelled pose."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import robot_rollout as rollout


def fixed_waypoints(start_q, start_h, goal_q, goal_h):
    """Keep the saved endpoint fixed and include two seconds of endpoint hold."""
    q = np.array([start_q, goal_q, goal_q], dtype=float)
    h = np.array([start_h, goal_h, goal_h], dtype=float)
    times = rollout.retime_actions(q, h, np.array([0., 8., 10.]), 1.)
    rollout.check_actions(q, h, times, 1.)
    return q, h, times


def prepare(state_path, reference, calibration_rollout, output):
    state = json.loads(state_path.read_text())
    record = json.loads((reference / 'report.json').read_text())
    original = json.loads((calibration_rollout / 'manifest.json').read_text())
    if not record.get('completed') or not record.get('stop_confirmed'):
        raise ValueError('Reference must be a successfully completed measured levelled pose.')
    arm, hand = state['arm'], state['hand']
    if (any(arm.get(k) for k in ('protective_stopped', 'emergency_stopped', 'is_truncated'))
            or np.max(np.abs(arm['Qd'])) > .01):
        raise ValueError('Return preparation requires an unfaulted stationary arm.')
    with np.load(reference / 'execution.npz', allow_pickle=False) as data:
        goal_q = data['samples'][-1, 1:7].copy()
    with np.load(reference / 'plan.npz', allow_pickle=False) as data:
        goal_h = data['hand_q'].copy()
    q, h, times = fixed_waypoints(arm['Q'], hand['positions'], goal_q, goal_h)
    model = rollout.RobotSurfaceModel()
    if (not original.get('execution_ready') or model.sha256 != original['collision_model_sha256']
            or rollout.digest(calibration_rollout / 'robot_rollout.npz') != original['rollout_sha256']):
        raise ValueError('Validated calibration rollout/model changed.')
    with np.load(calibration_rollout / 'robot_rollout.npz', allow_pickle=False) as data:
        normal, offset = data['plane_normal'].copy(), float(data['plane_offset'])
    curves, clearance = rollout.validate_rebased_actions(model, q, h, times, normal, offset, 1.)
    start = model.transforms(q[0], h[0])['rl_dg_palm'][:3, 3]
    wrist = []
    for at in np.linspace(0, times[-1], 1201):
        a, b = (curve(at) for curve in curves)
        frames = model.transforms(a, b)
        if np.linalg.norm(frames['rl_dg_palm'][:3, 3] - start) > .3:
            raise ValueError('Absolute return leaves the 30 cm local neighbourhood.')
        wrist.append(frames['tool0'][2, 3])
    if min(wrist) < min(wrist[0], wrist[-1]) - .002:
        raise ValueError('Absolute return dips below both endpoint wrist heights.')
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'robot_rollout.npz', arm_q=q, hand_q=h,
        timestamp=times, plane_normal=normal, plane_offset=offset,
        interpolation=np.array('fixed_pose_quintic_v1'), start_policy=np.array('captured_absolute'))
    shutil.copyfile(calibration_rollout / 'capture.json', output / 'capture.json')
    report = dict(status='completed', execution_ready=True, hardware_executed=False,
        mode='fixed_pose_return', return_scope='original_measured_levelled_pose',
        feedback_profile='demo_pi_rad_s',
        return_start={'arm_q':q[0].tolist(), 'hand_q':h[0].tolist()},
        return_target={'arm_q':goal_q.tolist(), 'hand_q':goal_h.tolist()},
        speed_scale=1., duration_s=float(times[-1]), motion_duration_s=float(times[-2]),
        endpoint_hold_s=float(times[-1]-times[-2]),
        minimum_table_clearance_m=clearance, wrist_height_change_m=float(wrist[-1]-wrist[0]),
        rollout_sha256=rollout.digest(output / 'robot_rollout.npz'),
        collision_model_sha256=model.sha256, calibration=original['calibration'],
        reference=str(reference.resolve()),
        reference_sha256={name:rollout.digest(reference / name)
                          for name in ('report.json', 'execution.npz', 'plan.npz')},
        state_sha256=rollout.digest(state_path),
        purpose='Absolute saved joint endpoint; never translate the return goal with current pose.')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--calibration-rollout', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.state, args.reference, args.calibration_rollout, args.out), indent=2))


if __name__ == '__main__':
    main()
