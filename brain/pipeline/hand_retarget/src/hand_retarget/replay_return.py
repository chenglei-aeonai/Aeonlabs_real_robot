"""Prepare an explicit return along a completed replay, without commanding hardware."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import native_timing
from . import adaptive_timing
from . import robot_rollout as rollout


def reverse_completed_path(context, q, h, times, samples):
    """Require endpoint delivery and measured arrival, not merely a successful exit code."""
    rollout.check_path(q, h, times, None)
    if not context.get('arm_stop_confirmed') or context.get('arm_only'):
        raise ValueError('Return needs a stopped arm+hand recording.')
    failure = context.get('failure')
    if failure and failure.get('message') != 'Final target did not settle within 2 seconds; replay is incomplete.':
        raise ValueError('Resolve the previous execution fault before planning a return.')
    sent = context.get('last_sent_targets', {})
    if (not np.allclose(sent.get('arm', []), q[-1], atol=1e-6, rtol=0)
            or not np.allclose(sent.get('hand', []), h[-1], atol=1e-6, rtol=0)):
        raise ValueError('Previous replay did not send the endpoint.')
    samples = np.asarray(samples)
    if (samples.ndim != 2 or samples.shape[1] != 53 or not len(samples)
            or not np.isfinite(samples[-1]).all() or samples[-1, 0] < times[-1]
            or np.max(np.abs(samples[-1, 1:7] - q[-1])) > .03
            or np.max(np.abs(samples[-1, 7:27] - h[-1])) > .08):
        raise ValueError('Recording does not show arrival near the return start.')
    return q[::-1].copy(), h[::-1].copy(), (times[-1] - times[::-1]).copy()


def prepare(execution_log, output):
    context = json.loads(execution_log.read_text())
    folder = execution_log.parent
    original = json.loads((folder / 'manifest.json').read_text())
    path = Path(context['resolved_rollout'])
    # Do not accept a path from another preparation or collision model.
    if path.resolve().parent != folder.resolve():
        raise ValueError('Resolved path must belong to this execution directory.')
    model = rollout.RobotSurfaceModel()
    if (not original.get('execution_ready') or model.sha256 != original['collision_model_sha256']
            or rollout.digest(folder / 'robot_rollout.npz') != original['rollout_sha256']):
        raise ValueError('Original preparation/model changed; cannot prepare a recorded return.')
    with np.load(path, allow_pickle=False) as data:
        if (str(data['source_rollout_sha256']) != original['rollout_sha256']
                or str(data['interpolation']) not in (native_timing.INTERPOLATION,adaptive_timing.INTERPOLATION)):
            raise ValueError('Return requires a native replay from this preparation.')
        q, h, times = (data[key].copy() for key in ('arm_q', 'hand_q', 'timestamp'))
        normal, offset = data['plane_normal'].copy(), float(data['plane_offset'])
    with np.load(folder / 'robot_rollout.npz', allow_pickle=False) as data:
        if not np.array_equal(normal, data['plane_normal']) or offset != float(data['plane_offset']):
            raise ValueError('Recorded table plane differs from preparation.')
    with np.load(execution_log.with_suffix('.npz'), allow_pickle=False) as data:
        q, h, times = reverse_completed_path(context, q, h, times, data['samples'])
    expected = {'arm_q': q[0].tolist(), 'hand_q': h[0].tolist()}
    q = np.vstack([q[0], q]); h = np.vstack([h[0], h]); times = np.r_[0., 2. + times]
    config = json.loads((rollout.ROOT / 'arm/pc_ur_new.json').read_text())
    acceleration = min(3., float(config['joint_acceleration']))
    timing = {}
    pq, ph, pt = rollout.rebase_cartesian_actions(
        model, q, h, times, q[0], h[0], 1., continuous=True,
        nominal_times=times, native=True, timing_report=timing, arm_acceleration=acceleration)
    _, clearance = rollout.validate_rebased_actions(
        model, pq, ph, pt, normal, offset, 1., continuous=True, native=True,
        arm_acceleration=acceleration)
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / 'robot_rollout.npz', arm_q=q, hand_q=h,
        timestamp=times, nominal_timestamp=times, plane_normal=normal, plane_offset=offset,
        interpolation=np.array(native_timing.INTERPOLATION), start_policy=np.array('live_cartesian_v2'))
    shutil.copyfile(folder / 'capture.json', output / 'capture.json')
    report = dict(status='completed', mode='recorded_replay_return', execution_ready=True,
        hardware_executed=False, return_start=expected, speed_scale=1.,
        return_scope='immediately_preceding_attempt_start',
        return_target={'arm_q':q[-1].tolist(),'hand_q':h[-1].tolist()},
        source_duration_s=float(times[-1]-times[1]), duration_s=float(pt[-1]),
        native_timing=timing, minimum_table_clearance_m=clearance,
        collision_model_sha256=model.sha256, calibration=original['calibration'],
        rollout_sha256=rollout.digest(output / 'robot_rollout.npz'),
        execution_log=str(execution_log.resolve()), execution_log_sha256=rollout.digest(execution_log),
        recorded_path=str(path.resolve()), recorded_path_sha256=rollout.digest(path),
        purpose='Explicit return along the reversed prior palm/finger path, smoothed and revalidated; small live endpoint errors are rebased.')
    (output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-log', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.execution_log, args.out), indent=2))


if __name__ == '__main__':
    main()
