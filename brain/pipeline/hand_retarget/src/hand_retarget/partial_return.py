"""Explicit offline recovery plan along only the prefix already sent to the robot."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from . import native_timing, robot_rollout as rollout


def prefix_waypoints(q,h,t,last,live_q,live_h):
    if not 0 < last[0] <= t[-1] or np.any(np.diff(t) <= 0):
        raise ValueError('Invalid executed-prefix time range.')
    keep=t < last[0]-1e-7
    return (np.vstack([live_q,last[27:33],q[keep][::-1]]),
            np.vstack([live_h,last[33:53],h[keep][::-1]]),
            np.r_[0.,1.,1.+3.*(last[0]-t[keep][::-1])])


def prepare(log, stopped_state, output):
    context = json.loads(log.read_text())
    state = json.loads(stopped_state.read_text())
    original = json.loads((log.parent/'manifest.json').read_text())
    if (not context.get('arm_stop_confirmed') or context.get('arm_only')
            or not context.get('failure') or context['failure'].get('layer') != 'application_safety_check'):
        raise ValueError('Explicit prefix recovery requires a stopped, logged application-guard event.')
    arm, hand = state['arm'], state['hand']
    if (any(arm.get(k) for k in ('protective_stopped', 'emergency_stopped', 'is_truncated'))
            or np.max(np.abs(arm['Qd'])) > .01
            or np.max(np.abs(hand['velocities'])) > .5):
        raise ValueError('Recovery requires stationary, unfaulted hardware feedback.')
    live_q, live_h = np.asarray(arm['Q']), np.asarray(hand['positions'])
    if live_q.shape!=(6,) or live_h.shape!=(20,) or not np.isfinite(np.r_[live_q,live_h]).all():
        raise ValueError('Invalid stopped joint state.')
    with np.load(log.with_suffix('.npz'), allow_pickle=False) as data:
        samples = data['samples']
        if samples.ndim != 2 or samples.shape[1] != 53 or len(samples) < 2:
            raise ValueError('No executed prefix available.')
        last = samples[-1]
        sent_arm = data['arm_command_targets'][-1,1:]
    # The guard snapshot is newer than the previous successful command-loop row;
    # compare stopped feedback against that observation, including braking travel.
    fault_q = np.asarray(context['failure']['arm_state']['Q'])
    fault_h = np.asarray(context['failure']['hand_state']['positions'])
    if (np.max(np.abs(live_q-fault_q)) > .08 or np.max(np.abs(live_h-fault_h)) > .2):
        raise ValueError('Stopped state moved outside the recorded endpoint neighbourhood.')
    np.testing.assert_allclose(sent_arm, context['last_sent_targets']['arm'], atol=1e-9,rtol=0)
    np.testing.assert_allclose(last[33:53], context['last_sent_targets']['hand'], atol=1e-9,rtol=0)
    path = Path(context['resolved_rollout'])
    if path.resolve().parent != log.resolve().parent:
        raise ValueError('Resolved prefix belongs to a different execution.')
    model = rollout.RobotSurfaceModel()
    if (not original.get('execution_ready') or model.sha256 != original['collision_model_sha256']
            or rollout.digest(log.parent/'robot_rollout.npz') != original['rollout_sha256']):
        raise ValueError('Original preparation or collision model changed.')
    with np.load(path, allow_pickle=False) as data:
        if str(data['source_rollout_sha256']) != original['rollout_sha256']:
            raise ValueError('Source path identity differs.')
        # Unsent future waypoints are never included in recovery.
        q,h,times=prefix_waypoints(data['arm_q'],data['hand_q'],data['timestamp'],last,live_q,live_h)
        normal, offset = data['plane_normal'].copy(), float(data['plane_offset'])
    with np.load(log.parent/'robot_rollout.npz', allow_pickle=False) as data:
        if not np.array_equal(normal, data['plane_normal']) or offset != float(data['plane_offset']):
            raise ValueError('Recorded table plane changed.')
    q = np.vstack([q[0], q]); h = np.vstack([h[0], h]); times = np.r_[0., 2.+times]
    timing = {}
    acceleration=min(3.,float(json.loads((rollout.ROOT/'arm/pc_ur_new.json').read_text())['joint_acceleration']))
    pq, ph, pt = rollout.rebase_cartesian_actions(model,q,h,times,live_q,live_h,1.,
        continuous=True,nominal_times=times,native=True,timing_report=timing,arm_acceleration=acceleration)
    _, clearance = rollout.validate_rebased_actions(model,pq,ph,pt,normal,offset,1.,
                                                     continuous=True,native=True,arm_acceleration=acceleration)
    output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(output/'robot_rollout.npz',arm_q=q,hand_q=h,timestamp=times,
        nominal_timestamp=times,plane_normal=normal,plane_offset=offset,
        interpolation=np.array(native_timing.INTERPOLATION),start_policy=np.array('live_cartesian_v2'))
    shutil.copyfile(log.parent/'capture.json',output/'capture.json')
    report=dict(status='completed',execution_ready=True,hardware_executed=False,
        mode='explicit_executed_prefix_recovery',return_start={'arm_q':live_q.tolist(),'hand_q':live_h.tolist()},
        return_scope='immediately_preceding_attempt_start',
        return_target={'arm_q':q[-1].tolist(),'hand_q':h[-1].tolist()},
        source_duration_s=float(times[-1]-times[1]),duration_s=float(pt[-1]),speed_scale=1.,
        rollout_sha256=rollout.digest(output/'robot_rollout.npz'),collision_model_sha256=model.sha256,
        calibration=original['calibration'],native_timing=timing,minimum_table_clearance_m=clearance,
        original_failed_log=str(log.resolve()),original_failed_log_sha256=rollout.digest(log),
        stopped_state_sha256=rollout.digest(stopped_state),last_sent_path_time_s=float(last[0]),
        purpose='Explicit recovery along a rechecked executed prefix; original failure record is unchanged.')
    (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-log',type=Path,required=True)
    parser.add_argument('--stopped-state',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(prepare(args.execution_log,args.stopped_state,args.out),indent=2))


if __name__=='__main__':main()
