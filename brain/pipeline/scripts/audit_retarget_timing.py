#!/usr/bin/env python3
"""Audit saved motion and timing without opening any robot connection.

Timing ablations are diagnostics, not trajectories authorized for execution.
Run with the pipeline Python environment. All frame indices are zero-based.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BPoly
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'brain/pipeline/hand_retarget/src'))
from hand_retarget.action_timing import continuous_curve, derivative_bounds, retime_continuous


def load_motion(path):
    with np.load(path, allow_pickle=False) as data:
        def get(*names):
            return next(data[name].astype(float) for name in names if name in data)
        t = get('timestamp', 'time_s')
        q = get('arm_q', 'arm_q_rad')
        h = get('hand_q_measured', 'hand_q', 'hand_q_rad')
    if (t.ndim != 1 or len(t) < 2 or q.shape != (len(t), 6)
            or h.shape != (len(t), 20) or not all(np.isfinite(a).all() for a in (t, q, h))
            or np.any(np.diff(t) <= 0)):
        raise ValueError(f'Invalid motion: {path}')
    return t, q, h


def motion_summary(path):
    t, q, h = load_motion(path)
    result = {'file': str(path), 'frames': len(t), 'duration_s': float(t[-1] - t[0]),
              'dt_min_s': float(np.diff(t).min()), 'dt_max_s': float(np.diff(t).max())}
    for name, values in (('arm', q), ('hand', h)):
        delta = np.diff(values, axis=0)
        rates = np.abs(delta) / np.diff(t)[:, None]
        frame, joint = np.unravel_index(rates.argmax(), rates.shape)
        result[name] = {'peak_secant_rad_s': float(rates[frame, joint]),
                        'peak_segment': [int(frame), int(frame + 1)],
                        'peak_joint_index': int(joint),
                        'max_adjacent_step_rad': float(np.abs(delta).max()),
                        'max_total_joint_travel_rad': float(np.abs(delta).sum(axis=0).max())}
    return result


def ablation(q, h, times, orders):
    """Same timing algorithm, retaining only the selected derivative constraints."""
    times = times.copy()
    for _ in range(60):
        factors = np.ones(len(times) - 1)
        for values, limits in ((q, (.15, .3, .6)), (h, (.35, .7, 1.4))):
            bounds = derivative_bounds(continuous_curve(times, values))
            for order, (bound, limit) in enumerate(zip(bounds[:orders], limits), 1):
                factors = np.maximum(factors, (bound / limit) ** (1 / order))
        if factors.max() <= 1 + 1e-8:
            return float(times[-1])
        times = np.r_[0, np.cumsum(np.diff(times) * np.where(factors > 1, factors * 1.01, 1.))]
    raise ValueError('Diagnostic retiming did not converge')


def feedback_summary(path):
    with np.load(path, allow_pickle=False) as data:
        s = data['samples']
        fb = data['hand_feedback'] if 'hand_feedback' in data else None
    if s.ndim != 2 or s.shape[1] != 53 or len(s) < 2:
        raise ValueError(f'Expected at least two 53-column rollout records: {path}')
    rates = np.abs(np.diff(s[:, 33:53], axis=0)) / np.diff(s[:, 0])[:, None]
    result = {'file': str(path), 'last_logged_elapsed_s': float(s[-1, 0]),
              'initial_hand_target_error_rad': float(np.abs(s[0, 7:27] - s[0, 33:53]).max()),
              'max_logged_hand_target_secant_rad_s': float(rates.max()),
              'max_logged_hand_tracking_error_rad': float(np.abs(s[:, 7:27] - s[:, 33:53]).max()),
              'max_measured_hand_range_rad': float(np.ptp(s[:, 7:27], axis=0).max())}
    if fb is not None and fb.ndim == 2 and len(fb) and fb.shape[1] == 61:
        i, j = np.unravel_index(np.nanargmax(np.abs(fb[:, 21:41])), fb[:, 21:41].shape)
        result.update(peak_reported_velocity_rad_s=float(fb[i, 21 + j]),
                      peak_velocity_joint_index=int(j),
                      peak_feedback_time_from_first_packet_s=float(fb[i, 0] - fb[0, 0]))
        averages = {}
        for window in (.1, .2, .5):
            before = np.searchsorted(fb[:, 0], fb[i, 0] - window)
            if before < i and fb[i, 0] > fb[before, 0]:
                averages[str(window)] = float((fb[i, 1 + j] - fb[before, 1 + j]) / (fb[i, 0] - fb[before, 0]))
        result['encoder_average_rad_s_ending_at_peak'] = averages
        result['encoder_average_caveat'] = 'A window average cannot rule out real short speed bursts.'
    rollout = path.parent / 'robot_rollout.npz'
    if rollout.exists():
        with np.load(rollout, allow_pickle=False) as data:
            rest = str(data.get('interpolation', '')) == 'rest_quintic_v2'
            absolute = str(data.get('start_policy', 'captured_absolute')) == 'captured_absolute'
        if rest and absolute:
            t, _, h = load_motion(rollout)
            zero = np.zeros(20)
            curve = BPoly.from_derivatives(t, [[value, zero, zero] for value in h])
            elapsed, targets = s[:, 0], s[:, 33:53]
            def residual(lead):
                u = np.clip(elapsed / lead, 0, 1)
                w = u**3 * (10 - 15*u + 6*u*u)
                predicted = np.where((elapsed < lead)[:, None],
                                     targets[0] + (h[0] - targets[0]) * w[:, None],
                                     curve(np.maximum(elapsed - lead, 0)))
                return float(np.abs(predicted - targets).max())
            fitted = minimize_scalar(residual, bounds=(.01, max(10., elapsed[-1] * 2)),
                                     method='bounded', options={'xatol': 1e-12})
            if fitted.fun < 1e-6 and elapsed[-1] > fitted.x:
                at = elapsed[-1] - fitted.x
                result['reconstructed_absolute_handover'] = {
                    'fitted_live_lead_s': float(fitted.x), 'max_target_residual_rad': float(fitted.fun),
                    'path_time_at_last_sample_s': float(at), 'first_retarget_frame_time_s': float(t[1]),
                    'last_logged_segment_index': int(np.searchsorted(t, at) - 1),
                    'peak_target_speed_at_last_sample_rad_s': float(np.abs(curve.derivative()(at)).max()),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--retarget', type=Path, required=True, help='Result directory containing capture and retargeted_hand')
    parser.add_argument('--demo', type=Path, action='append', default=[])
    parser.add_argument('--rollout', type=Path)
    parser.add_argument('--feedback-log', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('offline_legacy_replay', ROOT / 'replay_real.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)  # Definitions only; no Arm/Hand instances, no sockets.

    raw_path = args.retarget / 'retargeted_hand.npz'
    export_path = args.retarget / 'robot_demonstration.npz'
    paths = args.demo + [export_path]
    report = {'hardware_connected': False, 'motion_sent': False, 'frame_indices': 'zero-based',
              'motions': [motion_summary(path) for path in paths + [raw_path]]}
    for entry, path in zip(report['motions'], paths):
        try:
            legacy.Demonstration(path, 1.)
            entry['legacy_file_check'] = 'pass'
        except (ValueError, KeyError) as error:
            entry['legacy_file_check'] = str(error)

    t, q, h = load_motion(raw_path)
    capture = json.loads((args.retarget / 'capture.json').read_text())['robot_state']
    cap_q, cap_h = np.asarray(capture['arm_q_rad']), np.asarray(capture['hand_q_rad'])
    error = np.abs(h[0] - cap_h)
    report['first_frame'] = {'arm_error_rad': float(np.abs(q[0] - cap_q).max()),
                             'hand_error_rad': float(error.max()), 'hand_error_deg': float(np.degrees(error.max())),
                             'hand_joint_index': int(error.argmax()), 'hand_error_each_rad': error.tolist()}
    filtered = gaussian_filter1d(h, sigma=1.5, axis=0, mode='nearest')
    filtered[[0, -1]] = h[[0, -1]]
    full_q, full_h = np.vstack([cap_q, q]), np.vstack([cap_h, filtered])
    ordinary = np.r_[0, 2 + t - t[0]]
    fourfold = np.r_[0, 2 + 4 * (t - t[0])]
    report['timing_at_scale_one'] = {
        'without_fourfold_s': float(retime_continuous(full_q, full_h, ordinary, 1.)[-1]),
        'with_fourfold_s': float(retime_continuous(full_q, full_h, fourfold, 1.)[-1]),
        'without_capture_lead_s': float(retime_continuous(q, filtered, t - t[0], 1.)[-1]),
        'velocity_only_ablation_s': ablation(full_q, full_h, fourfold, 1),
        'velocity_acceleration_ablation_s': ablation(full_q, full_h, fourfold, 2),
        'ablation_warning': 'Removing limits is diagnostic only; these are not executable outputs.',
    }
    if args.rollout:
        report['saved_rollout'] = motion_summary(args.rollout)
        saved_t, saved_q, saved_h = load_motion(args.rollout)
        report['saved_rollout']['matches_current_prepared_positions'] = bool(
            saved_q.shape == full_q.shape and saved_h.shape == full_h.shape
            and np.allclose(saved_q, full_q) and np.allclose(saved_h, full_h))
        report['saved_rollout']['recomputed_duration_error_s'] = float(
            report['timing_at_scale_one']['with_fourfold_s'] - saved_t[-1])
    report['feedback'] = [feedback_summary(path) for path in args.feedback_log]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
