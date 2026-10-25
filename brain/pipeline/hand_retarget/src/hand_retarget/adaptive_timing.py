"""Allocate time locally without changing joint waypoints or increasing guards."""
import time

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize

from . import native_timing as native

INTERPOLATION = 'adaptive_cubic_v5'
EXECUTION_BUDGET = 8.
SETTLE_TIME = 1.5
MIN_SETTLE_TIME = .5
STOP_MARGIN = .25
SETTLE_RESERVE = MIN_SETTLE_TIME + STOP_MARGIN
HAND_SPEED = np.full(20, .9)
HAND_ACCELERATION = np.full(20, 10.)  # Planning constraints, not manufacturer limits.
# Real feedback showed the thumb distal joint overshooting at a 0.84 rad/s
# reference. Allocate more time to that joint without changing the measured stop.
HAND_SPEED[3] = .5
HAND_ACCELERATION[3] = 4.


def settle_time(duration, execution_budget=EXECUTION_BUDGET):
    """Use remaining time for fine settling; always reserve time to stop."""
    remaining = execution_budget - duration - STOP_MARGIN
    if not np.isfinite(remaining) or remaining < MIN_SETTLE_TIME - 1e-8:
        raise ValueError('Motion leaves insufficient settling/stop budget.')
    return min(SETTLE_TIME, remaining)


def interval_peaks(curve):
    a, b, c, _ = curve.c
    dt = np.diff(curve.x)[:, None]
    vertex = np.clip(np.divide(-b, 3*a, out=np.zeros_like(a), where=a != 0), 0, dt)
    velocity = np.maximum.reduce([abs(c), abs(3*a*dt**2 + 2*b*dt + c),
                                  abs(3*a*vertex**2 + 2*b*vertex + c)])
    acceleration = np.maximum(abs(2*b), abs(6*a*dt + 2*b))
    return velocity, acceleration


def ratios(q, h, dt, arm_acceleration):
    times = np.r_[0., np.cumsum(dt)]
    values = []
    for joints, monotone, speed, acceleration in (
            (q, False, native.DEMO_JOINT_SPEED, arm_acceleration),
            (h, True, HAND_SPEED, HAND_ACCELERATION)):
        v, a = interval_peaks(native.native_curve(times, joints, monotone))
        values.extend(((v/speed).max(axis=1), (a/acceleration).max(axis=1)))
    return np.asarray(values)


def checked_timing(q, h, times, arm_acceleration, execution_budget=EXECUTION_BUDGET):
    if times[-1] + SETTLE_RESERVE > execution_budget + 1e-8:
        raise ValueError(f'Local timing needs {times[-1]:.3f} s motion + '
                         f'{SETTLE_RESERVE:.3f} s settling reserve; '
                         f'exceeds {execution_budget:.3f} s execution budget.')
    if np.max(ratios(q, h, np.diff(times), arm_acceleration)) > 1 + 1e-6:
        raise ValueError('Local timing violates its additional planning limits.')


def select_timing(q, h, source_times, requested_speed=1., arm_acceleration=3.,
                  tick=lambda: None, execution_budget=EXECUTION_BUDGET):
    source = np.asarray(source_times, float)
    if (source.ndim != 1 or len(source) < 2 or q.shape != (len(source), 6)
            or h.shape != (len(source), 20) or not np.isfinite(source).all()
            or not np.isfinite(q).all() or not np.isfinite(h).all()
            or np.any(np.diff(source) <= 0)):
        raise ValueError('Invalid local timing inputs.')
    if not np.isfinite(requested_speed) or not .25 <= requested_speed <= 1:
        raise ValueError('Requested source speed must be in [0.25, 1].')
    if not np.isfinite(arm_acceleration) or not 0 < arm_acceleration <= 3:
        raise ValueError('Invalid arm acceleration budget.')
    if not np.isfinite(execution_budget) or execution_budget <= SETTLE_RESERVE:
        raise ValueError('Invalid execution budget.')
    # Leave 20% of the configured acceleration for tracking; speed guards stay unchanged.
    planning_acceleration = .8 * arm_acceleration
    base = np.diff(source) / requested_speed
    if base.sum() + SETTLE_RESERVE > execution_budget:
        raise ValueError('Requested source rate leaves no room for the execution budget.')
    dt = base.copy()
    last_tick = time.monotonic()

    def evaluate(intervals):
        nonlocal last_tick
        if time.monotonic() - last_tick >= .05:
            tick(); last_tick = time.monotonic()
        return ratios(q, h, intervals, planning_acceleration)

    # Construct a feasible starting clock by expanding only offending intervals.
    for _ in range(60):
        r = evaluate(dt)
        factors = np.maximum.reduce([np.ones(len(dt)), r[0], np.sqrt(r[1]),
                                     r[2], np.sqrt(r[3])])
        if factors.max() <= 1 + 1e-7:
            break
        dt *= np.where(factors > 1, np.maximum(factors, gaussian_filter1d(factors, 1)) * 1.005, 1)
    initial = dt.copy()
    objective = lambda x: np.sum(base*np.exp(x))
    bounds = [(0., max(3., float(np.max(np.log(dt/base))) + 1))] * len(dt)
    constraints = {'type': 'ineq', 'fun': lambda x: 1-evaluate(base*np.exp(x)).ravel()}
    result = minimize(objective, np.log(dt/base), bounds=bounds, constraints=constraints,
                      method='SLSQP', options={'maxiter': 120, 'ftol': 1e-5})
    # Optimizer exit status is not a safety certificate. Independently check both
    # clocks, repairing only numerical/remaining constraint violations by slowing.
    candidates = []
    def verify(proposal):
        if not np.isfinite(proposal).all():
            return
        r = evaluate(proposal)
        factor = max(1., r[0].max(), np.sqrt(r[1].max()), r[2].max(), np.sqrt(r[3].max()))
        candidate = proposal * factor * 1.001
        if np.max(evaluate(candidate)) <= 1 + 1e-7:
            candidates.append(candidate)
    verify(base*np.exp(result.x))
    verify(initial)
    if not candidates:
        raise ValueError('No independently verified local timing solution.')
    dt = min(candidates, key=np.sum)
    # If numerical repair left us near the deadline, refine from the feasible
    # clock instead of weakening a limit or silently exceeding the deadline.
    if dt.sum() + SETTLE_RESERVE > execution_budget - .04:
        refined = minimize(objective, np.log(dt/base), bounds=bounds, constraints=constraints,
                           method='SLSQP', options={'maxiter': 80, 'ftol': 1e-6})
        verify(base*np.exp(refined.x))
        dt = min(candidates, key=np.sum)
    times = np.r_[0., np.cumsum(dt)]
    checked_timing(q, h, times, planning_acceleration, execution_budget)
    ca, ch = native.native_curve(times, q), native.native_curve(times, h, True)
    pa, ph = native.peaks(ca), native.peaks(ch)
    duration = float(times[-1]); source_duration = float(source[-1]-source[0])
    report = dict(timing_policy='local_time_allocation', timeline_uniform=False,
        source_duration_s=source_duration, requested_speed_scale=float(requested_speed),
        effective_speed_scale=source_duration/duration, duration_s=duration,
        maximum_execution_s=execution_budget, settling_reserve_s=SETTLE_RESERVE,
        arm_acceleration_limit_rad_s2=arm_acceleration,
        arm_planning_acceleration_rad_s2=planning_acceleration,
        hand_planning_speed_rad_s=HAND_SPEED.tolist(), hand_planning_acceleration_rad_s2=HAND_ACCELERATION.tolist(),
        maximum_settle_time_s=settle_time(duration, execution_budget),
        stop_margin_s=STOP_MARGIN,
        optimizer_success=bool(result.success), optimizer_message=str(result.message),
        maximum_interval_stretch=float((dt/base).max()), waypoint_positions_unchanged=True,
        candidates=[dict(speed_scale=source_duration/duration, duration_s=duration,
            passed=True, violations=[], arm_peak_speed_rad_s=float(pa['speed'].max()),
            hand_peak_speed_rad_s=float(ph['speed'].max()),
            arm_peak_acceleration_rad_s2=float(pa['acceleration'].max()),
            hand_peak_acceleration_rad_s2=float(ph['acceleration'].max()))])
    tick()
    return times, report
