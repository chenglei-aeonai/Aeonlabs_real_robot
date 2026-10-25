"""Replay timing expressed as a uniform fraction of the source timeline.

The demo profile uses replay_real.py's pi rad/s joint speed limit and the
arm controller's acceleration setting (capped at 3 rad/s^2). It does not
invent a finger acceleration or jerk limit absent from that demo interface.
"""
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

INTERPOLATION = 'native_cubic_v4'
MIN_SPEED = .25
DEMO_JOINT_SPEED = np.pi


def endpoint_filter(values, sigma):
    """Preserve endpoints with a smooth correction, not isolated frame resets."""
    values = np.asarray(values, float)
    filtered = gaussian_filter1d(values, sigma, axis=0, mode='nearest')
    weight = np.linspace(0., 1., len(values)).reshape((-1,) + (1,) * (values.ndim - 1))
    filtered += (1 - weight) * (values[0] - filtered[0]) + weight * (values[-1] - filtered[-1])
    filtered[[0, -1]] = values[[0, -1]]
    return filtered


def native_curve(times, values, monotone=False):
    if monotone:
        slopes=PchipInterpolator(times,values).derivative()(times)
        slopes[[0,-1]]=0.
        return CubicHermiteSpline(times,values,slopes)
    zero = np.zeros(values.shape[1])
    return CubicSpline(times, values, bc_type=((1, zero), (1, zero)))


def peaks(curve):
    """Analytic cubic extrema, per joint: position min/max, speed, acceleration, jerk."""
    a, b, c, d = curve.c
    dt = np.diff(curve.x)[:, None]
    vertex = np.clip(np.divide(-b, 3 * a, out=np.zeros_like(a), where=a != 0), 0, dt)
    speed = np.max(np.maximum.reduce([np.abs(c), np.abs(3*a*dt**2 + 2*b*dt + c),
                                     np.abs(3*a*vertex**2 + 2*b*vertex + c)]), axis=0)
    acceleration = np.max(np.maximum(np.abs(2*b), np.abs(6*a*dt + 2*b)), axis=0)
    jerk = np.max(np.abs(6*a), axis=0)
    end = ((a*dt + b)*dt + c)*dt + d
    low, high = np.minimum(d, end), np.maximum(d, end)
    discriminant = b*b - 3*a*c
    for sign in (-1, 1):
        roots = np.divide(-b + sign*np.sqrt(np.maximum(0., discriminant)), 3*a,
                          out=np.zeros_like(a), where=a != 0)
        # If the cubic coefficient vanishes, the velocity is linear.
        linear = np.divide(-c, 2*b, out=np.zeros_like(b), where=b != 0)
        roots = np.where(a == 0, linear, roots)
        valid = (roots > 0) & (roots < dt) & ((a == 0) | (discriminant >= 0))
        value = ((a*roots + b)*roots + c)*roots + d
        low = np.minimum(low, np.where(valid, value, low))
        high = np.maximum(high, np.where(valid, value, high))
    return dict(minimum=np.min(low, axis=0), maximum=np.max(high, axis=0),
                speed=speed, acceleration=acceleration, jerk=jerk)


class NativeTimingError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__('No source speed at or above 0.25x passes the demo motion limits. '
                         'Refusing to silently stretch the action further. ' +
                         '; '.join(f"{item['speed_scale']:g}x: {', '.join(item['violations'])}"
                                   for item in report['candidates']))


def select_timing(q, h, source_times, requested_speed=1., arm_acceleration=3.):
    """Select 1x, 0.5x, 0.25x (or halvings of the request); never change node spacing."""
    t = np.asarray(source_times, float)
    if (t.ndim != 1 or len(t) < 2 or np.any(np.diff(t) <= 0)
            or not np.isfinite(t).all() or q.shape != (len(t), 6) or h.shape != (len(t), 20)
            or not np.isfinite(q).all() or not np.isfinite(h).all()):
        raise ValueError('Invalid native trajectory arrays/timestamps.')
    if not np.isfinite(requested_speed) or not MIN_SPEED <= requested_speed <= 1:
        raise ValueError('Native source speed must be between 0.25x and 1x.')
    if not np.isfinite(arm_acceleration) or not 0 < arm_acceleration <= 3:
        raise ValueError('Invalid arm acceleration budget.')
    t = t - t[0]
    curves = [native_curve(t, q), native_curve(t, h, monotone=True)]
    limits = [peaks(curve) for curve in curves]
    report = dict(timing_policy='source_rate_demo', source_duration_s=float(t[-1]),
                  requested_speed_scale=float(requested_speed), minimum_speed_scale=MIN_SPEED,
                  joint_speed_limit_rad_s=DEMO_JOINT_SPEED,
                  arm_acceleration_limit_rad_s2=float(arm_acceleration), candidates=[],
                  finger_acceleration_limit_rad_s2=None, jerk_limit_rad_s3=None)
    scale = float(requested_speed)
    while scale >= MIN_SPEED - 1e-10:
        v_arm, v_hand = (float(np.max(item['speed'])) * scale for item in limits)
        a_arm = float(np.max(limits[0]['acceleration'])) * scale**2
        violations = []
        if v_arm > DEMO_JOINT_SPEED + 1e-8:
            violations.append(f'arm speed {v_arm:.3f} > {DEMO_JOINT_SPEED:.3f} rad/s')
        if v_hand > DEMO_JOINT_SPEED + 1e-8:
            violations.append(f'hand speed {v_hand:.3f} > {DEMO_JOINT_SPEED:.3f} rad/s')
        if a_arm > arm_acceleration + 1e-8:
            violations.append(f'arm acceleration {a_arm:.3f} > {arm_acceleration:.3f} rad/s^2')
        report['candidates'].append(dict(speed_scale=scale, duration_s=float(t[-1]/scale),
                                         arm_peak_speed_rad_s=v_arm, hand_peak_speed_rad_s=v_hand,
                                         arm_peak_acceleration_rad_s2=a_arm,
                                         violations=violations, passed=not violations))
        if not violations:
            report.update(effective_speed_scale=scale, duration_s=float(t[-1]/scale))
            return t/scale, report
        scale /= 2
    raise NativeTimingError(report)


def checked_curves(q, h, t, arm_limit, hand_lower, hand_upper, arm_acceleration=3.):
    curves = [native_curve(t, q), native_curve(t, h, monotone=True)]
    for name, curve, lower, upper in zip(('arm', 'hand'), curves,
                                        (-arm_limit, hand_lower), (arm_limit, hand_upper)):
        bounds = peaks(curve)
        if np.any(bounds['minimum'] < lower-1e-6) or np.any(bounds['maximum'] > upper+1e-6):
            raise ValueError(f'Native {name} curve overshoots joint limits.')
        if np.max(bounds['speed']) > DEMO_JOINT_SPEED+1e-6:
            raise ValueError(f'Native {name} curve exceeds demo speed limit.')
        if name == 'arm' and np.max(bounds['acceleration']) > arm_acceleration+1e-6:
            raise ValueError('Native arm curve exceeds controller acceleration budget.')
    return curves


def servo_target(position, velocity, actual, gain, joint_limit):
    """Implement legacy v_ref + P*(q_ref-q) via the existing P-only target_q servo."""
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError('Invalid configured joint tracking gain.')
    speed = velocity + gain * (position - actual)
    target = position + velocity / gain
    if not np.isfinite(target).all() or np.any(np.abs(target) > joint_limit):
        raise ValueError('Feed-forward target exceeds joint limits.')
    if np.max(np.abs(speed)) > DEMO_JOINT_SPEED:
        raise ValueError('Feedback plus feed-forward exceeds demo joint speed budget.')
    return target
