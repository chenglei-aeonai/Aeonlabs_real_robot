from types import SimpleNamespace

import numpy as np
import pytest

from hand_retarget import adaptive_timing as adaptive, native_timing as native


def trajectory():
    t = np.linspace(0, 4.4, 45)
    q = np.zeros((len(t), 6)); q[:, 0] = .15*t
    h = np.zeros((len(t), 20)); h[:, 2] = .2 + .1*np.sin(t)
    h[:, 3] = .3 + .15*np.sin(2*t)
    return q, h, t


def test_local_clock_preserves_waypoints_and_reserves_settling():
    q, h, source = trajectory(); before = q.copy(), h.copy()
    t, report = adaptive.select_timing(q, h, source)
    assert t[-1] + adaptive.SETTLE_RESERVE <= 8
    assert np.all(np.diff(t) >= np.diff(source))
    assert np.ptp(np.diff(t)/np.diff(source)) > .01
    assert not report['timeline_uniform']
    for original, current in zip(before, (q, h)):
        np.testing.assert_array_equal(original, current)
    for values, monotone, v_limit, a_limit in [(q, False, np.pi, 2.4), (h, True, adaptive.HAND_SPEED, adaptive.HAND_ACCELERATION)]:
        curve = native.native_curve(t, values, monotone)
        grid = np.linspace(0, t[-1], 20001)
        np.testing.assert_allclose(curve(t), values, atol=1e-12)
        np.testing.assert_allclose(curve([0, t[-1]], 1), 0, atol=1e-12)
        assert np.all(np.abs(curve(grid, 1)).max(axis=0) <= v_limit)
        assert np.all(np.abs(curve(grid, 2)).max(axis=0) <= a_limit)


def test_settling_fits_the_deadline_without_shortening_motion():
    assert adaptive.settle_time(5) == 1.5
    assert adaptive.settle_time(6.447) == pytest.approx(1.303)
    assert adaptive.settle_time(7.25) == .5
    with pytest.raises(ValueError): adaptive.settle_time(7.251)


def test_optimizer_success_does_not_bypass_independent_bounds(monkeypatch):
    q, h, source = trajectory()
    monkeypatch.setattr(adaptive, 'minimize', lambda *a, **k: SimpleNamespace(
        x=np.zeros(len(source)-1), success=True, message='unverified optimizer result'))
    t, report = adaptive.select_timing(q, h, source)
    adaptive.checked_timing(q, h, t, 2.4)


def test_unachievable_deadline_refuses_instead_of_truncating():
    q, h, source = trajectory()
    with pytest.raises(ValueError, match='budget'):
        adaptive.select_timing(q, h, source, execution_budget=5)


def test_feasibility_not_solver_status_is_used(monkeypatch):
    q, h, source = trajectory()
    real = adaptive.minimize
    def stopped_solver(*args, **kwargs):
        result = real(*args, **kwargs)
        result.success = False
        return result
    monkeypatch.setattr(adaptive, 'minimize', stopped_solver)
    t, report = adaptive.select_timing(q, h, source)
    assert not report['optimizer_success']
    adaptive.checked_timing(q, h, t, 2.4)


def test_live_feedback_failures_abort_planning(monkeypatch):
    clock = iter(range(10000))
    monkeypatch.setattr(adaptive.time, 'monotonic', lambda: next(clock))
    def lost_feedback(): raise RuntimeError('Lost live feedback')
    with pytest.raises(RuntimeError, match='Lost live feedback'):
        adaptive.select_timing(*trajectory(), tick=lost_feedback)


def test_invalid_nonfinite_input_and_speed_rejected():
    q, h, t = trajectory()
    with pytest.raises(ValueError): adaptive.select_timing(q, h, t, requested_speed=0)
    h[3, 1] = np.nan
    with pytest.raises(ValueError): adaptive.select_timing(q, h, t)
