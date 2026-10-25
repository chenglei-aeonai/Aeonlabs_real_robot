import copy

import numpy as np
import pytest

from hand_retarget.replay_return import reverse_completed_path
from hand_retarget.robot_rollout import check_return_start, SafetyStop


@pytest.fixture
def completed():
    q = np.zeros((3, 6)); q[:, 0] = [0., .01, .03]
    h = np.zeros((3, 20)); h[:, 0] = [0., .02, .03]
    t = np.array([0., .7, 2.])
    context = {'arm_stop_confirmed': True, 'arm_only': False,
               'last_sent_targets': {'arm': q[-1].tolist(), 'hand': h[-1].tolist()}}
    samples = np.array([np.r_[2.1, q[-1], h[-1], q[-1], h[-1]]])
    return context, q, h, t, samples


def test_return_reverses_path_and_nonuniform_intervals(completed):
    context, q, h, t, samples = completed
    a, b, times = reverse_completed_path(*completed)
    np.testing.assert_array_equal(a, q[::-1])
    np.testing.assert_array_equal(b, h[::-1])
    np.testing.assert_allclose(times, [0., 1.3, 2.])
    # Outputs own their memory and cannot mutate the recorded forward path.
    a[0] += 1
    assert np.max(q) == .03


@pytest.mark.parametrize('fault', ['early_end', 'unsent_endpoint', 'stop_unconfirmed', 'overspeed', 'not_arrived'])
def test_incomplete_or_faulted_execution_cannot_authorize_return(completed, fault):
    context, q, h, t, samples = copy.deepcopy(completed)
    if fault == 'early_end': samples[-1, 0] = 1.9
    if fault == 'unsent_endpoint': context['last_sent_targets']['arm'][0] = .02
    if fault == 'stop_unconfirmed': context['arm_stop_confirmed'] = False
    if fault == 'overspeed': context['failure'] = {'message': 'Measured finger overspeed'}
    if fault == 'not_arrived': samples[-1, 1] += .031
    with pytest.raises(ValueError): reverse_completed_path(context, q, h, t, samples)


def test_historical_fine_settle_failure_still_requires_measured_arrival(completed):
    context, q, h, t, samples = completed
    context['failure'] = {'message': 'Final target did not settle within 2 seconds; replay is incomplete.'}
    samples[-1, 7] += .06
    reverse_completed_path(context, q, h, t, samples)
    samples[-1, 7] += .03
    with pytest.raises(ValueError): reverse_completed_path(context, q, h, t, samples)


@pytest.mark.parametrize('part,index,change', [('arm', 0, .031), ('hand', 0, .081)])
def test_return_cannot_be_rebased_to_another_start(part, index, change):
    report = {'return_start': {'arm_q': [0.] * 6, 'hand_q': [0.] * 20}}
    q, h = np.zeros(6), np.zeros(20)
    check_return_start(report, q, h)
    (q if part == 'arm' else h)[index] += change
    with pytest.raises(SafetyStop, match='recorded return start'):
        check_return_start(report, q, h)
