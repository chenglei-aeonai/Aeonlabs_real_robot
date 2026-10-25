import numpy as np
import pytest

from hand_retarget.fixed_pose_return import fixed_waypoints
from hand_retarget import robot_rollout as rollout


def test_fixed_return_keeps_goal_when_start_changes():
    goal = np.zeros(6)
    hand_goal = (rollout.HAND_LOWER + rollout.HAND_UPPER) / 2
    for start in (np.full(6, .1), np.full(6, -.2)):
        q, h, t = fixed_waypoints(start, hand_goal - .05, goal, hand_goal)
        np.testing.assert_array_equal(q[0], start)
        np.testing.assert_array_equal(q[-1], goal)
        np.testing.assert_array_equal(h[-1], hand_goal)
        assert t[-1] - t[-2] >= 2
        curves = rollout.check_actions(q, h, t, 1.)
        for c, end in zip(curves, (goal, hand_goal)):
            np.testing.assert_allclose(c(t[-1]), end, atol=1e-12)
            np.testing.assert_allclose(c(t[-1], 1), 0, atol=1e-12)


def test_fixed_return_rejects_unreachable_joint_range():
    with pytest.raises(ValueError, match='joint limits'):
        fixed_waypoints(np.zeros(6), np.zeros(20), np.full(6, 20.), np.zeros(20))
