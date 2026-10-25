import numpy as np
import pytest
from hand_retarget.partial_return import prefix_waypoints


def test_recovery_excludes_every_unsent_future_waypoint():
    q=np.repeat(np.arange(5.)[:,None],6,axis=1)
    h=np.repeat(np.arange(5.)[:,None],20,axis=1)
    last=np.r_[1.5,np.zeros(26),np.full(6,1.5),np.full(20,1.5)]
    a,b,t=prefix_waypoints(q,h,np.arange(5.),last,np.full(6,1.4),np.full(20,1.4))
    np.testing.assert_allclose(a[:,0],[1.4,1.5,1.,0.])
    np.testing.assert_allclose(b[:,0],a[:,0])
    assert np.all(np.diff(t)>0)
    assert np.max(a)<=1.5


@pytest.mark.parametrize('when',[-1,0,5])
def test_recovery_rejects_invalid_prefix_time(when):
    with pytest.raises(ValueError):
        prefix_waypoints(np.zeros((5,6)),np.zeros((5,20)),np.arange(5.),
                         np.r_[when,np.zeros(52)],np.zeros(6),np.zeros(20))
