import numpy as np
import pytest
from hand_retarget.action_timing import continuous_curve,retime_continuous,check_continuous,derivative_bounds,simplify_actions


def test_continuous_curve_passes_waypoints_without_stopping_or_overshooting():
    t=np.array([0.,1.,2.,3.,4.]);q=np.array([[0.,0.],[.1,.2],[.2,.1],[.3,.1],[.4,0.]])
    curve=continuous_curve(t,q)
    np.testing.assert_allclose(curve(t),q,atol=1e-12)
    assert curve.derivative()(2)[0]>0
    np.testing.assert_allclose(curve.derivative()(t[[0,-1]]),0,atol=1e-12)
    np.testing.assert_allclose(curve.derivative(2)(t),0,atol=1e-12)
    for i in range(len(t)-1):
        values=curve(np.linspace(t[i],t[i+1],401))
        assert np.all(values>=np.minimum(q[i],q[i+1])-1e-12)
        assert np.all(values<=np.maximum(q[i],q[i+1])+1e-12)
    for knot in t[1:-1]:
        for order in (1,2):
            np.testing.assert_allclose(curve.derivative(order)(knot-1e-9),curve.derivative(order)(knot+1e-9),atol=1e-7)


def test_retiming_has_bounded_derivatives_and_is_faster_for_smooth_path():
    from hand_retarget.robot_rollout import retime_actions
    q=np.zeros((41,6));h=np.zeros((41,20));q[:,0]=np.linspace(0,.4,41)
    nominal=np.linspace(0,4,41)
    original=retime_actions(q,h,nominal,1.)[-1]
    keep=simplify_actions(q,h,nominal)
    q=q[keep];h=h[keep];nominal=nominal[keep]
    t=retime_continuous(q,h,nominal,1.)
    assert t[-1]<original*.65
    curves=check_continuous(q,h,t,1.)
    for curve,limits in zip(curves,((.15,.3,.6),(.35,.7,1.4))):
        for order,limit in enumerate(limits,1):
            grid=np.linspace(t[0],t[-1],10001)
            assert np.max(abs(curve.derivative(order)(grid)))<=limit+1e-6
    with pytest.raises(ValueError,match='limit'):check_continuous(q,h,t*.1,1.)
    np.testing.assert_allclose(retime_continuous(q,h,nominal,.5),t*2)
