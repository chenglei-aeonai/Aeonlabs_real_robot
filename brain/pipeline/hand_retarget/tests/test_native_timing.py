import numpy as np
import pytest

from hand_retarget import native_timing as native
from hand_retarget import robot_rollout as rollout


def test_source_clock_is_preserved_and_only_uniformly_halved():
    q=np.zeros((3,6));q[:,0]=[0.,.5,1.]
    h=np.zeros((3,20));source=np.array([10.,10.25,10.5])
    t,report=native.select_timing(q,h,source)
    assert [c['speed_scale'] for c in report['candidates']]==[1.,.5,.25]
    assert [c['passed'] for c in report['candidates']]==[False,False,True]
    assert report['source_duration_s']==.5
    assert report['duration_s']==2.
    np.testing.assert_allclose(t,(source-source[0])/.25)
    np.testing.assert_array_equal(q[:,0],[0.,.5,1.])


def test_never_silently_turn_fast_action_into_minute_long_replay():
    q=np.zeros((3,6));q[:,0]=[0.,1.,2.]
    with pytest.raises(native.NativeTimingError) as failure:
        native.select_timing(q,np.zeros((3,20)),np.array([0.,.05,.1]))
    assert failure.value.report['candidates'][-1]['speed_scale']==.25
    assert not failure.value.report['candidates'][-1]['passed']


def test_exact_cubic_bounds_cover_dense_samples_and_detect_joint_overshoot():
    t=np.array([0.,.7,1.5,3.,4.]);v=np.array([[0.,1.],[.3,.4],[.2,.5],[.6,.5],[.1,.2]])
    c=native.native_curve(t,v);bounds=native.peaks(c);grid=np.linspace(0,4,30001)
    for order,key in [(1,'speed'),(2,'acceleration'),(3,'jerk')]:
        sampled=np.max(np.abs(c(grid,order)),axis=0)
        assert np.all(sampled<=bounds[key]+1e-9)
        np.testing.assert_allclose(sampled,bounds[key],atol=1e-3)
    assert np.all(c(grid)>=bounds['minimum']-1e-9)
    assert np.all(c(grid)<=bounds['maximum']+1e-9)
    np.testing.assert_allclose(c(t),v)
    np.testing.assert_allclose(c([0,4],1),0,atol=1e-12)
    for knot in t[1:-1]:
        np.testing.assert_allclose(c(knot-1e-9,2),c(knot+1e-9,2),atol=1e-7)
    q=np.zeros((4,6));h=np.zeros((4,20));h[:,0]=[0.,.9,.95,1.]
    with pytest.raises(ValueError,match='joint limits'):
        native.checked_curves(q,h,np.arange(4.),np.ones(6),np.full(20,-1.),np.full(20,.99))


def test_endpoint_filter_does_not_restore_only_one_discontinuous_frame():
    raw=np.linspace(0,1,107)[:,None]
    old=rollout.gaussian_filter1d(raw,2,axis=0,mode='nearest');old[[0,-1]]=raw[[0,-1]]
    new=native.endpoint_filter(raw,2)
    np.testing.assert_array_equal(new[[0,-1]],raw[[0,-1]])
    assert np.max(np.abs(np.diff(new,2,axis=0)))<np.max(np.abs(np.diff(old,2,axis=0)))*.5


def test_finger_interpolation_preserves_plateaus_at_joint_limits():
    times=np.arange(6.)
    values=np.array([[0.,1.],[.5,1.],[1.,.3],[1.,0.],[1.,0.],[.8,.1]])
    curve=native.native_curve(times,values,monotone=True)
    for index in range(len(times)-1):
        sampled=curve(np.linspace(times[index],times[index+1],101))
        assert np.all(sampled>=np.minimum(values[index],values[index+1])-1e-12)
        assert np.all(sampled<=np.maximum(values[index],values[index+1])+1e-12)


def test_streamed_feedforward_matches_legacy_control_law():
    q=np.array([.1,.2,.3,.4,.5,.6]);actual=q-.01;v=np.linspace(-.1,.1,6);gain=3.
    command=native.servo_target(q,v,actual,gain,np.full(6,6.))
    np.testing.assert_allclose(gain*(command-actual),v+gain*(q-actual),atol=1e-14)
    np.testing.assert_array_equal(native.servo_target(q,np.zeros(6),q,gain,np.full(6,6.)),q)
    with pytest.raises(ValueError,match='speed budget'):
        native.servo_target(q,np.full(6,4.),q,gain,np.full(6,6.))


def test_approved_demo_speed_profile_retains_hard_measured_stop():
    v=np.zeros(20);v[11]=.942477796
    with pytest.raises(rollout.SafetyStop):rollout.check_finger_velocity(v)
    rollout.check_finger_velocity(v,np.pi)
    v[11]=np.pi+.01
    with pytest.raises(rollout.SafetyStop):rollout.check_finger_velocity(v,np.pi)
