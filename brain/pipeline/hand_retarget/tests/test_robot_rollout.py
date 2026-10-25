import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import Mock
import numpy as np
import pytest
import zmq
from hand_retarget import robot_rollout as rollout


@pytest.fixture
def source(tmp_path,monkeypatch):
    model=rollout.RobotSurfaceModel()
    q=np.array([-1.57,-1.2,1.8,-.6,1.571,-1.571]);h=np.zeros(20)
    frames=model.transforms(q,h)
    folder=tmp_path/'source';folder.mkdir()
    capture={'robot_state':{'arm_q_rad':q.tolist(),'hand_q_rad':h.tolist()},
             'camera_from_robot_base':np.eye(4).tolist(),
             'camera_from_hand_mount':frames['rl_dg_mount'].tolist(),'calibration':{'test':'pinned'}}
    (folder/'capture.json').write_text(json.dumps(capture))
    np.savez(folder/'retargeted_hand.npz',palm_pose_robot_base=np.array([frames['rl_dg_palm']]*2),
             hand_q_rad=np.array([h,h]),time_s=[0.,.05],joint_names=rollout.HAND_JOINT_NAMES)
    monkeypatch.setattr(rollout,'scene_geometry',lambda _: {'table_plane':{'normal':[0,0,1],'offset_m':2.}})
    return folder


def test_prepare_solves_arm_preserves_capture_and_validates_table(source,tmp_path):
    output=tmp_path/'result';report=rollout.prepare(source,output)
    assert report['execution_ready'] and not report['hardware_executed']
    assert report['maximum_ik_position_error_m']<1e-6
    assert (output/'capture.json').read_bytes()==(source/'capture.json').read_bytes()
    with np.load(output/'robot_rollout.npz') as data:
        rollout.check_path(data['arm_q'],data['hand_q'],data['timestamp'])


def test_unsafe_surface_refuses_export(source,tmp_path,monkeypatch):
    monkeypatch.setattr(rollout,'surface_clearance',lambda *a: .005)
    output=tmp_path/'result'
    with pytest.raises(ValueError,match='Table clearance'):rollout.prepare(source,output)
    assert not (output/'robot_rollout.npz').exists()
    assert not json.loads((output/'manifest.json').read_text())['execution_ready']


def test_native_preparation_exports_bounded_filtered_source(source,tmp_path):
    with np.load(source/'retargeted_hand.npz') as data:
        values={key:data[key].copy() for key in data.files}
    values['palm_pose_robot_base']=np.repeat(values['palm_pose_robot_base'][:1],5,axis=0)
    values['hand_q_rad']=np.zeros((5,20));values['hand_q_rad'][:,7]=[0.,1.,0.,0.,0.]
    values['time_s']=np.arange(5.)
    np.savez(source/'retargeted_hand.npz',**values)
    output=tmp_path/'native'
    report=rollout.prepare(source,output,speed_scale=1.,timing_policy='native')
    assert report['execution_ready']
    with np.load(output/'robot_rollout.npz') as data:
        rollout.check_path(data['arm_q'],data['hand_q'],data['timestamp'],None)


@pytest.mark.parametrize('fault',['speed','nan','hand_limit'])
def test_invalid_targets(fault):
    q=np.zeros((2,6));h=np.zeros((2,20));t=np.array([0.,1.])
    if fault=='speed':q[1,0]=.2
    if fault=='nan':q[1,0]=np.nan
    if fault=='hand_limit':h[1,0]=4
    with pytest.raises(ValueError):rollout.check_path(q,h,t)


@pytest.mark.parametrize('arm_only',[False,True])
@pytest.mark.parametrize('timing_policy,ending_residual',[('legacy',0.),('native',0.),('native',.06),('fixed',0.)])
@pytest.mark.parametrize('send,confirmation,watchdog,commands',[(False,'SEND',.2,False),(True,'cancel',.2,False),(True,'SEND',.2,True),(True,'SEND',0,False)])
def test_replay_only_mock_endpoints_requires_confirmation_and_watchdog(source,tmp_path,monkeypatch,send,confirmation,watchdog,commands,arm_only,timing_policy,ending_residual):
    folder=tmp_path/'result';rollout.prepare(source,folder,timing_policy='legacy' if timing_policy=='fixed' else timing_policy)
    with np.load(folder/'robot_rollout.npz') as d:values={k:d[k].copy() for k in d.files}
    if timing_policy=='fixed':
        values['interpolation']=np.array('fixed_pose_quintic_v1')
        values['start_policy']=np.array('captured_absolute')
    values['hand_q'][-1,0]+=ending_residual
    values['timestamp']=np.linspace(0,.08,len(values['arm_q']))
    np.savez(folder/'robot_rollout.npz',**values)
    report=json.loads((folder/'manifest.json').read_text());report['rollout_sha256']=rollout.digest(folder/'robot_rollout.npz')
    (folder/'manifest.json').write_text(json.dumps(report))
    sent=[]
    live_q=values['arm_q'][0].copy();live_q[0]+=.001
    live_h=values['hand_q'][0].copy();live_h[0]+=.01
    class Arm:
        def __init__(self,*args):
            self.q=None;self.q_at=None;self.command=SimpleNamespace(send_json=lambda x:sent.append(('arm',x)))
            self.state=SimpleNamespace(recv_json=self.recv);self.once=False
        def recv(self,**kw):
            if self.once:raise zmq.Again()
            self.once=True
            return {'timestamp_ms':time.time()*1000,'Q':live_q.tolist(),'Qd':[0.]*6,'joint_target_watchdog_s':watchdog}
        def wait(self):return self.fresh()
        def fresh(self):self.poll();return self.q.copy()
        def stop(self):sent.append(('stop',None))
        def close(self):pass
    class Hand:
        def __init__(self,*args):
            self.q=None;self.q_at=None;self.state=SimpleNamespace(recvfrom=self.recv);self.once=False
        def recv(self,*args):
            if self.once:raise BlockingIOError()
            self.once=True
            return json.dumps({'type':'hand_state','measured_age_s':0,'positions':live_h.tolist()}).encode(),None
        def wait(self):return self.fresh()
        def fresh(self):self.poll();return self.q.copy()
        def send(self,q):sent.append(('hand',q))
        def close(self):pass
    monkeypatch.setattr(rollout,'transport_module',lambda:SimpleNamespace(Arm=Arm,Hand=Hand))
    monkeypatch.setattr('builtins.input',lambda _:confirmation)
    monkeypatch.setattr(rollout.signal,'signal',lambda *args:None)
    if timing_policy=='fixed' and arm_only:
        with pytest.raises(ValueError,match='arm-only'):rollout.replay(folder,send,arm_only=True)
        assert not sent
        return
    if watchdog==0:
        with pytest.raises(RuntimeError,match='watchdog'):rollout.replay(folder,send,arm_only=arm_only)
    else:rollout.replay(folder,send,arm_only=arm_only)
    assert bool(sent)==commands
    if commands:
        assert any(kind=='stop' for kind,_ in sent)
        np.testing.assert_array_equal(next(v['target_q'] for k,v in sent if k=='arm'),live_q)
        if arm_only:
            assert not any(k=='hand' for k,_ in sent)
        else:
            np.testing.assert_array_equal(next(v for k,v in sent if k=='hand'),live_h)
        if timing_policy in ('native','fixed'):
            context=json.loads(next(folder.glob('rollout_log_*.json')).read_text())
            residual_expected=bool(ending_residual and not arm_only)
            assert context['outcome']==('trajectory_finished_with_residual_error' if residual_expected else 'trajectory_finished')
            assert context['completion']['targets_delivered']
            assert context['completion']['within_settle_tolerance'] is not residual_expected
            if residual_expected:
                assert context['completion']['hand_residual_rad']==pytest.approx(ending_residual)


def test_runner_lease_stops_on_parent_exit_and_stale_browser():
    import os
    read,write=os.pipe()
    try:
        lease=rollout.BrowserLease(read)
        os.write(write,b'HEARTBEAT\n');lease.check()
        lease.last-=4
        with pytest.raises(RuntimeError,match='heartbeat expired'):lease.check()
        os.write(write,b'HEARTBEAT\n');lease.check()
        os.close(write);write=None
        with pytest.raises(RuntimeError,match='server disconnected'):lease.check()
    finally:
        os.close(read)
        if write is not None:os.close(write)


def test_velocity_stop_preserves_unsent_candidate_and_last_command(source,tmp_path,monkeypatch):
    folder=tmp_path/'result';rollout.prepare(source,folder)
    with np.load(folder/'robot_rollout.npz') as data:
        q=data['arm_q'][0].copy();h=data['hand_q'][0].copy()
    sends=[]
    class Arm:
        def __init__(self,*args):
            self.q=q.copy();self.qd=np.zeros(6)
            self.command=SimpleNamespace(send_json=lambda item:sends.append(('arm',item)))
            self.state=SimpleNamespace(recv_json=self.recv);self.once=False
        def recv(self,**kwargs):
            if self.once:raise zmq.Again()
            self.once=True
            return {'timestamp_ms':time.time()*1000,'Q':q.tolist(),'Qd':[0.]*6,'joint_target_watchdog_s':.2}
        def wait(self):return self.fresh()
        def fresh(self):self.poll();return self.q.copy()
        def stop(self):sends.append(('stop',None))
        def close(self):pass
    class Hand:
        def __init__(self,*args):
            self.q=h.copy();self.state=SimpleNamespace(recvfrom=self.recv);self.delivered=-1
        def recv(self,*args):
            phase=int(any(kind=='arm' for kind,_ in sends))
            if phase==self.delivered:raise BlockingIOError()
            self.delivered=phase
            velocities=np.zeros(20);velocities[11]=.942477796 if phase else 0.
            return json.dumps({'type':'hand_state','measured_age_s':0.,'positions':h.tolist(),
                               'velocities':velocities.tolist(),'currents_ma':[0.]*20}).encode(),None
        def wait(self):return self.fresh()
        def fresh(self):self.poll();return self.q.copy()
        def send(self,target):sends.append(('hand',np.asarray(target).copy()))
        def close(self):pass
    monkeypatch.setattr(rollout,'transport_module',lambda:SimpleNamespace(Arm=Arm,Hand=Hand))
    monkeypatch.setattr('builtins.input',lambda _:'SEND')
    monkeypatch.setattr(rollout.signal,'signal',lambda *args:None)
    with pytest.raises(rollout.SafetyStop,match='rj_dg_3_4'):
        rollout.replay(folder,send=True)
    context=json.loads(next(folder.glob('rollout_log_*.json')).read_text())
    assert context['outcome']=='stopped_with_error'
    assert context['failure']['layer']=='application_safety_check'
    assert context['failure']['hand_state']['velocities'][11]==.942477796
    assert context['loop']['phase']=='feedback_check'
    assert context['sent_counts']=={'arm':1,'hand':1}
    assert context['arm_stop_confirmed']
    assert sum(kind=='arm' for kind,_ in sends)==1
    # One trajectory hand target and one measured hold after stop, never a next target.
    assert [kind for kind,_ in sends]==['arm','hand','stop','hand']
    np.testing.assert_array_equal(context['last_sent_targets']['hand'],h)
    with np.load(next(folder.glob('rollout_log_*.npz'))) as log:
        assert len(log['samples'])==1
        assert log['hand_feedback'][-1,21+11]==.942477796


def test_rollout_preserves_retargeted_fingers_and_ignores_simulated_motion(source,tmp_path):
    with np.load(source/'retargeted_hand.npz') as data:original={k:data[k].copy() for k in data.files}
    original['hand_q_rad'][1,0]=.01
    np.savez(source/'retargeted_hand.npz',**original)
    np.savez(source/'mujoco_motion.npz',hand_q_rad=np.full((2,20),999.))
    output=tmp_path/'out';report=rollout.prepare(source,output)
    with np.load(output/'robot_rollout.npz') as data:
        # The first row is the measured capture pose for the slow initial transition.
        np.testing.assert_array_equal(data['hand_q'][1:],original['hand_q_rad'])
    assert not report['mujoco_motion_used']
    assert set(report['source_sha256'])=={'capture.json','retargeted_hand.npz'}


def test_api_rejects_mujoco_source_and_copies_only_retargeted_input(tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    from hand_retarget.webapp import Config,create_app
    app=create_app(Config(tmp_path/'runs'));jobs=app.state.jobs
    queued=[];monkeypatch.setattr(jobs.executor,'submit',lambda *args:queued.append(args))
    for identifier,mode in [('a'*32,'generated_hand_retarget'),('b'*32,'mujoco_replay')]:
        folder=tmp_path/'runs'/identifier;result=folder/'result';result.mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps({'status':'completed','mode':mode}))
        (result/'capture.json').write_text('{}')
        (result/'retargeted_hand.npz').write_bytes(b'retargeted-input')
        (result/'mujoco_motion.npz').write_bytes(b'never-use-this')
    with TestClient(app) as client:
        headers={'X-Retarget-Client':'1'}
        rejected=client.post('/api/rollout/prepare',headers=headers,json={'source_job_id':'b'*32})
        assert rejected.status_code==400 and not queued
        accepted=client.post('/api/rollout/prepare',headers=headers,json={'source_job_id':'a'*32})
        assert accepted.status_code==202
        inputs=tmp_path/'runs'/accepted.json()['id']/'inputs'
        assert {p.name for p in inputs.iterdir()}=={'capture.json','retargeted_hand.npz'}
        assert (inputs/'retargeted_hand.npz').read_bytes()==b'retargeted-input'


def test_prepare_speed_changes_timing_only_including_initial_transition(source,tmp_path):
    baseline=tmp_path/'baseline';slow=tmp_path/'slow'
    a=rollout.prepare(source,baseline,speed_scale=1.)
    b=rollout.prepare(source,slow,speed_scale=.5)
    with np.load(baseline/'robot_rollout.npz') as x,np.load(slow/'robot_rollout.npz') as y:
        np.testing.assert_array_equal(x['arm_q'],y['arm_q'])
        np.testing.assert_array_equal(x['hand_q'],y['hand_q'])
        np.testing.assert_allclose(y['timestamp'],2*x['timestamp'])
    assert b['duration_s']==2*a['duration_s']
    assert b['initial_transition_duration_s']==2*a['initial_transition_duration_s']
    assert b['hand_speed_limit_rad_s']==.175


@pytest.mark.parametrize('speed',[0.,-1.,1.1,float('nan')])
def test_invalid_rollout_speed_rejected(source,tmp_path,speed):
    with pytest.raises(ValueError,match='Replay speed'):rollout.prepare(source,tmp_path/'result',speed_scale=speed)
    assert not (tmp_path/'result').exists()


def test_filter_reduces_jitter_preserves_endpoints_and_range():
    values=np.tile((.3+.06*(-1.)**np.arange(41))[:,None],(1,20))
    filtered=rollout.smooth_fingers(values)
    np.testing.assert_array_equal(filtered[[0,-1]],values[[0,-1]])
    assert np.max(np.abs(np.diff(filtered[5:-5],axis=0)))<.01
    assert filtered.min()>=values.min() and filtered.max()<=values.max()


def test_curve_is_continuous_bounded_and_starts_stops_at_rest():
    times=np.array([0.,.7,2.,2.5,4.])
    values=np.array([[0.,1.],[.3,.4],[.2,.5],[.6,.5],[.1,.2]])
    curve=rollout.smooth_curve(times,values)
    np.testing.assert_allclose(curve(times),values)
    np.testing.assert_allclose(curve.derivative()(times[[0,-1]]),0,atol=1e-12)
    for i in range(len(times)-1):
        sample=curve(np.linspace(times[i],times[i+1],101))
        assert np.all(sample>=np.minimum(values[i],values[i+1])-1e-12)
        assert np.all(sample<=np.maximum(values[i],values[i+1])+1e-12)
    v,a=rollout.curve_limits(curve)
    grid=np.linspace(0,4,10001)
    assert np.max(np.abs(curve.derivative()(grid)))<=v+1e-10
    assert np.max(np.abs(curve.derivative(2)(grid)))<=a+1e-10
    for t in times[1:-1]:
        np.testing.assert_allclose(curve.derivative()(t-1e-8),curve.derivative()(t+1e-8),atol=1e-6)


def test_live_lead_in_is_checked_and_speed_scaled(monkeypatch):
    samples=[]
    monkeypatch.setattr(rollout,'surface_clearance',lambda model,q,h,*args: samples.append((q.copy(),h.copy())) or .1)
    q=np.zeros(6);h=np.zeros(20);target=h+.1
    duration=rollout.checked_lead_in(None,q,h,q,target,[0,0,1],0,.25)
    assert duration>=2 and samples
    np.testing.assert_array_equal(samples[0][1],h)
    np.testing.assert_array_equal(samples[-1][1],target)
    assert 1.875*.1/duration<=.175*.25
    assert 6*.1/duration**2<=.35*.25**2
    monkeypatch.setattr(rollout,'surface_clearance',lambda *args:.005)
    with pytest.raises(RuntimeError,match='clearance'):
        rollout.checked_lead_in(None,q,h,q,target,[0,0,1],0,.25)
    with pytest.raises(RuntimeError,match='neighbourhood'):
        rollout.checked_lead_in(None,q,h,q,h+.21,[0,0,1],0,.25)


def test_preparation_saves_filtered_path_without_changing_source(source,tmp_path):
    with np.load(source/'retargeted_hand.npz') as data:original={k:data[k].copy() for k in data.files}
    original['palm_pose_robot_base']=np.repeat(original['palm_pose_robot_base'][:1],21,axis=0)
    original['time_s']=np.arange(21)*.05
    original['hand_q_rad']=np.zeros((21,20))
    original['hand_q_rad'][:,0]=.1+.03*(-1.)**np.arange(21)
    np.savez(source/'retargeted_hand.npz',**original)
    before=rollout.digest(source/'retargeted_hand.npz')
    output=tmp_path/'filtered';report=rollout.prepare(source,output,speed_scale=.25)
    assert rollout.digest(source/'retargeted_hand.npz')==before
    assert report['maximum_finger_smoothing_change_rad']>0
    with np.load(output/'robot_rollout.npz') as data:
        np.testing.assert_allclose(data['hand_q'][1:],rollout.smooth_fingers(original['hand_q_rad']))
        assert str(data['interpolation'])=='flow_quintic_v3'
        rollout.check_continuous(data['arm_q'],data['hand_q'],data['timestamp'],.25)
        assert np.max(rollout.derivative_bounds(rollout.continuous_curve(data['timestamp'],data['hand_q']))[0])<=.35*.25+1e-6


def test_actual_finger_overspeed_remains_a_stop():
    rollout.check_finger_velocity(np.full(20,.7))
    velocity=np.zeros(20);velocity[12]=-1.152
    with pytest.raises(rollout.SafetyStop,match='rj_dg_4_1 measured -1.152'):
        rollout.check_finger_velocity(velocity)
    with pytest.raises(rollout.SafetyStop,match='missing or invalid'):
        rollout.check_finger_velocity(np.full(20,np.nan))


def test_cli_reports_safety_stop_without_python_traceback(monkeypatch,capsys):
    monkeypatch.setattr('sys.argv',['robot_rollout','replay','--folder','unused'])
    def stop(*args):raise rollout.SafetyStop('finger overspeed')
    monkeypatch.setattr(rollout,'replay',stop)
    with pytest.raises(SystemExit) as error:rollout.main()
    assert error.value.code==2
    assert capsys.readouterr().err=='SAFETY STOP: finger overspeed\n'


def test_action_postprocessing_is_bounded_c2_and_synchronized():
    q=np.zeros((5,6));h=np.zeros((5,20))
    q[:,0]=[0,.01,-.02,.02,0];h[:,0]=[0,.1,.03,.04,.02]
    t=rollout.retime_actions(q,h,np.arange(5)*.02,.5)
    curves=rollout.check_actions(q,h,t,.5)
    for curve,values,vmax,amax,jmax in zip(curves,(q,h),(.075,.175),(.075,.175),(.075,.175)):
        np.testing.assert_allclose(curve(t),values,atol=1e-12)
        np.testing.assert_allclose(curve.derivative()(t),0,atol=1e-12)
        np.testing.assert_allclose(curve.derivative(2)(t),0,atol=1e-12)
        for i in range(len(t)-1):
            grid=np.linspace(t[i],t[i+1],401)
            sample=curve(grid)
            assert np.all(sample>=np.minimum(values[i],values[i+1])-1e-12)
            assert np.all(sample<=np.maximum(values[i],values[i+1])+1e-12)
            for order,limit in enumerate((vmax,amax,jmax),1):
                assert np.max(np.abs(curve.derivative(order)(grid)))<=limit+1e-9
    with pytest.raises(ValueError,match='speed|acceleration|jerk'):
        rollout.check_actions(q,h,t*.1,.5)


def test_rebase_anchors_to_live_pose_preserves_relative_actions_and_timing():
    q=np.zeros((4,6));h=np.zeros((4,20));t=np.array([0.,3.,6.,9.])
    q[:,0]=[.03,.05,.06,.04];h[:,0]=[.2,.25,.3,.28]
    live_q=np.zeros(6);live_q[0]=.02
    live_h=np.zeros(20);live_h[0]=.1
    a,b,at=rollout.rebase_actions(q,h,t,live_q,live_h)
    np.testing.assert_array_equal(a[0],live_q)
    np.testing.assert_array_equal(b[0],live_h)
    np.testing.assert_allclose(np.diff(a,axis=0),np.diff(q[1:],axis=0))
    np.testing.assert_allclose(np.diff(b,axis=0),np.diff(h[1:],axis=0))
    np.testing.assert_array_equal(at,[0.,3.,6.])
    np.testing.assert_allclose(rollout.action_limits(a,at),rollout.action_limits(q[1:],at))
    bad_live=live_h.copy();bad_live[0]=rollout.HAND_UPPER[0]
    _,bounded,_=rollout.rebase_actions(q,h,t,live_q,bad_live)
    np.testing.assert_array_equal(bounded[0],bad_live)
    assert np.all(bounded<=rollout.HAND_UPPER+1e-9)
    bad_live[0]+=.01
    with pytest.raises(rollout.SafetyStop,match='starting joint'):
        rollout.rebase_actions(q,h,t,live_q,bad_live)


def test_shifted_path_rechecks_intermediate_table_clearance(monkeypatch):
    q=np.zeros((2,6));q[1,0]=.01;h=np.zeros((2,20));t=np.array([0.,3.])
    visited=[]
    def clearance(model,a,*args):
        visited.append(a[0])
        return .005 if .004<a[0]<.006 else .1
    monkeypatch.setattr(rollout,'surface_clearance',clearance)
    with pytest.raises(rollout.SafetyStop,match='Shifted rollout'):
        rollout.validate_rebased_actions(None,q,h,t,[0,0,1],0,1.)
    assert any(.004<x<.006 for x in visited)


def test_small_negative_finger_action_is_projected_without_start_jump():
    q=np.zeros((4,6));h=np.zeros((4,20));t=np.array([0.,2.,4.,6.])
    h[:,19]=[.1,.1,.0972,.12]
    live=np.zeros(20)
    a,b,at=rollout.rebase_actions(q,h,t,np.zeros(6),live)
    np.testing.assert_array_equal(b[0],live)
    np.testing.assert_allclose(b[:,19],[0,0,.02])
    rollout.check_actions(a,b,at,1.)
    curve=rollout.action_curve(at,b)
    assert np.min(curve(np.linspace(0,at[-1],1001)))>=-1e-12


@pytest.mark.parametrize('speed',[0.,-1.,1.01,float('nan')])
def test_invalid_execution_speed_fails_before_hardware(tmp_path,speed):
    with pytest.raises(ValueError,match='Replay speed'):
        rollout.replay(tmp_path,speed=speed)


def test_arm_only_uses_live_fingers_for_entire_collision_path():
    q=np.zeros((4,6));h=np.zeros((4,20));h[2:,0]=.2
    live=np.zeros(20);live[0]=.05
    _,shifted,_=rollout.rebase_actions(q,h,np.arange(4)*2.,np.zeros(6),live,arm_only=True)
    np.testing.assert_array_equal(shifted,np.repeat(live[None,:],3,axis=0))


def test_prepare_uses_existing_arm_retarget_without_solving_again(source,tmp_path,monkeypatch):
    with np.load(source/'retargeted_hand.npz') as data:values={k:data[k].copy() for k in data.files}
    capture=json.loads((source/'capture.json').read_text())
    q=np.asarray(capture['robot_state']['arm_q_rad'])
    values['arm_q_rad']=np.repeat(q[None,:],2,axis=0)
    np.savez(source/'retargeted_hand.npz',**values)
    monkeypatch.setattr(rollout,'least_squares',lambda *a,**kw:pytest.fail('Do not redo upstream arm IK'))
    result=tmp_path/'out';report=rollout.prepare(source,result)
    assert report['arm_motion_source']=='Step 04 upstream ArmStage'
    with np.load(result/'robot_rollout.npz') as data:
        np.testing.assert_array_equal(data['arm_q'][1:],values['arm_q_rad'])


def test_cartesian_anchor_preserves_translation_and_relative_rotations():
    model=rollout.RobotSurfaceModel()
    first=np.array([-1.57,-1.2,1.8,-.6,1.571,-1.571]);h=np.zeros((4,20))
    q=np.repeat(first[None,:],4,axis=0);q[2,0]+=.01;q[3,1]+=.01
    live=first+np.array([.03,-.02,.02,0,0,.01])
    current=model.transforms(live,h[0])['rl_dg_palm']
    source=np.array([model.transforms(a,b)['rl_dg_palm'] for a,b in zip(q[1:],h[1:])])
    shifted,fingers,times=rollout.rebase_cartesian_actions(model,q,h,np.array([0.,2.,4.,6.]),live,h[0],.5,arm_only=True)
    np.testing.assert_array_equal(shifted[0],live)
    actual=np.array([model.transforms(a,b)['rl_dg_palm'] for a,b in zip(shifted,fingers)])
    np.testing.assert_allclose(actual[:,:3,3]-current[:3,3],source[:,:3,3]-source[0,:3,3],atol=1e-5)
    np.testing.assert_allclose(actual[:,:3,:3],source[:,:3,:3]@source[0,:3,:3].T@current[:3,:3],atol=1e-5)
    rollout.check_actions(shifted,fingers,times,.5)
