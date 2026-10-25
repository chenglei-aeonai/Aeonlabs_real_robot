import json
from pathlib import Path
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hand_retarget.generated_retarget import calibrated_input,retarget
from hand_retarget.placement import apply
from hand_retarget.skeleton import KEYPOINT_NAMES,FINGER_CHAINS
from hand_retarget.webapp import Config,create_app


@pytest.fixture
def source(tmp_path):
    folder=tmp_path/'input';folder.mkdir()
    joints=np.zeros((21,3));joints[:,2]=1.
    for chain,base in zip(FINGER_CHAINS,[[-.05,-.02],[-.03,-.06],[0,-.07],[.02,-.06],[.04,-.05]]):
        for step,index in enumerate(chain):joints[index,:2]=np.array(base)+[0,-.025*step]
    transform=np.array([[0.,-1,0,.2],[1,0,0,-.1],[0,0,1,.4],[0,0,0,1]])
    camera={'width':832,'height':640,'undistorted_intrinsics':{'matrix':[[500,0,416],[0,500,320],[0,0,1]],'dist_coeffs':[0]*5}}
    capture={'handedness':'right','position_units':'metres','camera':camera,
             'camera_from_robot_base':transform.tolist(),'camera_from_hand_mount':np.eye(4).tolist(),
             'mano':{'coordinate_frame':'RealSense color optical','joints':[{'index':i,'name':n,'position_camera_m':p.tolist()} for i,(n,p) in enumerate(zip(KEYPOINT_NAMES,joints))]}}
    (folder/'capture.json').write_text(json.dumps(capture))
    generated=np.zeros((3,2,21,3));generated[:,1]=joints;generated[1:,1,:,0]+=.01
    np.savez(folder/'future_hands.npz',joints=generated,K=camera['undistorted_intrinsics']['matrix'],fps=24.,valid=np.array([False,True]))
    return folder


def test_calibration_inverse_not_forward_and_positions_unchanged(source):
    capture,camera,base,fps,transform,error=calibrated_input(source)
    expected=np.linalg.inv(capture['camera_from_robot_base'])
    np.testing.assert_allclose(transform,expected)
    np.testing.assert_allclose(base,apply(expected,camera))
    assert error<1e-12 and fps==24
    np.testing.assert_allclose(base[1]-base[0],np.tile([0,-.01,0],(21,1)),atol=1e-12)


@pytest.mark.parametrize('fault',['reflection','wrong_first_frame','left','wrong_camera','degenerate_palm'])
def test_invalid_capture_or_trajectory_refused(source,fault):
    capture=json.loads((source/'capture.json').read_text())
    with np.load(source/'future_hands.npz') as file:data={k:file[k].copy() for k in file.files}
    if fault=='reflection':capture['camera_from_robot_base'][0][1]=1.
    if fault=='wrong_first_frame':data['joints'][0,1,5,0]+=.1
    if fault=='left':data['valid']=np.array([True,False])
    if fault=='degenerate_palm':data['joints'][1,1]=data['joints'][1,1,0]
    if fault=='wrong_camera':data['K'][0,0]+=50
    (source/'capture.json').write_text(json.dumps(capture));np.savez(source/'future_hands.npz',**data)
    with pytest.raises(ValueError):calibrated_input(source)


def test_real_urdf_retarget_is_finite_bounded_and_offline(source,tmp_path):
    from hand_retarget.robot import HAND_LOWER,HAND_UPPER
    report=retarget(source,tmp_path/'out')
    assert report['status']=='completed' and not report['hardware_executed'] and not report['execution_ready']
    with np.load(tmp_path/'out/retargeted_hand.npz') as data:
        q=data['hand_q_rad'];assert q.shape==(3,20) and np.isfinite(q).all()
        assert np.all(q>=HAND_LOWER-1e-8) and np.all(q<=HAND_UPPER+1e-8)
        assert data['robot_points_base_m'].shape==(3,26,3)
        np.testing.assert_allclose(data['mano_joints_robot_base_m'],calibrated_input(source)[2])
    preview=json.loads((tmp_path/'out/retarget_preview.json').read_text())
    assert preview['coordinate_frame']=='robot_base' and len(preview['frames'])==3


def test_api_uses_generation_source_capture_not_latest_and_no_gpu(source,tmp_path,monkeypatch):
    app=create_app(Config(tmp_path/'runs'));jobs=app.state.jobs
    edit_id=uuid.uuid4().hex;future_id=uuid.uuid4().hex
    edit=tmp_path/'runs'/edit_id;future=tmp_path/'runs'/future_id
    for folder,mode in ((edit,'capture_to_human'),(future,'diffusion_video')):
        (folder/'result').mkdir(parents=True)
        (folder/'status.json').write_text(json.dumps({'id':folder.name,'status':'completed','mode':mode,'source_job_id':edit_id}))
    (edit/'result/edited.png').write_bytes(b'test')
    (edit/'result/capture.json').write_bytes((source/'capture.json').read_bytes())
    (future/'result/future_hands.npz').write_bytes((source/'future_hands.npz').read_bytes())
    queued=[];monkeypatch.setattr(jobs.executor,'submit',lambda *args:queued.append(args))
    monkeypatch.setattr(jobs.diffusion,'status',lambda:pytest.fail('Retarget must not contact GPU service'))
    with TestClient(app) as client:
        assert client.post('/api/retarget/generate',json={'source_job_id':future_id}).status_code==403
        response=client.post('/api/retarget/generate',headers={'X-Retarget-Client':'1'},json={'source_job_id':future_id})
        assert response.status_code==202,response.text
        job=tmp_path/'runs'/response.json()['id']
        assert (job/'inputs/capture.json').read_bytes()==(source/'capture.json').read_bytes()
        assert 'hand_retarget.generated_retarget' in queued[0][-1]
        assert response.json()['hardware_executed'] is False
        assert client.get('/retarget.js').status_code==200


def test_captured_arm_uses_upstream_stage_and_exports_replay_format(source,tmp_path):
    from hand_retarget.robot_rollout import RobotSurfaceModel,transport_module
    capture=json.loads((source/'capture.json').read_text())
    q=np.array([-1.57,-1.2,1.8,-.6,1.571,-1.571]);h=np.zeros(20)
    model=RobotSurfaceModel();mount=model.transforms(q,h)['rl_dg_mount']
    capture['robot_state']={'arm_q_rad':q.tolist(),'hand_q_rad':h.tolist()}
    capture['camera_from_hand_mount']=(np.asarray(capture['camera_from_robot_base'])@mount).tolist()
    (source/'capture.json').write_text(json.dumps(capture))
    output=tmp_path/'arm_out';report=retarget(source,output)
    assert report['arm_ik_computed'] and not report['execution_ready']
    assert report['replay_export_rate_hz']==60
    with np.load(output/'retargeted_hand.npz') as data:
        np.testing.assert_allclose(data['arm_q_rad'][0],q,atol=1e-8)
        np.testing.assert_allclose(data['mano_joints_robot_base_m'],calibrated_input(source)[2])
        for arm,hand,pose in zip(data['arm_q_rad'],data['hand_q_rad'],data['palm_pose_robot_base']):
            np.testing.assert_allclose(model.transforms(arm,hand)['rl_dg_palm'],pose,atol=1e-6)
    exported=transport_module().Demonstration(output/'robot_demonstration.npz',.25)
    assert len(exported.arm)==6
    with np.load(output/'robot_demonstration.npz') as data:
        assert 'table_top_z' not in data and not data['execution_validated']
        np.testing.assert_allclose(np.diff(data['timestamp']),1/60)
