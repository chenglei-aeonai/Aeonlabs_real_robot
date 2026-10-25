import base64
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx
import imageio_ffmpeg
import numpy as np
from PIL import Image
import pytest

from hand_retarget.aeon_client import AeonClient, service_url
from hand_retarget.diffusion_bundle import build_bundle, wrist_rotation, validate_mano_right
from hand_retarget.diffusion import generate
from hand_retarget.skeleton import KEYPOINT_NAMES, FINGER_CHAINS
from hand_retarget.webapp import Config, create_app

PIPELINE=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def no_live_inference_service(monkeypatch):
    monkeypatch.delenv('AEON_WEBUI_URL',raising=False)


@pytest.fixture
def source(tmp_path):
    folder=tmp_path/'source';folder.mkdir()
    points=np.zeros((21,3));points[:,2]=1.
    bases=[[-.05,-.02],[-.03,-.06],[0,-.07],[.02,-.06],[.04,-.05]]
    for chain,base in zip(FINGER_CHAINS,bases):
        for step,index in enumerate(chain):points[index,:2]=np.array(base)+[0,-.025*step]
    matrix=np.array([[500.,0,416],[0,500.,320],[0,0,1.]])
    entries=[]
    for index,(name,point) in enumerate(zip(KEYPOINT_NAMES,points)):
        uv=matrix@point;pixel=(uv[:2]/uv[2]).tolist()
        entries.append({'index':index,'name':name,'position_camera_m':point.tolist(),
                        'pixel_undistorted':pixel,'synthetic':index in (0,9)})
    data={'position_units':'metres','camera':{'width':832,'height':640,
          'undistorted_intrinsics':{'matrix':matrix.tolist(),'dist_coeffs':[0]*5}},
          'mano':{'coordinate_frame':'RealSense color optical','joints':entries},'capture':{'serial_number':'test'}}
    (folder/'capture.json').write_text(json.dumps(data))
    Image.new('RGB',(832,640),(120,130,140)).save(folder/'edited.png')
    return folder


def test_conditioning_keeps_rgb_camera_and_gt_exactly(source):
    output=source/'first_frame_gt.npz'
    report=build_bundle(source/'edited.png',source/'capture.json',output)
    original=json.loads((source/'capture.json').read_text())
    with np.load(output,allow_pickle=False) as data:
        np.testing.assert_array_equal(data['frame'],np.asarray(Image.open(source/'edited.png')))
        np.testing.assert_array_equal(data['joints3d'][1],[j['position_camera_m'] for j in original['mano']['joints']])
        np.testing.assert_array_equal(data['K'],original['camera']['undistorted_intrinsics']['matrix'])
        assert data['valid'].tolist()==[False,True]
        np.testing.assert_allclose(data['wrist_R'][1]@data['wrist_R'][1].T,np.eye(3),atol=1e-12)
        assert np.linalg.det(data['wrist_R'][1])==pytest.approx(1.)
    assert report['pose_reestimated'] is False
    definition=report['hand_3d_definition']
    assert definition['joint_names']==list(KEYPOINT_NAMES)
    assert definition['hand_order']==['left','right']
    assert definition['synthetic_joint_indices']==[0,9]
    assert 'self-consistency' in report['wrist_convention_validation']['metric_scope']
    with pytest.raises(ValueError,match='Degenerate'):
        wrist_rotation(np.zeros((21,3)))


def test_mano_right_wrist_convention_rotation_direction_and_flip_rejection(source):
    data=json.loads((source/'capture.json').read_text())
    joints=np.array([j['position_camera_m'] for j in data['mano']['joints']])
    original=joints.copy();rotation=wrist_rotation(joints)
    check=validate_mano_right(joints,rotation)
    assert check['mano_right_axes_error_deg']<1e-3
    assert check['camera_roundtrip_max_error_m']<1e-12
    np.testing.assert_array_equal(joints,original)
    # A 180-degree wrong-convention rotation must fail rather than pass an SO(3)-only check.
    with pytest.raises(ValueError,match='MANO-right wrist convention failed'):
        validate_mano_right(joints,rotation@np.diag([-1.,1.,-1.]))
    theta=.7;c,s=np.cos(theta),np.sin(theta)
    camera_change=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    moved=joints@camera_change.T+np.array([.2,-.3,.4])
    np.testing.assert_allclose(wrist_rotation(moved),camera_change@rotation,atol=1e-12)
    local=(joints-joints[0])@rotation
    moved_local=(moved-moved[0])@wrist_rotation(moved)
    np.testing.assert_allclose(moved_local,local,atol=1e-12)


def test_client_uses_upstream_profiles_load_and_multitab_contract():
    token=str(uuid.uuid4());calls=[]
    def handle(request):
        calls.append((request.method,request.url.path))
        if request.url.path=='/api/tabs/config':return httpx.Response(200,json={'enabled':True})
        if request.url.path=='/api/tabs':
            assert json.loads(request.content)['token']==token
            return httpx.Response(200,json={'prefix':'/tabs/'+token})
        if request.url.path.endswith('/api/hands/status'):
            return httpx.Response(200,json={'profiles':[{'id':'p'}],'state':'ready','selected_profile':'p'})
        if request.url.path.endswith('/api/robot/gt-capabilities'):
            return httpx.Response(200,json={'capture_gt_v1':True,'profiles':['p']})
        if request.url.path.endswith('/api/hands/start'):
            payload=json.loads(request.content)
            assert payload=={'profile':'p','partition':'normal','sol_enabled':True,'servers':1}
            return httpx.Response(200,json={'state':'booting'})
        return httpx.Response(404)
    client=AeonClient('http://remote',token,httpx.Client(transport=httpx.MockTransport(handle)))
    assert client.start('p')['state']=='booting'
    assert ('POST','/api/tabs') in calls
    assert ('POST','/tabs/'+token+'/api/hands/start') in calls
    for invalid in ('file:///tmp/test','http://user:password@host','http://host/?token=secret'):
        with pytest.raises(ValueError):service_url(invalid)


def test_connect_explains_tunnel_reaching_wrong_service():
    client=AeonClient('http://remote','token',httpx.Client(transport=httpx.MockTransport(
        lambda request:httpx.Response(404,json={'detail':'Not Found'}))))
    try:
        with pytest.raises(RuntimeError,match='reachable, but it does not provide the aeon-webui hands API'):
            client.status()
    finally:
        client.close()


def test_status_preserves_remote_boot_failure_and_log():
    def handle(request):
        if request.url.path=='/api/tabs/config':return httpx.Response(404)
        if request.url.path=='/api/hands/status':return httpx.Response(200,json={'state':'down','profiles':[{'id':'p'}]})
        if request.url.path=='/api/robot/gt-capabilities':return httpx.Response(200,json={'capture_gt_v1':True,'startup_status_v1':True})
        return httpx.Response(200,json={'state':'failed','terminal':True,'jobid':'123',
                                       'log_tail':'Missing dataset file','exit_code':'1:0'})
    client=AeonClient('http://remote','token',httpx.Client(transport=httpx.MockTransport(handle)))
    try:
        state=client.status()
        assert state['state']=='failed' and state['terminal']
        assert state['log_tail']=='Missing dataset file'
        assert state['profiles']==[{'id':'p'}]
    finally:client.close()


def test_video_and_future_hand_generation_protocol_with_real_artifact_validation(source,tmp_path):
    anchor=source/'anchor.npz';build_bundle(source/'edited.png',source/'capture.json',anchor)
    with np.load(anchor) as a:
        points=np.repeat(a['joints3d'][None],3,axis=0);points[1:,1,:,0]+=.01
        k=a['K'].copy()
    buffer=io.BytesIO();np.savez(buffer,joints=points,K=k,fps=24.)
    video=tmp_path/'video.mp4'
    writer=imageio_ffmpeg.write_frames(str(video),(832,640),fps=24);writer.send(None)
    for _ in range(3):writer.send(np.full((640,832,3),120,dtype=np.uint8))
    writer.close()
    class Remote:
        def status(self):return {'state':'ready','selected_profile':'p','sol_enabled':True,
                    'capture_gt_bridge':{'capture_gt_v1':True,'profiles':['p'],'zero_contact_profiles':[]}}
        def register(self,path,assume_zero):
            import hashlib
            assert not assume_zero
            return {'clip_id':'custom::gt','dataset':'uploads','bundle_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'pose_reestimated':False,'server_reloaded':True}
        def generate(self,clip,instruction,seed,sol):
            assert instruction.startswith('Lift it\n') and 'RIGHT-handed' in instruction and sol
            return {'id':'abc123'}
        def request(self,*args):return {'status':'done','video':'/gallery/out.mp4','overlay':'/gallery/out_overlay.mp4'}
        def download(self,path,target):
            target.write_bytes(buffer.getvalue() if path.endswith('.npz') else video.read_bytes())
    output=tmp_path/'result'
    report=generate(source,output,'http://remote','token','p','Lift it',client=Remote(),poll_interval=0)
    assert report['status']=='completed'
    assert report['initial_gt_max_error_m']==0
    assert report['output']['shape']==[3,2,21,3]
    assert (output/'diffusion_submission.json').is_file()
    assert (output/'predicted_hand_overlay.mp4').is_file()


def bridge_module():
    path=PIPELINE/'integrations/aeon_webui/robot_gt_bridge.py'
    spec=importlib.util.spec_from_file_location('gt_bridge_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_remote_gt_import_skips_pose_estimation_and_preserves_bundle(source,tmp_path):
    module=bridge_module();path=source/'gt.npz';build_bundle(source/'edited.png',source/'capture.json',path)
    registered=[]
    async def status():return {'state':'down'}
    env={'UPLIFT_PROFILES':{'p'},'ZERO_CONTACT_PROFILES':set(),'STATE':{'hands_profile':'p'},
         'uplift_custom_row':lambda:{'custom_dir':str(tmp_path/'uploads'),'frame_db':'unused'},
         'append_frame_db_candidate':lambda path,candidate:registered.append(candidate),
         'register_custom_candidate':lambda *args:None,'UPLIFT_DATASET':'uploads','hands_status':status}
    app=FastAPI();module.install_routes(app,env)
    with TestClient(app) as client:
        body={'bundle_b64':base64.b64encode(path.read_bytes()).decode()}
        assert client.post('/api/robot/gt-frames',json=body).status_code==403
        response=client.post('/api/robot/gt-frames',json=body,headers={'X-Retarget-Client':'1'})
        assert response.status_code==200,response.text
        assert response.json()['pose_reestimated'] is False
        saved=next((tmp_path/'uploads').glob('*.npz'))
        assert saved.read_bytes()==path.read_bytes()
        assert registered[0]['clip_id'].startswith('custom::robotgt-')
        env['ZERO_CONTACT_PROFILES']={'p'}
        assert client.post('/api/robot/gt-frames',json=body,headers={'X-Retarget-Client':'1'}).status_code==409


def test_export_conditioning_from_completed_edit_without_remote_service(source,tmp_path):
    app=create_app(Config(tmp_path/'runs'))
    identifier=uuid.uuid4().hex;job=tmp_path/'runs'/identifier;job.mkdir()
    import shutil
    shutil.copytree(source,job/'result')
    (job/'status.json').write_text(json.dumps({'id':identifier,'status':'completed','mode':'capture_to_human'}))
    with TestClient(app) as client:
        response=client.post(f'/api/jobs/{identifier}/conditioning',headers={'X-Retarget-Client':'1'})
        assert response.status_code==200,response.text
        assert client.get(response.json()['bundle']).status_code==200
        status=client.get('/api/diffusion/status').json()
        assert status['state']=='not_configured'
        assert len(status['profiles'])==10


def test_future_queue_uses_existing_edit_and_preserves_literal_instruction(source,tmp_path,monkeypatch):
    app=create_app(Config(tmp_path/'runs'))
    identifier=uuid.uuid4().hex;job=tmp_path/'runs'/identifier;job.mkdir()
    import shutil
    shutil.copytree(source,job/'result')
    (job/'status.json').write_text(json.dumps({'id':identifier,'status':'completed','mode':'capture_to_human'}))
    jobs=app.state.jobs
    monkeypatch.setattr(jobs.diffusion,'config',lambda:{'url':'http://remote','token':'test'})
    monkeypatch.setattr(jobs.diffusion,'status',lambda:{'connected':True,'state':'ready','selected_profile':'p',
                        'capture_gt_bridge':{'profiles':['p'],'zero_contact_profiles':[]}})
    queued=[];monkeypatch.setattr(jobs.executor,'submit',lambda *args:queued.append(args))
    with TestClient(app) as client:
        instruction='Lift the block; literal $(do_not_execute)'
        response=client.post('/api/diffusion/generate',headers={'X-Retarget-Client':'1'},
                             json={'source_job_id':identifier,'profile':'p','instruction':instruction})
        assert response.status_code==202,response.text
        command=queued[0][-1]
        assert command[command.index('--instruction')+1]==instruction
        assert 'hand_retarget.diffusion' in command
        assert 'hand_retarget.image_edit' not in command
        assert '--uplift-python' not in command
        assert response.json()['mode']=='diffusion_video'


def test_start_rejects_service_without_gt_extension_before_allocating(tmp_path,monkeypatch):
    from hand_retarget.diffusion_service import DiffusionService
    service=DiffusionService(tmp_path)
    class Client:
        def status(self):return {'state':'down','capture_gt_bridge':None}
        def start(self,*args):raise AssertionError('Must not allocate a GPU before GT extension is available')
        def close(self):pass
    monkeypatch.setattr(service,'client',lambda:Client())
    with pytest.raises(ValueError,match='capture-GT bridge'):
        service.start('p','normal',True)


def test_rejects_left_capture_and_shares_stage2_rotation(source):
    from hand_retarget.capture_guidance import load_guidance
    guidance=load_guidance(source/'capture.json',source/'edited.png')
    report=build_bundle(source/'edited.png',source/'capture.json',source/'gt.npz')
    np.testing.assert_array_equal(guidance['wrist_R_right'],report['wrist_R_right'])
    assert guidance['valid']==[False,True] and guidance['handedness']=='right'
    data=json.loads((source/'capture.json').read_text())
    data['mano']['handedness']='left'
    (source/'capture.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='right-handed'):
        build_bundle(source/'edited.png',source/'capture.json',source/'bad.npz')


def test_custom_runtime_overrides_template_validity_without_mutating_it(monkeypatch):
    import sys
    import torch
    from types import ModuleType
    spec=importlib.util.spec_from_file_location('robot_gt_runtime',PIPELINE/'integrations/aeon_webui/robot_gt_runtime.py')
    runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
    representations=ModuleType('h3_joint.representations')
    for name,part in dict(S_LTSL=slice(0,3),S_RTSL=slice(3,6),S_LROT=slice(6,12),
                          S_RROT=slice(12,18),S_LFING=slice(18,78),S_RFING=slice(78,138)).items():
        setattr(representations,name,part)
    monkeypatch.setitem(sys.modules,'h3_joint.representations',representations)
    base={'skel_raw':torch.ones(107,138),'chan_valid':torch.ones(107,138,dtype=torch.bool)}
    for valid in ([False,True],[True,False],[True,True]):
        result=runtime.custom_validity(base,valid)
        for hand,parts in enumerate(((slice(0,3),slice(6,12),slice(18,78)),(slice(3,6),slice(12,18),slice(78,138)))):
            for part in parts:
                assert bool(torch.all(result['chan_valid'][:,part]==valid[hand]))
                assert bool(torch.all(result['skel_raw'][:,part]==int(valid[hand])))
        assert bool(base['chan_valid'].all()) and bool((base['skel_raw']==1).all())
    with pytest.raises(ValueError,match='Unsupported'):
        runtime.patch_source('print("unexpected runtime version")')


def test_joint_order_matches_aeon_demo_conversion_and_saved_capture(source):
    import ast
    upstream=PIPELINE/'vendor/aeon-webui/runtime/uplift/custom_frame.py'
    if not upstream.exists():
        pytest.skip('Optional aeon-webui audit checkout is not installed')
    tree=ast.parse(upstream.read_text())
    nodes=[node for node in tree.body if isinstance(node,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id in ('_OP','_ORDER','DPL_FROM_OP') for t in node.targets)]
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(upstream),'exec'),ns)
    openpose=['wrist','thumb_cmc','thumb_mcp','thumb_ip','thumb_tip',
              'index_mcp','index_pip','index_dip','index_tip',
              'middle_mcp','middle_pip','middle_dip','middle_tip',
              'ring_mcp','ring_pip','ring_dip','ring_tip',
              'pinky_mcp','pinky_pip','pinky_dip','pinky_tip']
    assert tuple(openpose[i] for i in ns['DPL_FROM_OP'])==KEYPOINT_NAMES
    data=json.loads((source/'capture.json').read_text())
    data['mano']['joints'][7],data['mano']['joints'][10]=data['mano']['joints'][10],data['mano']['joints'][7]
    (source/'capture.json').write_text(json.dumps(data))
    with pytest.raises(ValueError,match='index, name'):
        build_bundle(source/'edited.png',source/'capture.json',source/'wrong_order.npz')
