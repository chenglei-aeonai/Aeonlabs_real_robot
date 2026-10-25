import base64
import copy
import json
from pathlib import Path
import subprocess
import time

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from hand_retarget import capture_bridge, codex_edit, webapp
from hand_retarget.capture_guidance import load_guidance, overlay_on_edited
from hand_retarget.skeleton import KEYPOINT_NAMES

HEADERS = {'X-Retarget-Client': '1'}


def bundle(folder):
    folder.mkdir(parents=True)
    matrix = [[32., 0, 32.], [0, 32., 24.], [0, 0, 1.]]
    joints = []
    for index, name in enumerate(KEYPOINT_NAMES):
        point = [.125 + index * .001, .21 if index == 4 else .25, 1.]
        pixel = (np.array(matrix) @ point)[:2]
        joints.append({'index': index, 'name': name, 'position_camera_m': point,
                       'pixel_undistorted': pixel.tolist(), 'synthetic': index in (0, 9)})
    data = {'position_units': 'metres', 'camera': {'width': 64, 'height': 48,
             'undistorted_intrinsics': {'matrix': matrix, 'dist_coeffs': [0] * 5}},
            'capture': {'captured_at_utc': '2026-10-24T00:00:00Z'},
            'mano': {'coordinate_frame': 'RealSense color optical', 'joints': joints}}
    data['hand_joints'] = copy.deepcopy(joints)
    (folder / 'hand_joints.json').write_text(json.dumps(data))
    for name in ('raw.png', 'undistorted.png', 'overlay.png', 'mano_overlay.png'):
        Image.new('RGB', (64, 48), 'blue').save(folder / name)
    return data


@pytest.fixture
def capture_app(tmp_path, monkeypatch):
    from hand_retarget import calibration_status as registry
    monkeypatch.setattr(registry, 'DIRECTORY', tmp_path)
    np.save(tmp_path/'20261024_190421_base_pose_robot_ur5e.npy', np.eye(4))
    np.save(tmp_path/'20261024_190421_camera_extrinsic_138322253845.npy', np.eye(4))
    runner = tmp_path / 'run_capture.sh'; runner.write_text('test runner')
    def run(command, **kwargs):
        assert command[0] == str(runner)
        assert command[1] == '--output_dir'
        folder = Path(command[2]) / 'timestamp'
        data = bundle(folder)
        data['calibration'] = registry.status()['provenance']
        (folder/'hand_joints.json').write_text(json.dumps(data))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(capture_bridge.subprocess, 'run', run)
    config = webapp.Config(tmp_path / 'out/runs', capture_runner=runner)
    app = webapp.create_app(config)
    return app


def test_capture_needs_no_inference_services_and_serves_paired_artifacts(capture_app):
    with TestClient(capture_app) as client:
        assert client.post('/api/captures').status_code == 403
        response = client.post('/api/captures', headers=HEADERS)
        assert response.status_code == 201, response.text
        capture = response.json()
        assert len(capture['mano']['joints']) == 21
        for url in capture['files'].values():
            assert client.get(url).status_code == 200
        assert client.get(f'/api/captures/{capture["id"]}/files/capture.log').status_code == 404
        assert client.get('/api/captures/invalid').status_code == 404
        lock = capture_app.state.jobs.captures.lock
        lock.acquire()
        try:
            assert client.post('/api/captures', headers=HEADERS).status_code == 409
        finally:
            lock.release()


def test_edit_job_copies_exact_capture_and_does_not_require_wilor_or_h3(capture_app, monkeypatch):
    monkeypatch.setattr(codex_edit, 'readiness', lambda: (True, 'ready'))
    queued = []
    monkeypatch.setattr(capture_app.state.jobs.executor, 'submit', lambda *args: queued.append(args))
    with TestClient(capture_app) as client:
        captured = client.post('/api/captures', headers=HEADERS).json()
        request = {'capture_id': captured['id'], 'stop_after': 'edit',
                   'instruction': 'Convert appearance using the captured GT pose.'}
        response = client.post('/api/jobs', headers=HEADERS, json=request)
        assert response.status_code == 202, response.text
        job = capture_app.state.jobs.config.runs / response.json()['id']
        original = capture_app.state.jobs.captures.locate(captured['id'])
        assert (job / 'input.png').read_bytes() == (original / 'undistorted.png').read_bytes()
        assert (job / 'capture.json').read_bytes() == (original / 'hand_joints.json').read_bytes()
        command = queued[0][-1]
        assert 'hand_retarget.image_edit' in command
        assert '--capture-json' in command
        assert '--uplift-python' not in command and '--h3-url' not in command
        persisted = json.loads((job / 'request.json').read_text())
        assert persisted['hfov'] == 90
        assert persisted['fixed_camera_pitch'] is None
        request['image_b64'] = base64.b64encode((job / 'input.png').read_bytes()).decode()
        assert client.post('/api/jobs', headers=HEADERS, json=request).status_code == 400


def test_guidance_preserves_gt_and_uses_only_intrinsics_for_reprojection(tmp_path):
    folder = tmp_path / 'capture'
    original = bundle(folder)
    # Deliberately unrelated extrinsics must never affect the output pixels.
    original['camera_from_robot_base'] = (np.eye(4) * 50).tolist()
    (folder / 'hand_joints.json').write_text(json.dumps(original))
    guidance = load_guidance(folder / 'hand_joints.json', folder / 'undistorted.png')
    assert guidance['mano'] == original['mano']
    assert guidance['pose_source'] == 'capture_ground_truth'
    assert guidance['pose_is_authoritative']
    Image.new('RGB', (64, 48), 'red').save(folder / 'edited.png')
    overlay_on_edited(folder / 'edited.png', guidance, folder / 'edited_overlay.png')
    output = np.asarray(Image.open(folder / 'edited_overlay.png'))
    assert output.shape == (48, 64, 3)
    assert not np.array_equal(output, np.asarray(Image.open(folder / 'edited.png')))
    assert guidance['mano']['joints'] == original['mano']['joints']
    bad = copy.deepcopy(original)
    bad['mano']['joints'][0]['pixel_undistorted'][0] += 10
    (folder / 'bad.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='projection'):
        load_guidance(folder / 'bad.json', folder / 'undistorted.png')


def test_codex_receives_gt_intrinsics_and_camera_points_without_pose_estimation(tmp_path, monkeypatch):
    folder = tmp_path / 'capture'; original = bundle(folder)
    prompt = tmp_path / 'prompt.txt'; prompt.write_text('Operator edit instructions')
    monkeypatch.setenv('CODEX_EDIT_PROMPT_FILE', str(prompt))
    calls = []
    def generate(workspace, images, instruction, schema):
        calls.append(instruction)
        assert len(images) == 4
        assert images[-1].name == 'alignment_reference.png'
        assert 'authoritative ground-truth' in instruction
        assert 'never apply extrinsics again' in instruction
        assert 'CAMERA INTRINSICS FOR THIS EXACT CROP' in instruction
        assert 'always the human RIGHT hand' in instruction
        assert 'wrist_R_right' in instruction
        assert 'THUMB BASE/TIP' in instruction and 'Do not turn the palm over' in instruction
        assert 'orientation_check' in schema['required']
        assert json.dumps(original['mano']['joints'][0]['position_camera_m']) in instruction
        workspace.mkdir()
        Image.new('RGB', (64, 48), 'red').save(workspace / 'edited.png')
        return {'status': 'completed', 'image_path': 'edited.png', 'detail': 'test', 'orientation_check': 'matched'}
    monkeypatch.setattr(codex_edit, 'run_codex', generate)
    from hand_retarget.inference_clients import replace_hands
    mask = Image.new('L', (64, 48)); mask.paste(255, (10, 10, 40, 40)); mask.save(folder / 'mask.png')
    result = replace_hands(folder / 'undistorted.png', folder / 'mask.png', folder / 'edited.png',
                           model='', capture_json=folder / 'hand_joints.json')
    assert len(calls) == 1
    assert result['pose_source'] == 'capture_ground_truth'
    assert (folder / 'edited_mano_overlay.png').is_file()
    assert json.loads((folder / 'capture_guidance.json').read_text())['mano'] == original['mano']


def test_crop_preserves_3d_gt_and_updates_intrinsics_consistently(tmp_path):
    from hand_retarget.capture_crop import crop_bundle
    source = tmp_path / 'source'; original = bundle(source)
    # An impulse at the first GT pixel verifies PIL sampling and K use the same
    # pixel-center convention, including the half-pixel shift when resizing.
    pixels = np.zeros((48, 64, 3), np.uint8)
    pixels[32, 36] = 255
    Image.fromarray(pixels).save(source / 'undistorted.png')
    before = (source / 'hand_joints.json').read_bytes()
    destination = tmp_path / 'cropped'
    # 52:40 = 832:640; exactly 16x resize, with nonzero crop offsets.
    result = crop_bundle(source, destination, x=6, y=4, width=52, height=40,
                         output_width=832, output_height=640)
    transform = np.array(result['crop']['source_to_crop_pixels'])
    expected_k = transform @ np.array(original['camera']['undistorted_intrinsics']['matrix'])
    np.testing.assert_allclose(result['camera']['undistorted_intrinsics']['matrix'], expected_k)
    for old, new in zip(original['mano']['joints'], result['mano']['joints']):
        assert new['position_camera_m'] == old['position_camera_m']
        expected = transform @ [*old['pixel_undistorted'], 1.]
        np.testing.assert_allclose(new['pixel_undistorted'], expected[:2])
    assert (source / 'hand_joints.json').read_bytes() == before
    assert Image.open(destination / 'undistorted.png').size == (832, 640)
    assert Image.open(destination / 'raw.png').size == (64, 48)
    intensity = np.asarray(Image.open(destination / 'undistorted.png'))[:, :, 0].astype(float)
    yy, xx = np.indices(intensity.shape)
    center = [np.sum(xx * intensity) / intensity.sum(), np.sum(yy * intensity) / intensity.sum()]
    np.testing.assert_allclose(center, result['mano']['joints'][0]['pixel_undistorted'], atol=.01)
    assert load_guidance(destination / 'hand_joints.json', destination / 'undistorted.png')['crop']


@pytest.mark.parametrize('changes', [{'x': 30}, {'width': 60}, {'x': -1}, {'output_width': 800}])
def test_invalid_or_stretched_crops_are_rejected(tmp_path, changes):
    from hand_retarget.capture_crop import crop_bundle
    source = tmp_path / 'source'; bundle(source)
    parameters = dict(x=6, y=4, width=52, height=40, output_width=832, output_height=640)
    parameters.update(changes)
    with pytest.raises(ValueError):
        crop_bundle(source, tmp_path / 'output', **parameters)


def test_crop_api_produces_distinct_immutable_capture(capture_app):
    with TestClient(capture_app) as client:
        source = client.post('/api/captures', headers=HEADERS).json()
        parameters = dict(x=6, y=4, width=52, height=40, output_width=832, output_height=640)
        response = client.post(f'/api/captures/{source["id"]}/crop', json=parameters, headers=HEADERS)
        assert response.status_code == 201, response.text
        cropped = response.json()
        assert cropped['id'] != source['id']
        assert cropped['camera']['width'] == 832
        assert cropped['camera']['height'] == 640
        assert client.get(f'/api/captures/{source["id"]}').json()['camera']['width'] == 64
        parameters['x'] = 40
        assert client.post(f'/api/captures/{source["id"]}/crop', json=parameters, headers=HEADERS).status_code == 400


def test_arm_alignment_uses_captured_geometry_and_crop_intrinsics(tmp_path):
    from hand_retarget.capture_crop import crop_bundle
    source = tmp_path / 'source'; data = bundle(source)
    camera_from_base = np.eye(4); camera_from_base[2, 3] = 2
    data['camera_from_robot_base'] = camera_from_base.tolist()
    data['robot_state'] = {'arm_q_rad': [0.] * 6, 'tcp_pose_base_xyzw': [0, 0, 0, 0, 0, 0, 1]}
    (source / 'hand_joints.json').write_text(json.dumps(data))
    before = load_guidance(source / 'hand_joints.json', source / 'undistorted.png')
    np.testing.assert_allclose(before['arm']['joints'][2]['position_camera_m'], [-.425, 0, 2.1625])
    np.testing.assert_allclose(before['arm']['joints'][6]['position_camera_m'], [-.8172, -.2329, 2.0628])
    assert before['arm']['human_forearm_reference']['wrist_camera_m'] == data['mano']['joints'][0]['position_camera_m']
    destination = tmp_path / 'crop'
    crop_bundle(source, destination, x=6, y=4, width=52, height=40, output_width=832, output_height=640)
    after = load_guidance(destination / 'hand_joints.json', destination / 'undistorted.png')
    matrix = np.array(after['camera']['undistorted_intrinsics']['matrix'])
    for original, cropped in zip(before['arm']['joints'], after['arm']['joints']):
        assert original['position_camera_m'] == cropped['position_camera_m']
        xyz = np.array(cropped['position_camera_m']); uv = matrix @ xyz
        np.testing.assert_allclose(cropped['pixel_undistorted'], uv[:2]/uv[2])


def test_applied_crop_defaults_persist_for_recapture_and_restart(capture_app):
    with TestClient(capture_app) as client:
        assert client.get('/api/crop-defaults').json() == {'defaults': None}
        first = client.post('/api/captures', headers=HEADERS).json()
        defaults = {'version': 1, 'width': 832, 'height': 640, 'zoom': 1.2, 'x': 50, 'y': 50}
        parameters = dict(x=6, y=4, width=52, height=40, output_width=832,
                          output_height=640, defaults=defaults)
        response = client.post(f'/api/captures/{first["id"]}/crop', headers=HEADERS, json=parameters)
        assert response.status_code == 201, response.text
        assert client.get('/api/crop-defaults').json()['defaults'] == defaults
        second = client.post('/api/captures', headers=HEADERS).json()
        assert client.get('/api/crop-defaults').json()['defaults'] == defaults
        response = client.post(f'/api/captures/{second["id"]}/crop', headers=HEADERS, json=parameters)
        assert response.status_code == 201
        assert response.json()['crop']['rectangle_source_px'] == [6, 4, 52, 40]
        manager = capture_app.state.jobs.captures
        restarted = capture_bridge.Captures(manager.root, manager.runner)
        assert restarted.saved_defaults() == defaults
        parameters['defaults'] = dict(defaults, zoom=2)
        assert client.post(f'/api/captures/{second["id"]}/crop', headers=HEADERS, json=parameters).status_code == 400
        assert restarted.saved_defaults() == defaults


def test_latest_capture_restores_applied_frame_and_original_source(capture_app):
    with TestClient(capture_app) as client:
        assert client.get('/api/captures/latest').status_code == 404
        original = client.post('/api/captures', headers=HEADERS).json()
        crop = client.post(f'/api/captures/{original["id"]}/crop', headers=HEADERS,
                           json=dict(x=6, y=4, width=52, height=40,
                                     output_width=832, output_height=640)).json()
        latest = client.get('/api/captures/latest').json()
        assert latest['id'] == crop['id']
        assert latest['source_capture_id'] == original['id']
        assert latest['crop']['rectangle_source_px'] == [6, 4, 52, 40]


@pytest.mark.parametrize('check', ['mismatched', 'uncertain', None])
def test_capture_edit_preserves_review_candidate_but_rejects_missing_check(tmp_path, monkeypatch, check):
    folder=tmp_path/'capture';bundle(folder)
    Image.new('L',(64,48),255).save(folder/'mask.png')
    def generate(workspace, images, instruction, schema):
        workspace.mkdir()
        Image.new('RGB',(64,48),'red').save(workspace/'edited.png')
        result={'status':'completed','image_path':'edited.png','detail':'thumb on wrong side'}
        if check is not None:result['orientation_check']=check
        return result
    monkeypatch.setattr(codex_edit,'run_codex',generate)
    if check is None:
        with pytest.raises(RuntimeError, match='omitted'):
            codex_edit.edit_image(folder/'undistorted.png',folder/'mask.png',folder/'accepted.png',folder/'hand_joints.json')
        assert not (folder/'accepted.png').exists()
    else:
        report = codex_edit.edit_image(folder/'undistorted.png',folder/'mask.png',folder/'accepted.png',folder/'hand_joints.json')
        assert (folder/'accepted.png').is_file()
        assert report['requires_review'] and not report['diffusion_eligible']
        assert report['orientation_self_check'] == check
    assert list(folder.glob('codex-edit-*/edited.png'))


def test_older_capture_keeps_its_calibration_for_edit(capture_app, monkeypatch):
    from hand_retarget import calibration_status as registry
    monkeypatch.setattr(codex_edit, 'readiness', lambda: (True, 'ready'))
    queued = []
    monkeypatch.setattr(capture_app.state.jobs.executor, 'submit', lambda *args: queued.append(args))
    with TestClient(capture_app) as client:
        captured = client.post('/api/captures', headers=HEADERS).json()
        np.save(registry.DIRECTORY/'20261025_090000_base_pose_robot_ur5e.npy', np.eye(4))
        np.save(registry.DIRECTORY/'20261025_090000_camera_extrinsic_138322253845.npy', np.eye(4))
        state = client.get('/api/calibration/status', params={'capture_id': captured['id']}).json()
        assert state['latest'] == '20261025_090000' and not state['current']
        response = client.post('/api/jobs', headers=HEADERS, json={
            'capture_id': captured['id'], 'stop_after': 'edit', 'instruction': 'Convert hand'})
        assert response.status_code == 202, response.text
        assert queued
        job = capture_app.state.jobs.locate(response.json()['id'])
        documents = list(job.rglob('capture.json'))
        assert documents
        saved = json.loads(documents[0].read_text())
        assert registry.timestamp(saved['calibration']) == '20261024_190421'


def test_unconfirmed_edit_cannot_enter_diffusion(capture_app):
    jobs = capture_app.state.jobs
    identifier = 'b'*32
    folder = jobs.config.runs/identifier
    result = folder/'result'; result.mkdir(parents=True)
    (folder/'status.json').write_text(json.dumps({'status': 'completed'}))
    (result/'edited.png').write_bytes(b'candidate')
    (result/'capture.json').write_text('{}')
    (result/'manifest.json').write_text(json.dumps({'edit': {
        'diffusion_eligible': False, 'orientation_self_check': 'uncertain'}}))
    with pytest.raises(ValueError, match='review candidate'):
        jobs.source_edit(identifier)
    jobs.close()
