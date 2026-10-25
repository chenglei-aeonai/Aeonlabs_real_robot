"""Add GT custom-frame import to an isolated aeon-webui checkout; no WiLoR."""
import asyncio
import base64
import hashlib
import io
import json
from pathlib import Path
import threading
import subprocess
import time
import zipfile

import httpx
import numpy as np
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

CATALOG_LOCK = threading.Lock()


class GTFrame(BaseModel):
    bundle_b64: str = Field(max_length=36_000_000)
    assume_zero_contact: bool = False


def validate_bundle(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 32_000_000:
            raise ValueError('GT frame exceeds 32 MB unpacked.')
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in ('frame','joints3d','wrist_R','valid','K')}
    frame, joints, rotations, valid, k = (bundle[key] for key in ('frame','joints3d','wrist_R','valid','K'))
    if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
            or min(frame.shape[:2]) < 32 or max(frame.shape[:2]) > 2048
            or any(value % 32 for value in frame.shape[:2])):
        raise ValueError('RGB dimensions must be multiples of 32, between 32 and 2048.')
    if joints.shape != (2,21,3) or rotations.shape != (2,3,3) or valid.shape != (2,) or valid.dtype != bool:
        raise ValueError('Invalid joint, wrist rotation or validity shape.')
    if valid.tolist() != [False, True]:
        raise ValueError('Expected only the captured right hand; do not fabricate a left hand.')
    if not np.isfinite(joints[1]).all() or np.any(joints[1,:,2] <= 0):
        raise ValueError('Invalid camera-space right-hand depth.')
    if (k.shape != (3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1]) <= 0
            or not np.allclose(k[2], [0,0,1])):
        raise ValueError('Invalid GT camera intrinsics.')
    if (not np.isfinite(rotations).all()
            or not np.allclose(rotations @ rotations.transpose(0,2,1), np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-5)):
        raise ValueError('Wrist rotations must be proper rotation matrices.')
    if np.any(joints[0]) or not np.allclose(rotations[0], np.eye(3)):
        raise ValueError('Absent left-hand anchor must be explicitly neutral.')
    return bundle


def install_routes(app, env):
    lock = CATALOG_LOCK
    terminal_cache = {'jobid': None, 'at': 0., 'value': {}}

    @app.get('/api/robot/gt-capabilities')
    def capabilities():
        return {'capture_gt_v1': True, 'startup_status_v1': True, 'pose_reestimated': False,
                'profiles': sorted(env['UPLIFT_PROFILES']),
                'zero_contact_profiles': sorted(env['ZERO_CONTACT_PROFILES']),
                'source': 'captured camera-frame GT, original intrinsics, no WiLoR'}

    @app.get('/api/robot/gt-startup')
    async def startup():
        """Keep the owned allocation's terminal failure/log visible after squeue drops it."""
        state = dict(await env['hands_status']())
        owned = str(env['STATE'].get('hands_owned_job') or '')
        if state.get('jobid'):
            if state.get('state') in ('booting', 'failed'):
                state['log_tail'] = await asyncio.to_thread(env['hands_log_tail'], state['jobid'], 40)
            return state
        if state.get('state') != 'down' or not owned.isdigit():
            return state
        if terminal_cache['jobid'] == owned and time.monotonic() - terminal_cache['at'] < 10:
            return {**state, **terminal_cache['value']}
        try:
            result = await asyncio.to_thread(
                subprocess.run, ['sacct', '-X', '-n', '-j', owned,
                                 '--format=JobIDRaw,State,ExitCode', '--parsable2'],
                capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            return {**state, 'startup_status_error': str(error)}
        terminal = {'FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED',
                    'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'CANCELLED', 'COMPLETED'}
        for line in result.stdout.splitlines() if result.returncode == 0 else []:
            fields = line.strip().split('|')
            if len(fields) < 3 or fields[0] != owned:
                continue
            slurm_state = fields[1].split()[0].rstrip('+')
            if slurm_state not in terminal:
                continue
            log = await asyncio.to_thread(env['hands_log_tail'], owned, 40)
            value = {'jobid': owned, 'state': 'cancelled' if slurm_state == 'CANCELLED' else
                     'completed' if slurm_state == 'COMPLETED' else 'failed',
                     'terminal': True, 'slurm_state': slurm_state, 'exit_code': fields[2],
                     'log_tail': log,
                     'load_error': f'Slurm job {owned}: {slurm_state} (exit {fields[2]}). See the boot log.'}
            terminal_cache.update(jobid=owned, at=time.monotonic(), value=value)
            return {**state, **value}
        return state

    @app.post('/api/robot/gt-frames')
    async def register(request: Request, body: GTFrame):
        if request.headers.get('x-retarget-client') != '1':
            raise HTTPException(403, 'Missing X-Retarget-Client: 1')
        profile = env['STATE']['hands_profile']
        if profile not in env['UPLIFT_PROFILES']:
            raise HTTPException(409, 'This checkpoint does not support custom RGB/pose frames.')
        zero = profile in env['ZERO_CONTACT_PROFILES']
        if zero and not body.assume_zero_contact:
            raise HTTPException(409, 'This checkpoint requires contact/force. Explicitly confirm a no-contact start or choose another profile.')
        try:
            payload = base64.b64decode(body.bundle_b64, validate=True)
            bundle = validate_bundle(payload)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as error:
            raise HTTPException(422, str(error))
        digest = hashlib.sha256(payload).hexdigest()
        stem = 'robotgt-' + digest[:32]
        cid = 'custom::' + stem
        row = env['uplift_custom_row']()
        output = Path(row['custom_dir']) / (stem + '.npz')
        k = bundle['K']; width = bundle['frame'].shape[1]
        hfov = float(np.degrees(np.arctan(k[0,2]/k[0,0]) + np.arctan((width-k[0,2])/k[0,0])))
        projected = bundle['joints3d'][1] @ k.T
        uv = projected[:,:2] / projected[:,2:]
        inside = int(((uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<bundle['frame'].shape[0])).sum())
        candidate = {'candidate_id':cid, 'clip_id':cid, 'frame_index':0, 'camera_hfov_deg':hfov,
                     'visible_joints':[0,inside], 'annotated_through_window':[False,True],
                     'source':'robot capture GT + converted human RGB', 'split':'custom', 'caption':''}
        with lock:
            output.parent.mkdir(parents=True, exist_ok=True)
            if not output.exists():
                temp = output.with_suffix('.tmp')
                with temp.open('wb') as stream:stream.write(payload)
                temp.replace(output)
            output.with_suffix('.json').write_text(json.dumps({'pose_source':'capture_ground_truth',
                'pose_reestimated':False,'bundle_sha256':digest,'K':k.tolist(),
                'contact_assumed_zero':zero,'valid':[False,True]})+'\n')
            env['append_frame_db_candidate'](Path(row['frame_db']), candidate)
            env['register_custom_candidate'](env['UPLIFT_DATASET'], candidate)
        state = await env['hands_status']()
        reloaded = False
        if state['state'] == 'ready':
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    response = await client.post(f"http://{state['node']}:{env['HANDS_PORT']}/v1/custom-frames/reload")
                    reloaded = response.status_code == 200 and cid in response.json().get('custom_frames', [])
            except httpx.HTTPError:
                pass
        return {'clip_id':cid,'dataset':env['UPLIFT_DATASET'],'profile':profile,
                'bundle_sha256':digest,'pose_source':'capture_ground_truth',
                'pose_reestimated':False,'contact_assumed_zero':zero,'server_reloaded':reloaded}
