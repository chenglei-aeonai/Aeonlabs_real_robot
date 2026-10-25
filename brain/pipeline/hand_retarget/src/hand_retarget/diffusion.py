"""Generate future RGB video and camera-space hands from a converted GT frame."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .aeon_client import AeonClient
from .diffusion_bundle import build_bundle


def generate(source, output, url, token, profile, instruction, seed=0, assume_zero_contact=False,
             client=None, timeout=1800, poll_interval=2):
    output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running','mode':'future_video_and_hands','pose_source':'capture_ground_truth',
              'handedness':'right','valid':[False,True],
              'profile':profile,'service_url':url,'hardware_executed':False}
    def progress(stage, remote=None):
        temporary=output/'progress.json.tmp'
        temporary.write_text(json.dumps({'stage':stage,'remote':remote,'updated_at':time.time()}))
        temporary.replace(output/'progress.json')
    owned = client is None
    client = client or AeonClient(url, token)
    try:
        progress('prepare')
        bundle = output/'first_frame_gt.npz'
        report['conditioning'] = build_bundle(source/'edited.png', source/'capture.json', bundle)
        progress('preflight')
        state = client.status()
        if state.get('state') != 'ready' or state.get('selected_profile') != profile:
            raise ValueError('Selected checkpoint is not ready; load it in the Diffusion panel first.')
        capability = state.get('capture_gt_bridge')
        if not capability or not capability.get('capture_gt_v1'):
            raise ValueError('Install integrations/aeon_webui GT bridge on the inference UI before submitting captured poses.')
        if profile not in capability.get('profiles', []):
            raise ValueError('Selected checkpoint cannot serve captured RGB/GT frames.')
        if profile in capability.get('zero_contact_profiles', []) and not assume_zero_contact:
            raise ValueError('Checkpoint requires contact data; explicitly choose a no-contact start assumption or another profile.')
        progress('register')
        clip = client.register(bundle, assume_zero_contact)
        report['registration'] = clip
        if clip.get('bundle_sha256') != hashlib.sha256(bundle.read_bytes()).hexdigest() or clip.get('pose_reestimated') is not False:
            raise ValueError('Remote registration did not confirm the exact captured GT bundle.')
        if not clip.get('server_reloaded'):
            raise ValueError('Inference runtime did not load the captured GT frame; check its custom-frame catalog/runtime.')
        progress('generate')
        effective_instruction = (instruction + '\nThe visible hand and forearm are always RIGHT-handed. '
                                 'Start from the supplied MANO-right wrist pose and preserve right-hand thumb anatomy; '
                                 'do not mirror the scene or substitute a left hand.')
        submitted = client.generate(clip, effective_instruction, seed, state.get('sol_enabled', False))
        remote_id = submitted['id']
        if not isinstance(remote_id,str) or not remote_id.isalnum():raise ValueError('Invalid remote generation ID.')
        report['remote_job_id'] = remote_id
        (output/'diffusion_submission.json').write_text(json.dumps({'service_url':url,'profile':profile,
             'remote_job_id':remote_id,'clip':clip,'request':{'instruction':effective_instruction,'user_instruction':instruction,'handedness':'right','seed':seed}},indent=2)+'\n')
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            remote = client.request('GET', '/api/jobs/'+remote_id)
            progress('generate',{'status':remote.get('status'), **(remote.get('progress') or {})})
            if remote.get('status')=='failed':raise RuntimeError(remote.get('error','Remote generation failed.'))
            if remote.get('status')=='done':break
            time.sleep(poll_interval)
        else:raise TimeoutError('Generation timed out; remote job may still run. See diffusion_submission.json.')
        video_path=remote.get('video','')
        if not video_path.endswith('.mp4'):raise ValueError('Remote job produced no video path.')
        progress('download')
        client.download(video_path, output/'video.mp4')
        client.download(video_path[:-4]+'_joints.npz',output/'future_hands.npz')
        if remote.get('overlay'):client.download(remote['overlay'],output/'predicted_hand_overlay.mp4')
        progress('validate')
        with np.load(bundle,allow_pickle=False) as anchor, np.load(output/'future_hands.npz',allow_pickle=False) as future:
            joints=np.asarray(future['joints']);k=np.asarray(future['K']);fps=float(future['fps'])
            if joints.ndim!=4 or joints.shape[1:]!=(2,21,3) or len(joints)<2 or not np.isfinite(joints[:,1]).all():
                raise ValueError('Generated hand arrays have an invalid shape or nonfinite right-hand positions.')
            if not np.isfinite(fps) or fps<=0 or k.shape!=(3,3) or not np.allclose(k,anchor['K'],atol=1e-4,rtol=0):
                raise ValueError('Generated camera intrinsics/fps do not match GT conditioning.')
            error=float(np.max(np.linalg.norm(joints[0,1]-anchor['joints3d'][1],axis=-1)))
            report['initial_gt_max_error_m']=error
            if 'valid' in future:
                valid=np.asarray(future['valid'])
                if valid.dtype!=bool or valid.shape not in ((2,),(len(joints),2)) or not np.all(valid==[False,True]):
                    raise ValueError('Generated handedness mask differs from right-only capture.')
            if 'wrist_R' in future:
                rotations=np.asarray(future['wrist_R'])
                if rotations.shape!=(len(joints),2,3,3) or not np.isfinite(rotations[:,1]).all():
                    raise ValueError('Generated wrist rotations have an invalid shape or nonfinite values.')
                relative=rotations[0,1]@anchor['wrist_R'][1].T
                angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
                report['initial_wrist_rotation_error_deg']=angle
                if angle>.1:
                    raise ValueError(f'Generated initial MANO-right wrist rotation differs by {angle:.4f} degrees.')
            report['runtime_exports_hand_validity']='valid' in future
            report['runtime_exports_wrist_rotation']='wrist_R' in future
            if error>1e-4:raise ValueError(f'Generated first-frame pose differs from captured GT by {error:.6f} m.')
            report['output']={'frames':len(joints),'fps':fps,'shape':list(joints.shape),
                              'coordinate_frame':'optical camera','units':'metres','right_hand_index':1,
                              'handedness':'right','valid':[False,True],
                              'left_slot':'absent; do not interpret unmasked decoder values as another hand'}
        import imageio_ffmpeg
        reader=imageio_ffmpeg.read_frames(str(output/'video.mp4'))
        try:
            metadata=next(reader)
            if tuple(metadata['size'])!=tuple(report['conditioning']['image_size_px']):
                raise ValueError('Generated video dimensions differ from the calibrated conditioning image.')
            if not np.isclose(metadata['fps'],report['output']['fps'],atol=1e-3):
                raise ValueError('Generated video and 3D hands have different frame rates.')
            if sum(1 for _ in reader)!=report['output']['frames']:
                raise ValueError('Generated video and 3D hands have different frame counts.')
        finally:reader.close()
        report['status']='completed';progress('completed')
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}');progress('failed');raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        if owned:client.close()
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-result',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--service-url',required=True)
    parser.add_argument('--token',required=True)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--instruction',required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--assume-zero-contact',action='store_true')
    args=parser.parse_args()
    generate(args.source_result,args.out,args.service_url,args.token,args.profile,args.instruction,
             args.seed,args.assume_zero_contact)


if __name__=='__main__':main()
