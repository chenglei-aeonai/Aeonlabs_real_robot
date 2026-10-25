"""Offline DG5F retargeting of generated camera-space hands into the captured robot base."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .h3_actions import rigid_transforms
from .placement import apply
from .skeleton import HumanDemonstration, KEYPOINT_NAMES, palm_frame


def calibrated_input(folder):
    capture=json.loads((folder/'capture.json').read_text())
    if capture.get('handedness','right')!='right' or capture['mano'].get('handedness','right')!='right':
        raise ValueError('Only the captured right hand is supported.')
    if capture.get('position_units')!='metres' or capture['mano'].get('coordinate_frame')!='RealSense color optical':
        raise ValueError('Expected camera optical coordinates in metres.')
    if [j['name'] for j in capture['mano']['joints']]!=list(KEYPOINT_NAMES):
        raise ValueError('Capture MANO joint order is inconsistent.')
    camera_from_base=rigid_transforms(capture['camera_from_robot_base'],1)[0]
    base_from_camera=np.linalg.inv(camera_from_base)
    with np.load(folder/'future_hands.npz',allow_pickle=False) as data:
        joints=np.asarray(data['joints'],dtype=float);fps=float(data['fps']);k=data['K']
        if joints.ndim!=4 or joints.shape[1:]!=(2,21,3) or not 2<=len(joints)<=10000:
            raise ValueError('Expected generated joints shaped (T,2,21,3).')
        if not np.isfinite(joints[:,1]).all() or not np.isfinite(fps) or not 0<fps<=240:
            raise ValueError('Nonfinite right-hand joints or invalid frame rate.')
        if 'valid' in data:
            valid=data['valid']
            if valid.dtype!=bool or valid.shape not in ((2,),(len(joints),2)) or not np.all(valid==[False,True]):
                raise ValueError('Generated validity must identify only the right hand.')
        if not np.allclose(k,capture['camera']['undistorted_intrinsics']['matrix'],atol=1e-4,rtol=0):
            raise ValueError('Generated intrinsics differ from the source capture.')
    captured=np.asarray([j['position_camera_m'] for j in capture['mano']['joints']])
    if np.max(np.linalg.norm(joints[0,1]-captured,axis=-1))>1e-4:
        raise ValueError('Generated first frame does not match the source capture.')
    palm_axes=joints[:,1,[1,4,7,10]].mean(axis=1)-joints[:,1,0]
    across=joints[:,1,1]-joints[:,1,7]
    if np.any(np.linalg.norm(palm_axes,axis=1)<1e-7) or np.any(np.linalg.norm(np.cross(across,palm_axes),axis=1)<1e-9):
        raise ValueError('Generated trajectory contains a degenerate palm frame.')
    base_joints=apply(base_from_camera,joints[:,1])
    roundtrip=float(np.max(np.linalg.norm(apply(camera_from_base,base_joints)-joints[:,1],axis=-1)))
    return capture,joints[:,1].copy(),base_joints,fps,base_from_camera,roundtrip


def retarget(folder,output):
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','mode':'generated_hand_retarget','handedness':'right',
            'coordinate_frame':'robot_base','position_units':'metres','angle_units':'radians',
            'hardware_executed':False,'execution_ready':False,'arm_ik_computed':False,
            'preview_type':'floating DG5F hand in calibrated robot-base coordinates'}
    def progress(stage,step=0,total=0):
        temp=output/'progress.json.tmp'
        temp.write_text(json.dumps({'stage':stage,'step':step,'total':total,'updated_at':time.time()}))
        temp.replace(output/'progress.json')
    try:
        progress('validate')
        capture,camera_joints,base_joints,fps,base_from_camera,error=calibrated_input(folder)
        from .pipeline import HandStage, ArmStage, Settings, Replay, export_robot_demonstration
        from .robot import RobotKinematics, HOME_ARM_Q, HAND_JOINT_NAMES, HAND_LOWER, HAND_UPPER, ROBOT_URDF
        robot=RobotKinematics()
        # Only the hand's fixed mount-to-palm transform is needed. The simulated
        # world/base offset and default scene placement are deliberately not used.
        robot._forward(HOME_ARM_Q,np.zeros(20))
        mount=robot.data.oMf[robot.model.getFrameId('rl_dg_mount')]
        mount_from_palm=(mount.inverse()*robot.data.oMf[robot.palm_frame]).homogeneous.copy()
        camera_from_mount=rigid_transforms(capture['camera_from_hand_mount'],1)[0]
        captured_palm=base_from_camera@camera_from_mount@mount_from_palm
        if 'robot_state' in capture and 'hand_joints' in capture:
            captured_q=np.asarray(capture['robot_state']['hand_q_rad'],dtype=float)
            reference={joint['name']:joint['position_camera_m'] for joint in capture['hand_joints']}
            expected=np.asarray([reference[f'finger_{f}_{j}'] for f in range(1,6) for j in range(1,5)])
            observed=apply(camera_from_mount@mount_from_palm,robot.hand_points_in_palm(captured_q,HAND_JOINT_NAMES))
            model_error=float(np.max(np.linalg.norm(expected-observed,axis=-1)))
            if model_error>1e-3:raise ValueError('DG5F URDF does not match the source capture geometry within 1 mm.')
            report['capture_urdf_joint_match_max_error_m']=model_error
        (output/'capture.json').write_bytes((folder/'capture.json').read_bytes())
        captured_human=apply(base_from_camera,np.asarray([j['position_camera_m'] for j in capture['mano']['joints']]))
        human_from_robot_palm=np.linalg.inv(palm_frame(captured_human))@captured_palm
        demo=HumanDemonstration('generated_right_hand',fps,base_joints)
        if np.any(demo.finger_lengths()<1e-5):raise ValueError('Degenerate generated finger lengths.')
        stage=HandStage(demo,robot,Settings(method='vector',optimizer='own',filter=False))
        settings=Settings(method='vector',optimizer='own',filter=False)
        arm_stage=None;arm_frames=[]
        if 'arm_q_rad' in capture.get('robot_state',{}):
            # Upstream kinematics normally uses a simulated world placement.
            # Express it instead in the UR controller base without changing GT.
            base_frame=robot.model.getFrameId('base')
            if base_frame>=len(robot.model.frames):raise ValueError('URDF lacks controller base frame.')
            robot.base=robot.data.oMf[base_frame].inverse()
            initial_q=np.asarray(capture['robot_state']['arm_q_rad'],float)
            if not np.allclose(robot.palm_pose(initial_q),captured_palm,atol=.002):
                raise ValueError('Capture arm pose differs from calibrated palm/URDF geometry.')
            arm_stage=ArmStage(robot,np.eye(4),False,fps,initial_arm_q=initial_q,palm_offset=human_from_robot_palm)
        qs=[];poses=[];points=[];errors=[];penetrations=[]
        for index,skeleton in enumerate(base_joints):
            frame=stage.step(skeleton)
            q=frame.hand_q
            if not np.isfinite(q).all() or np.any(q<HAND_LOWER-1e-8) or np.any(q>HAND_UPPER+1e-8):
                raise ValueError(f'Invalid retargeted joint angles at frame {index}.')
            if arm_stage is not None:
                arm_frame=arm_stage.step(frame);arm_frames.append(arm_frame)
                pose=arm_frame.palm_pose
            else:
                pose=palm_frame(skeleton)@human_from_robot_palm
            origins=robot.hand_points_in_palm(q,HAND_JOINT_NAMES).reshape(5,4,3)
            tips=robot.fingertips_in_palm(q)
            local=np.concatenate([np.zeros((1,3)),np.concatenate([origins,tips[:,None,:]],axis=1).reshape(25,3)])
            qs.append(q.copy());poses.append(pose);points.append(apply(pose,local))
            errors.append(np.linalg.norm(tips-frame.fingertip_targets_in_palm,axis=1))
            penetrations.append(max(frame.finger_penetration.values(),default=0.))
            progress('retarget',index+1,len(base_joints))
        qs=np.asarray(qs);poses=np.asarray(poses);points=np.asarray(points);errors=np.asarray(errors)
        names=['palm']+[name for f in range(1,6) for name in [*(f'rj_dg_{f}_{j}' for j in range(1,5)),f'rl_dg_{f}_tip']]
        arm_data={}
        if arm_frames:
            arm_data=dict(arm_q_rad=np.stack([f.arm_q for f in arm_frames]),
                          palm_target_robot_base=np.stack([f.palm_target for f in arm_frames]),
                          arm_position_error_m=np.array([f.palm_position_error for f in arm_frames]),
                          arm_rotation_error_rad=np.array([f.palm_rotation_error for f in arm_frames]))
            replay=Replay(demo,settings,np.eye(4),arm_frames)
            exported=export_robot_demonstration(replay,output/'robot_demonstration.npz')
            with np.load(exported,allow_pickle=False) as data:export_data={key:data[key].copy() for key in data.files if key!='table_top_z'}
            # Do not export the reference repo's simulated table height as a measured plane.
            export_data.update(coordinate_frame=np.array('UR controller base'),
                               capture_sha256=np.array(hashlib.sha256((folder/'capture.json').read_bytes()).hexdigest()),
                               execution_validated=np.array(False))
            np.savez_compressed(exported,**export_data)
            report.update(arm_ik_computed=True,preview_type='DG5F pose from retargeted UR5e arm FK in calibrated base',
                          maximum_arm_position_error_m=float(arm_data['arm_position_error_m'].max()),
                          maximum_arm_rotation_error_rad=float(arm_data['arm_rotation_error_rad'].max()),
                          arm_method='Upstream ArmStage: sequential palm IK, captured seed, per-frame URDF velocity limits',
                          replay_export='robot_demonstration.npz',replay_export_rate_hz=60.)
        np.savez_compressed(output/'retargeted_hand.npz',**arm_data,hand_q_rad=qs,joint_names=np.asarray(HAND_JOINT_NAMES),
            mano_joints_camera_m=camera_joints,mano_joints_robot_base_m=base_joints,
            robot_points_base_m=points,robot_point_names=np.asarray(names),palm_pose_robot_base=poses,
            robot_base_from_camera=base_from_camera,fps=fps,time_s=np.arange(len(qs))/fps,
            fingertip_error_m=errors,valid=np.array([False,True]))
        preview={'coordinate_frame':'robot_base','position_units':'metres','fps':fps,'joint_names':list(HAND_JOINT_NAMES),
            'robot_point_names':names,'robot_chains':[[0,*range(1+f*5,6+f*5)] for f in range(5)],
            'mano_chains':[[0,13,14,15,20],[0,1,2,3,16],[0,4,5,6,17],[0,10,11,12,19],[0,7,8,9,18]],
            'frames':[{'robot_points':p.tolist(),'mano_points':h.tolist(),'hand_q_rad':q.tolist(),
                       'mean_fingertip_error_m':float(e.mean())} for p,h,q,e in zip(points,base_joints,qs,errors)]}
        if arm_frames:
            for frame,result in zip(preview['frames'],arm_frames):
                frame.update(arm_q_rad=result.arm_q.tolist(),arm_position_error_m=result.palm_position_error,
                             arm_rotation_error_rad=result.palm_rotation_error)
        (output/'retarget_preview.json').write_text(json.dumps(preview,allow_nan=False))
        report.update(status='completed',frames=len(qs),fps=fps,robot_base_from_camera=base_from_camera.tolist(),
            calibration=capture.get('calibration'),camera_roundtrip_max_error_m=error,
            mean_scaled_fingertip_error_m=float(errors.mean()),max_scaled_fingertip_error_m=float(errors.max()),
            max_finger_penetration_m=float(max(penetrations)),method='vector / own optimizer; per-finger scaling; no trajectory filter',
            palm_alignment='Captured robot palm follows generated palm motion with a fixed anatomical offset determined from the source capture.',
            human_from_robot_palm=human_from_robot_palm.tolist(),
            urdf_sha256=hashlib.sha256(ROBOT_URDF.read_bytes()).hexdigest(),
            input_sha256={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in ('capture.json','future_hands.npz')},
            limitations='Kinematic retargeting only. The 60 Hz export is format-compatible with replay_real.py, not cleared for physical execution. Prepare and check robot rollout to validate table clearance and motion limits.')
        progress('completed',len(qs),len(qs))
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');progress('failed');raise
    finally:
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();retarget(args.input_dir,args.out)


if __name__=='__main__':main()
