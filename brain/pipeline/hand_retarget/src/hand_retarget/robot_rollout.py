"""Prepare calibrated arm IK and supervise a confirmed, watchdog-protected replay."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator, CubicHermiteSpline, BPoly
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT/'calibration'))
from surface_guard import RobotSurfaceModel
from .retarget_scene import scene_geometry
from .action_timing import continuous_curve,retime_continuous,check_continuous,derivative_bounds,simplify_actions
from .robot import HAND_LOWER, HAND_UPPER, HAND_JOINT_NAMES
from . import native_timing
from . import adaptive_timing

ARM_LIMIT = np.array([2*np.pi, 2*np.pi, np.pi, 2*np.pi, 2*np.pi, 2*np.pi])
CLEARANCE = .015


class SafetyStop(RuntimeError):
    """A measured safety violation, distinct from a software exception."""


def check_finger_velocity(velocity,limit=.7):
    if velocity.shape!=(20,) or not np.isfinite(velocity).all():
        raise SafetyStop("Measured finger velocity is missing or invalid.")
    index=int(np.argmax(np.abs(velocity)))
    if abs(velocity[index])>limit:
        raise SafetyStop(
            f"Finger overspeed: {HAND_JOINT_NAMES[index]} measured {velocity[index]:.3f} rad/s "
            f"(limit {limit:.3f} rad/s). Inspect hand feedback/controller response before retrying; "
            "slower planned targets alone do not guarantee safe measured speed.")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_path(q, hand, times,speed_limits=(.15,.35)):
    if (q.ndim != 2 or q.shape[1] != 6 or hand.shape != (len(q),20)
            or times.shape != (len(q),) or len(q)<2
            or not all(np.isfinite(x).all() for x in (q,hand,times))
            or abs(times[0])>1e-9 or np.any(np.diff(times)<=0)):
        raise ValueError('Invalid rollout dimensions, values, or timestamps.')
    if np.any(np.abs(q)>ARM_LIMIT) or np.any(hand<HAND_LOWER-1e-6) or np.any(hand>HAND_UPPER+1e-6):
        raise ValueError('Rollout exceeds arm or finger joint limits.')
    for values, limit in zip((q,hand),speed_limits or ()):
        if np.max(np.abs(np.diff(values,axis=0))/np.diff(times)[:,None])>limit+1e-6:
            raise ValueError('Rollout exceeds supervised joint speed limit.')


def smooth_fingers(hands):
    """Convex temporal filter; retain the demonstrated start/end grasp."""
    result=gaussian_filter1d(hands.astype(float),sigma=1.5,axis=0,mode='nearest')
    result[[0,-1]]=hands[[0,-1]]
    return result


def smooth_curve(times,values):
    slopes=PchipInterpolator(times,values).derivative()(times)
    slopes[[0,-1]]=0  # stationary handover and finish
    return CubicHermiteSpline(times,values,slopes)


def interval_limits(curve):
    """Exact per-interval maximum speed/acceleration across joints."""
    a,b,c,_=curve.c
    dt=np.diff(curve.x)[:,None]
    vertex=np.divide(-b,3*a,out=np.zeros_like(a),where=a!=0)
    vertex=np.clip(vertex,0,dt)
    velocity=np.max([np.max(np.abs(3*a*x*x+2*b*x+c),axis=1) for x in (0,dt,vertex)],axis=0)
    acceleration=np.max([np.max(np.abs(6*a*x+2*b),axis=1) for x in (0,dt)],axis=0)
    return velocity,acceleration


def curve_limits(curve):
    return tuple(float(np.max(x)) for x in interval_limits(curve))


def check_curve(q,h,t,speed_scale=1.):
    check_path(q,h,t)
    curves=[smooth_curve(t,values) for values in (q,h)]
    for curve,vmax,amax in zip(curves,(.15,.35),(.3,.7)):
        v,a=curve_limits(curve)
        if v>vmax*speed_scale+1e-6 or a>amax*speed_scale**2+1e-6:
            raise ValueError('Smoothed trajectory exceeds supervised speed/acceleration limits.')
    return curves


# Conservative rest-to-rest segments: no position overshoot, continuous velocity
# and acceleration. Jerk is bounded (not necessarily continuous at waypoints).
JERK_LIMITS=(.6,1.4)  # arm, fingers; rad/s^3 at 100% replay speed


def action_curve(times,values):
    zero=np.zeros(values.shape[1])
    return BPoly.from_derivatives(times,[[value,zero,zero] for value in values])


def action_limits(values,times):
    delta=np.abs(np.diff(values,axis=0));dt=np.diff(times)[:,None]
    return tuple(float(np.max(k*delta/dt**power)) for k,power in
                 ((1.875,1),(10/np.sqrt(3),2),(60.,3)))


def retime_actions(q,h,times,speed_scale):
    durations=np.diff(times).copy()
    for values,vmax,amax,jmax in zip((q,h),(.15,.35),(.3,.7),JERK_LIMITS):
        delta=np.max(np.abs(np.diff(values,axis=0)),axis=1)
        durations=np.maximum(durations,np.maximum.reduce([
            1.875*delta/vmax,np.sqrt((10/np.sqrt(3))*delta/amax),
            np.cbrt(60*delta/jmax)]))
    return np.r_[0,np.cumsum(durations)]/speed_scale


def check_actions(q,h,times,speed_scale):
    check_path(q,h,times)
    for values,vmax,amax,jmax in zip((q,h),(.15,.35),(.3,.7),JERK_LIMITS):
        actual=action_limits(values,times)
        limits=(vmax*speed_scale,amax*speed_scale**2,jmax*speed_scale**3)
        if any(a>b+1e-6 for a,b in zip(actual,limits)):
            raise ValueError('Post-processed action exceeds speed, acceleration or jerk limits.')
    return [action_curve(times,values) for values in (q,h)]


def rebase_actions(q,h,t,current_q,current_h,arm_only=False):
    """Discard the capture handover and anchor retargeted deltas to live joints."""
    if len(t)<3:
        raise ValueError('Live-relative rollout needs a captured pose and at least two motion frames.')
    shifted_q=q[1:]-q[1]+current_q
    shifted_h=np.repeat(current_h[None,:],len(h)-1,axis=0) if arm_only else h[1:]-h[1]+current_h
    shifted_t=t[1:]-t[1]
    for values,lower,upper,names in ((shifted_q,-ARM_LIMIT,ARM_LIMIT,[f'arm_joint_{j+1}' for j in range(6)]),
                                    (shifted_h,HAND_LOWER,HAND_UPPER,HAND_JOINT_NAMES)):
        # Never clip the measured first target: that would introduce a start jump.
        invalid=np.flatnonzero((values[0]<lower-1e-6)|(values[0]>upper+1e-6))
        if len(invalid):
            joint=invalid[0]
            raise SafetyStop(f'Measured starting joint {names[joint]} is outside joint limits; '
                             'cannot preserve this starting pose. No commands sent.')
        projected=np.clip(values[1:],lower,upper)
        correction=np.max(np.abs(projected-values[1:]),axis=0)
        for joint in np.flatnonzero(correction>1e-9):
            print(f'Constrained relative action: {names[joint]}, maximum correction '
                  f'{correction[joint]:.6f} rad; first measured target preserved.',flush=True)
        values[1:]=projected
        # Projection on a fixed joint interval cannot enlarge segment deltas.
        # The quintic interpolation is rebuilt below and all limits rechecked.
    check_path(shifted_q,shifted_h,shifted_t)
    return shifted_q,shifted_h,shifted_t


def rebase_cartesian_actions(model,q,h,t,current_q,current_h,speed_scale,arm_only=False,tick=lambda:None,continuous=False,nominal_times=None,native=False,timing_report=None,arm_acceleration=3.,adaptive=False):
    """Translate the palm path in robot base; preserve world-relative rotations."""
    if len(t)<3:raise ValueError('Cartesian replay needs at least two motion frames.')
    source=np.array([model.transforms(a,b)['rl_dg_palm'] for a,b in zip(q[1:],h[1:])])
    smoothing_sigma=.1/float(np.median(np.diff(nominal_times[1:]))) if native and nominal_times is not None else 2.
    if continuous:
        original=source.copy()
        source[:,:3,3]=(native_timing.endpoint_filter(source[:,:3,3],smoothing_sigma) if native else
                         gaussian_filter1d(source[:,:3,3],2.,axis=0,mode='nearest'))
        relative=Rotation.from_matrix(original[:,:3,:3]@original[0,:3,:3].T).as_rotvec()
        relative=(native_timing.endpoint_filter(relative,smoothing_sigma) if native else
                  gaussian_filter1d(relative,2.,axis=0,mode='nearest'))
        source[:,:3,:3]=Rotation.from_rotvec(relative).as_matrix()@original[0,:3,:3]
        source[[0,-1]]=original[[0,-1]]
        print(f'Cartesian jitter filter: maximum position adjustment '
              f'{np.max(np.linalg.norm(source[:,:3,3]-original[:,:3,3],axis=1))*1000:.2f} mm; endpoints preserved.',flush=True)
    current=model.transforms(current_q,current_h)['rl_dg_palm']
    translation=current[:3,3]-source[0,:3,3]
    if np.linalg.norm(translation)>.3:
        raise SafetyStop('Cartesian start translation exceeds 30 cm; reposition or retarget before replay.')
    desired=source.copy()
    desired[:,:3,3]+=translation
    desired[:,:3,:3]=source[:,:3,:3]@source[0,:3,:3].T@current[:3,:3]
    if np.max(np.linalg.norm(desired[:,:3,3]-current[:3,3],axis=1))>.3:
        raise SafetyStop('Cartesian path leaves the 30 cm neighbourhood of its live start.')
    hands=np.repeat(current_h[None,:],len(source),axis=0) if arm_only else h[1:]-h[1]+current_h
    if np.any(current_h<HAND_LOWER-1e-6) or np.any(current_h>HAND_UPPER+1e-6):
        raise SafetyStop('Measured starting hand is outside joint limits.')
    if continuous and not arm_only:
        endpoints=hands[[0,-1]].copy()
        hands=(native_timing.endpoint_filter(hands,smoothing_sigma) if native else
               gaussian_filter1d(hands,2.,axis=0,mode='nearest'))
        hands[[0,-1]]=endpoints
    hands[1:]=np.clip(hands[1:],HAND_LOWER,HAND_UPPER)
    solved=[current_q.copy()];seed=current_q.copy();max_error=0.
    for i,target in enumerate(desired[1:],1):
        tick()
        def residual(angles):
            actual=model.transforms(angles,hands[i])['rl_dg_palm']
            return np.r_[actual[:3,3]-target[:3,3],.15*Rotation.from_matrix(target[:3,:3].T@actual[:3,:3]).as_rotvec()]
        result=least_squares(residual,seed,bounds=(-ARM_LIMIT+1e-6,ARM_LIMIT-1e-6),max_nfev=100,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        error=residual(result.x);max_error=max(max_error,float(np.linalg.norm(error[:3])))
        if np.linalg.norm(error[:3])>.001 or np.linalg.norm(error[3:])/.15>np.deg2rad(1):
            raise SafetyStop(f'Cartesian translated path is unreachable at frame {i}; no commands sent.')
        if np.max(np.abs(result.x-seed))>.12:
            raise SafetyStop(f'Cartesian IK branch jump at frame {i}; no commands sent.')
        solved.append(result.x.copy());seed=result.x
    solved=np.asarray(solved)
    if continuous:
        baseline=(nominal_times[1:]-nominal_times[1]) if nominal_times is not None else (t[1:]-t[1])*speed_scale
        if native:
            if adaptive:
                times,selected=adaptive_timing.select_timing(solved,hands,baseline,speed_scale,arm_acceleration,tick)
            else:
                times,selected=native_timing.select_timing(solved,hands,baseline,speed_scale,arm_acceleration)
            if timing_report is not None:timing_report.update(selected)
            for candidate in selected['candidates']:
                result='PASS' if candidate['passed'] else '; '.join(candidate['violations'])
                label='Average source rate (local timing)' if adaptive else 'Source speed'
                print(f"{label} {candidate['speed_scale']:g}x: {candidate['duration_s']:.3f} s — {result}",flush=True)
        else:
            keep=simplify_actions(solved,hands,baseline)
            print(f'Continuous motion: {len(keep)} of {len(solved)} waypoints retained; '
                  'simplification tolerance 0.003 rad arm / 0.010 rad fingers.',flush=True)
            solved=solved[keep];hands=hands[keep];baseline=baseline[keep]
            times=retime_continuous(solved,hands,baseline,speed_scale)
    else:
        times=retime_actions(solved,hands,(t[1:]-t[1])*speed_scale,speed_scale)
    if times[-1]>600:raise SafetyStop('Cartesian replay exceeds ten minutes after limiting.')
    print(f'Cartesian translation [m]: {translation.tolist()}; first pose equals live pose; '
          f'maximum waypoint IK error {max_error*1000:.3f} mm.',flush=True)
    return solved,hands,times


def validate_rebased_actions(model,q,h,t,normal,offset,speed_scale,tick=lambda:None,continuous=False,native=False,arm_acceleration=3.):
    check_path(q,h,t,(np.pi,np.pi) if native else (.15,.35))
    curves=(native_timing.checked_curves(q,h,t,ARM_LIMIT,HAND_LOWER,HAND_UPPER,arm_acceleration) if native else
            check_continuous(q,h,t,speed_scale) if continuous else check_actions(q,h,t,speed_scale))
    native_speed=max(np.max(native_timing.peaks(curve)['speed']) for curve in curves) if native else 0.
    lowest=float('inf')
    for i in range(len(t)-1):
        tick()
        delta=max(np.max(np.abs(q[i+1]-q[i])),np.max(np.abs(h[i+1]-h[i])))
        if native:delta=max(delta,native_speed*(t[i+1]-t[i]))
        count=max(11,int(np.ceil(2*delta/.005))+1)
        for at in np.linspace(t[i],t[i+1],count):
            clearance=surface_clearance(model,curves[0](at),curves[1](at),normal,offset)
            lowest=min(lowest,clearance)
            if clearance<CLEARANCE:
                raise SafetyStop(f'Shifted rollout violates table clearance at segment {i}: {clearance*1000:.1f} mm. No commands sent.')
    return curves,lowest


def checked_lead_in(model,a,b,q,h,normal,offset,speed_scale):
    """Bounded live-pose handover, at rest at both ends; never automatic homing."""
    if np.max(np.abs(a-q))>.08 or np.max(np.abs(b-h))>.2:
        raise RuntimeError('Live starting pose moved outside the allowed neighbourhood.')
    delta=np.r_[q-a,h-b]
    duration=max(2.,float(np.max(1.875*np.abs(delta)/np.r_[np.full(6,.075),np.full(20,.175)]))/speed_scale,
                 float(np.sqrt(np.max(6*np.abs(delta)/np.r_[np.full(6,.15),np.full(20,.35)])))/speed_scale,
                 float(np.cbrt(np.max(60*np.abs(delta)/np.r_[np.full(6,JERK_LIMITS[0]),np.full(20,JERK_LIMITS[1])])))/speed_scale)
    for u in np.linspace(0,1,max(21,int(np.ceil(np.max(np.abs(delta))/.002))+1)):
        w=u**3*(10-15*u+6*u*u)
        if surface_clearance(model,a+(q-a)*w,b+(h-b)*w,normal,offset)<CLEARANCE:
            raise RuntimeError('Live start transition violates table clearance; no motion authorized.')
    return duration


def surface_clearance(model,q,h,normal,offset):
    transforms=model.transforms(q,h)
    return min(float(np.min((vertices@transforms[name][:3,:3].T+transforms[name][:3,3])@normal+offset))
               for name,vertices in model.mesh_links if name in model.moving_links)


def prepare(source, output, speed_scale=.5,timing_policy='legacy'):
    if timing_policy not in ('legacy','native','adaptive'):raise ValueError('Unknown timing policy.')
    adaptive=timing_policy=='adaptive'
    native=timing_policy in ('native','adaptive')
    if native and speed_scale<native_timing.MIN_SPEED:raise ValueError('Native source speed must be at least 0.25x.')
    if not np.isfinite(speed_scale) or not .1<=speed_scale<=1:
        raise ValueError("Replay speed must be between 10% and 100% of the supervised baseline.")
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','mode':'robot_rollout','hardware_executed':False,'execution_ready':False,
            'motion_source':'retargeted_hand.npz','mujoco_motion_used':False}
    def progress(stage,step=0,total=0):
        temporary=output/'progress.json.tmp'
        temporary.write_text(json.dumps({'stage':stage,'step':step,'total':total,'updated_at':time.time()}))
        temporary.replace(output/'progress.json')
    try:
        capture=json.loads((source/'capture.json').read_text())
        geometry=scene_geometry(source)
        model=RobotSurfaceModel()
        start=np.asarray(capture['robot_state']['arm_q_rad'],float)
        start_hand=np.asarray(capture['robot_state']['hand_q_rad'],float)
        if start.shape!=(6,) or start_hand.shape!=(20,) or not np.isfinite(np.r_[start,start_hand]).all():
            raise ValueError('Capture must contain measured arm and finger joint states.')
        with np.load(source/'retargeted_hand.npz',allow_pickle=False) as data:
            poses=data['palm_pose_robot_base'].copy();hands=data['hand_q_rad'].copy();times=data['time_s'].copy()
            upstream_arm=data['arm_q_rad'].copy() if 'arm_q_rad' in data else None
            if data['joint_names'].tolist()!=list(HAND_JOINT_NAMES):raise ValueError('Unexpected finger order.')
        from .h3_actions import rigid_transforms
        poses=rigid_transforms(poses,len(poses))
        if hands.shape!=(len(poses),20) or times.shape!=(len(poses),) or len(poses)<2 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
            raise ValueError('Invalid retarget trajectory.')
        if not np.isfinite(hands).all() or np.any(hands<HAND_LOWER-1e-6) or np.any(hands>HAND_UPPER+1e-6):
            raise ValueError('Retargeted fingers contain invalid values or exceed joint limits.')
        frames=model.transforms(start,start_hand)
        # Both models must agree with the original camera-space mount, in UR base.
        expected=np.linalg.inv(np.asarray(capture['camera_from_robot_base']))@np.asarray(capture['camera_from_hand_mount'])
        if not np.allclose(frames['rl_dg_mount'],expected,atol=.002):
            raise ValueError('Capture hand mount differs from collision/IK model; measure and resolve mount before execution.')
        if upstream_arm is not None and (upstream_arm.shape!=(len(poses),6) or not np.isfinite(upstream_arm).all() or np.any(np.abs(upstream_arm)>ARM_LIMIT)):
            raise ValueError('Invalid retargeted arm joint sequence.')
        report['arm_motion_source']='Step 04 upstream ArmStage' if upstream_arm is not None else 'Preparation palm IK (historical hand-only result)'
        qs=[];errors=[];seed=start.copy()
        for i,(target,hand) in enumerate(zip(poses,hands)):
            progress('arm_ik',i,len(poses))
            def residual(q):
                pose=model.transforms(q,hand)['rl_dg_palm']
                return np.r_[pose[:3,3]-target[:3,3], .15*Rotation.from_matrix(target[:3,:3].T@pose[:3,:3]).as_rotvec()]
            if upstream_arm is None:
                result=least_squares(residual,seed,bounds=(-ARM_LIMIT+1e-6,ARM_LIMIT-1e-6),max_nfev=100,ftol=1e-9,xtol=1e-9,gtol=1e-9)
                solved=result.x
            else:
                solved=upstream_arm[i]
            error=residual(solved)
            if np.linalg.norm(error[:3])>.003 or np.linalg.norm(error[3:])/.15>np.deg2rad(2):
                raise ValueError(f'Arm IK cannot reach frame {i} within 3 mm / 2 degrees.')
            if np.max(np.abs(solved-seed))>.12:
                raise ValueError(f'Arm IK branch jump at frame {i}; refusing discontinuous motion.')
            qs.append(solved);errors.append(np.linalg.norm(error[:3]));seed=solved
        filtered_hands=native_timing.endpoint_filter(hands,1.5) if native else smooth_fingers(hands)
        if native:filtered_hands=np.clip(filtered_hands,HAND_LOWER,HAND_UPPER)
        q=np.vstack([start,qs]);hand=np.vstack([start_hand,filtered_hands]);times=np.r_[0,2+(times-times[0])*(1 if native else 4)]
        check_path(q,hand,times,None)
        nominal_times=times.copy()
        normal=np.asarray(geometry['table_plane']['normal']);offset=geometry['table_plane']['offset_m']
        timing={}
        arm_acceleration=min(3.,float(json.loads((ROOT/'arm/pc_ur_new.json').read_text())['joint_acceleration']))
        if native:
            preview_q,preview_h,preview_t=rebase_cartesian_actions(
                model,q,hand,times,start,start_hand,speed_scale,continuous=True,nominal_times=nominal_times,
                native=True,timing_report=timing,arm_acceleration=arm_acceleration,adaptive=adaptive)
            arm_curve,hand_curve=native_timing.checked_curves(preview_q,preview_h,preview_t,
                ARM_LIMIT,HAND_LOWER,HAND_UPPER,arm_acceleration)
            checked_q,checked_h,checked_t=preview_q,preview_h,preview_t
            report['native_timing']=timing
        else:
            times=retime_continuous(q,hand,times,speed_scale)
            if times[-1]>600:raise ValueError('Supervised rollout exceeds ten minutes after speed limiting.')
            check_path(q,hand,times)
            arm_curve,hand_curve=check_continuous(q,hand,times,speed_scale)
            checked_q,checked_h,checked_t=q,hand,times
        lowest=float('inf');checked=0
        native_speed=max(np.max(native_timing.peaks(curve)['speed']) for curve in (arm_curve,hand_curve)) if native else 0.
        for i in range(len(checked_q)-1):
            delta=max(np.max(np.abs(checked_q[i+1]-checked_q[i])),np.max(np.abs(checked_h[i+1]-checked_h[i])))
            if native:delta=max(delta,native_speed*(checked_t[i+1]-checked_t[i]))
            count=max(11,int(np.ceil(2*delta/.005))+1)
            for weight in np.linspace(0,1,count):
                sample_time=checked_t[i]+weight*(checked_t[i+1]-checked_t[i])
                clearance=surface_clearance(model,arm_curve(sample_time),hand_curve(sample_time),normal,offset)
                lowest=min(lowest,clearance);checked+=1
                if clearance<CLEARANCE:raise ValueError(f'Table clearance {clearance*1000:.1f} mm at segment {i}; requires {CLEARANCE*1000:.0f} mm. No rollout authorized.')
            progress('table_surface_check',i,len(checked_q)-1)
        interpolation=adaptive_timing.INTERPOLATION if adaptive else native_timing.INTERPOLATION if native else 'flow_quintic_v3'
        np.savez_compressed(output/'robot_rollout.npz',arm_q=q,hand_q=hand,timestamp=times,nominal_timestamp=nominal_times,interpolation=np.array(interpolation),start_policy=np.array('live_cartesian_v2'),plane_normal=normal,plane_offset=offset)
        if native:
            np.savez_compressed(output/'native_preview.npz',arm_q=preview_q,hand_q=preview_h,timestamp=preview_t,
                                source_timestamp=nominal_times[1:]-nominal_times[1])
        (output/'capture.json').write_bytes((source/'capture.json').read_bytes())
        derivatives=lambda curve:([float(np.max(native_timing.peaks(curve)[key])) for key in ('speed','acceleration','jerk')]
                                  if native else [float(np.max(b)) for b in derivative_bounds(curve)])
        report.update(status='completed',execution_ready=True,duration_s=float(checked_t[-1]),frames=len(checked_q),
                      speed_scale=float(speed_scale),arm_speed_limit_rad_s=np.pi if native else .15*speed_scale,hand_speed_limit_rad_s=np.pi if native else .35*speed_scale,
                      initial_transition_duration_s=0. if native else float(times[1]),
                      interpolation=interpolation,start_policy='live_cartesian_v2',
                      live_relative_duration_s=float(checked_t[-1]) if native else float(times[-1]-times[1]),
                      action_postprocessing=('Endpoint-preserving filters, C2 arm cubic / monotone finger cubic, source timeline with uniform 1x/0.5x/0.25x selection' if native else
                                            'Gaussian finger filter, synchronized continuous-acceleration quintic segments, velocity/acceleration/jerk retiming'),
                      arm_peak_command_derivatives=derivatives(arm_curve),
                      hand_peak_command_derivatives=derivatives(hand_curve),
                      arm_jerk_limit_rad_s3=None if native else JERK_LIMITS[0]*speed_scale**3,
                      hand_jerk_limit_rad_s3=None if native else JERK_LIMITS[1]*speed_scale**3,
                      finger_smoothing_sigma_frames=1.5,
                      maximum_finger_smoothing_change_rad=float(np.max(np.abs(filtered_hands-hands))),
                      minimum_table_clearance_m=lowest,checked_configurations=checked,maximum_ik_position_error_m=float(max(errors)),
                      rollout_sha256=digest(output/'robot_rollout.npz'),collision_model_sha256=model.sha256,
                      source_sha256={name:digest(source/name) for name in ('capture.json','retargeted_hand.npz')},
                      calibration=capture['calibration'],checks='Joint limits, IK continuity, continuous-curve speed/acceleration/jerk and sampled URDF table clearance.',
                      limitations='Supervised replay only. Self-collision, other objects, cables, and calibration uncertainty are not certified. Inspect the entire path and clear the workspace. No automatic homing.')
        if native:
            report.update(source_duration_s=timing['source_duration_s'],effective_speed_scale=timing['effective_speed_scale'],
                          timing_policy='source_rate_demo',arm_acceleration_limit_rad_s2=arm_acceleration,
                          checks='Joint limits, IK continuity, exact cubic speed/arm-acceleration extrema and sampled URDF table clearance. Demo profile has no finger-acceleration or jerk hard limit.')
        if adaptive:
            report.update(timing_policy='local_time_allocation',
                          action_postprocessing='Preserved filtered waypoints; locally allocated segment times; eight-second budget including bounded fine settling and stop margin.',
                          checks='Joint limits, IK, cubic extrema, added finger speed/acceleration planning limits and sampled URDF table clearance.')
        progress('completed')
    except Exception as error:
        if isinstance(error,native_timing.NativeTimingError):report['native_timing']=error.report
        report.update(status='failed',error=f'{type(error).__name__}: {error}');progress('failed');raise
    finally:
        temporary=output/'manifest.json.tmp'
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False))
        temporary.replace(output/'manifest.json')
    return report


def transport_module():
    spec=importlib.util.spec_from_file_location('rollout_transport',ROOT/'replay_real.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class BrowserLease:
    """Parent pipe closes on server death; browser heartbeats must reach the runner."""
    def __init__(self,fd):
        self.fd=fd;self.last=time.monotonic();self.buffer=b''
        os.set_blocking(fd,False)

    def check(self):
        try:
            data=os.read(self.fd,4096)
            if not data:raise RuntimeError('Web server disconnected; stopping physical replay.')
            self.buffer+=data
            while b'\n' in self.buffer:
                line,self.buffer=self.buffer.split(b'\n',1)
                if line==b'HEARTBEAT':self.last=time.monotonic()
        except BlockingIOError:pass
        if time.monotonic()-self.last>3:raise RuntimeError('Browser heartbeat expired; stopping physical replay.')


def check_return_start(report, arm_q, hand_q):
    """A planned return is only valid near the recorded endpoint; never rebase it elsewhere."""
    expected = report.get('return_start')
    if expected is None:
        return
    q = np.asarray(expected['arm_q'], float)
    h = np.asarray(expected['hand_q'], float)
    if (q.shape != (6,) or h.shape != (20,) or not np.isfinite(np.r_[q, h]).all()
            or np.max(np.abs(arm_q - q)) > .03
            or np.max(np.abs(hand_q - h)) > .08):
        raise SafetyStop('Robot moved away from the recorded return start; no commands sent.')


def replay(folder,send=False,web_session=False,arm_only=False,speed=1.):
    if not np.isfinite(speed) or not 0<speed<=1:
        raise ValueError("Replay speed must be in (0, 1].")
    import zmq
    transport=transport_module()
    report=json.loads((folder/'manifest.json').read_text())
    config=json.loads((ROOT/'arm/pc_ur_new.json').read_text())
    arm_acceleration=min(3.,float(config['joint_acceleration']))
    if not report.get('execution_ready') or digest(folder/'robot_rollout.npz')!=report['rollout_sha256']:
        raise ValueError('Missing or modified validated rollout.')
    model=RobotSurfaceModel()
    if model.sha256!=report['collision_model_sha256']:raise ValueError('Collision model changed; prepare again.')
    with np.load(folder/'robot_rollout.npz',allow_pickle=False) as data:
        q=data['arm_q'].copy();h=data['hand_q'].copy();t=data['timestamp'].copy();normal=data['plane_normal'].copy();offset=float(data['plane_offset'])
        interpolation=str(data['interpolation']) if 'interpolation' in data else 'linear'
        nominal_times=data['nominal_timestamp'].copy() if 'nominal_timestamp' in data else None
        start_policy=str(data['start_policy']) if 'start_policy' in data else 'captured_absolute'
    adaptive=interpolation==adaptive_timing.INTERPOLATION
    native=interpolation in (native_timing.INTERPOLATION,adaptive_timing.INTERPOLATION)
    fixed_pose=interpolation=='fixed_pose_quintic_v1'
    if fixed_pose and start_policy!='captured_absolute':raise ValueError('Fixed pose return requires an absolute endpoint.')
    timing={}
    if start_policy not in ('captured_absolute','live_relative_v1','live_cartesian_v2'):raise ValueError('Unknown start policy.')
    if start_policy in ('live_relative_v1','live_cartesian_v2') and interpolation not in ('rest_quintic_v2','flow_quintic_v3',native_timing.INTERPOLATION,adaptive_timing.INTERPOLATION):raise ValueError('Live-relative motion requires checked interpolation.')
    if native and (start_policy!='live_cartesian_v2' or nominal_times is None):raise ValueError('Native replay requires its source timestamps and Cartesian start policy.')
    t/=speed
    run_scale=report.get('speed_scale',1.)*speed
    if arm_only and start_policy not in ('live_relative_v1','live_cartesian_v2'):
        raise ValueError('Prepare a new live-relative rollout before arm-only replay.')
    check_path(q,h,t,None if native else (.15,.35))
    if interpolation not in ('linear','smooth_pchip_v1','rest_quintic_v2','fixed_pose_quintic_v1','flow_quintic_v3',native_timing.INTERPOLATION,adaptive_timing.INTERPOLATION):raise ValueError('Unknown rollout interpolation.')
    curves=None
    if interpolation=='smooth_pchip_v1':curves=check_curve(q,h,t,run_scale)
    if interpolation in ('rest_quintic_v2','fixed_pose_quintic_v1'):curves=check_actions(q,h,t,run_scale)
    continuous=interpolation=='flow_quintic_v3' or native
    if continuous and not native:curves=check_continuous(q,h,t,run_scale)
    feedback=[]
    class Arm(transport.Arm):
        def poll(self):
            while True:
                try:state=self.state.recv_json(flags=zmq.NOBLOCK)
                except zmq.Again:return
                self.diagnostic_state=state
                age=time.time()-state.get('timestamp_ms',0)/1000
                flags=('is_truncated','protective_stopped','emergency_stopped')
                if (not -.05<=age<=.25 or any(type(state.get(k)) is not bool for k in flags)
                        or any(state[k] for k in flags)):
                    raise RuntimeError('Stale or faulted arm state.')
                if (state.get('rtde_connected') is not True
                        or not 0<=state.get('robot_state_age_s',float('inf'))<=.25):
                    raise RuntimeError('Missing or stale RTDE sample age; rebuild and restart the arm controller.')
                if not 0<state.get('joint_target_watchdog_s',0)<=.25:
                    raise RuntimeError('Controller must advertise a joint-target watchdog <= 0.25 s. Rebuild and restart the normal arm controller.')
                if isinstance(state.get('calibration_guard'),dict):raise RuntimeError('Stop calibration and use the normal arm controller.')
                angles=np.asarray(state.get('Q'),float)
                if angles.shape!=(6,) or not np.isfinite(angles).all():raise ValueError('Invalid arm state.')
                self.q=angles;self.q_at=time.monotonic();self.qd=np.asarray(state.get('Qd',[]),float)
                velocity_limit=np.pi if native or fixed_pose else .3
                if armed and (self.qd.shape!=(6,) or not np.isfinite(self.qd).all() or np.max(np.abs(self.qd))>velocity_limit):
                    raise RuntimeError(f'Measured arm velocity is missing or exceeds {velocity_limit:.3f} rad/s.')
    class Hand(transport.Hand):
        def poll(self):
            while True:
                try:raw,_=self.state.recvfrom(65535)
                except BlockingIOError:return
                state=json.loads(raw)
                if state.get('type')!='hand_state':continue
                self.diagnostic_state=state
                if not 0<=state.get('measured_age_s',float('inf'))<=.25:raise RuntimeError('Stale measured finger state.')
                angles=np.asarray(state.get('positions'),float)
                if angles.shape!=(20,) or not np.isfinite(angles).all():raise ValueError('Invalid finger state.')
                self.q=angles;self.q_at=time.monotonic()
                velocity=np.asarray(state.get('velocities',[]),float)
                if armed:
                    # Save the offending packet too: the command-loop record is reached
                    # only after fresh() returns, which previously lost the fault evidence.
                    currents=np.asarray(state.get('currents_ma',[]),float)
                    feedback.append(np.r_[time.monotonic(),angles,
                        velocity if velocity.shape==(20,) else np.full(20,np.nan),
                        currents if currents.shape==(20,) else np.full(20,np.nan)])
                    check_finger_velocity(velocity,np.pi if native or fixed_pose else .7)
    arm=hand=None;armed=False;records=[]
    command_records=[]
    failure=None;loop_context=None;resolved_path=None;completion=None;final_target_sent=False
    sent_counts={'arm':0,'hand':0};last_sent={'arm':None,'hand':None}
    def abort_signal(*_):raise KeyboardInterrupt('Operator stopped replay')
    signal.signal(signal.SIGTERM,abort_signal)
    try:
        arm=Arm(config['socket_port'],config['publisher_port']);hand=Hand(5562,5563)
        arm.wait();hand.wait()
        def check(a,b,target_a,target_b):
            arm_error=np.abs(a-target_a);finger_error=np.abs(b-target_b)
            if np.max(arm_error)>.08:
                raise SafetyStop(f'Arm tracking error {np.max(arm_error):.3f} rad exceeds 0.080 rad at joint {np.argmax(arm_error)+1}.')
            if np.max(finger_error)>.2:
                index=int(np.argmax(finger_error))
                raise SafetyStop(f'Finger tracking error {finger_error[index]:.3f} rad at {HAND_JOINT_NAMES[index]} exceeds 0.200 rad; inspect hand response before retrying.')
            if surface_clearance(model,a,b,normal,offset)<CLEARANCE:raise RuntimeError('Measured robot is too close to the calibrated table.')
        a,b=arm.fresh(),hand.fresh()
        check_return_start(report,a,b)
        check(a,b,a if start_policy=='live_cartesian_v2' else q[0],b if arm_only or start_policy=='live_cartesian_v2' else h[0])
        print('Execution mode: ARM ONLY (no finger commands).' if arm_only else 'Execution mode: ARM + HAND.',flush=True)
        if native:
            print(f"Native source: {report['source_duration_s']:.3f} s; requested {run_scale:g}x. "
                  f"Demo measured-speed limit {np.pi:.3f} rad/s; arm acceleration budget {arm_acceleration:.3f} rad/s^2. No commands sent.",flush=True)
        else:print(f'Validated {len(t)} targets, {t[-1]:.1f} seconds. No commands sent.',flush=True)
        if fixed_pose:print('Fixed absolute return; conservative quintic planning, approved demo measured-speed profile (pi rad/s).',flush=True)
        if not send:
            if start_policy in ('live_relative_v1','live_cartesian_v2'):
                a,b=arm.fresh(),hand.fresh()
                if start_policy=='live_cartesian_v2':
                    shifted_q,shifted_h,shifted_t=rebase_cartesian_actions(model,q,h,t,a,b,run_scale,arm_only,lambda:(arm.fresh(),hand.fresh()),continuous,nominal_times,native,timing,arm_acceleration,adaptive)
                else:
                    shifted_q,shifted_h,shifted_t=rebase_actions(q,h,t,a,b,arm_only)
                validate_rebased_actions(model,shifted_q,shifted_h,shifted_t,normal,offset,run_scale,lambda:(arm.fresh(),hand.fresh()),continuous,native,arm_acceleration)
                print(f'Live-relative path checked: {shifted_t[-1]:.1f} seconds. First target equals measured pose. No commands sent.',flush=True)
            return
        print('Clear the complete path, stop competing controllers, and keep the e-stop accessible. This checks the table, not all scene obstacles.',flush=True)
        if input('Type SEND to execute on the physical robot: ').strip()!='SEND':return
        lease=BrowserLease(sys.stdin.fileno()) if web_session else None
        if lease:lease.check()
        initial_a,initial_b=arm.fresh(),hand.fresh()
        check_return_start(report,initial_a,initial_b)
        check(initial_a,initial_b,initial_a if start_policy=='live_cartesian_v2' else q[0],initial_b if arm_only or start_policy=='live_cartesian_v2' else h[0])
        if start_policy in ('live_relative_v1','live_cartesian_v2'):
            def tick():
                if lease:lease.check()
                arm.fresh();hand.fresh()
            if start_policy=='live_cartesian_v2':
                q,h,t=rebase_cartesian_actions(model,q,h,t,initial_a,initial_b,run_scale,arm_only,tick,continuous,nominal_times,native,timing,arm_acceleration,adaptive)
            else:
                q,h,t=rebase_actions(q,h,t,initial_a,initial_b,arm_only)
            print('Checking the complete path shifted to the live starting pose; no commands sent.',flush=True)
            curves,clearance=validate_rebased_actions(model,q,h,t,normal,offset,run_scale,tick,continuous,native,arm_acceleration)
            lead=0.
            resolved_path=folder/f'resolved_rollout_{time.time_ns()}.npz'
            np.savez_compressed(resolved_path,
                                arm_q=q,hand_q=h,timestamp=t,interpolation=np.array(interpolation),
                                plane_normal=normal,plane_offset=offset,
                                minimum_table_clearance_m=clearance,
                                source_rollout_sha256=np.array(report['rollout_sha256']),
                                arm_only=np.array(arm_only),replay_speed=np.array(speed),
                                native_timing=np.array(json.dumps(timing)))
            print(f'Live-relative rollout: first target equals measured pose; {t[-1]:.1f} seconds, minimum sampled clearance {clearance*1000:.1f} mm.',flush=True)
        else:
            lead=checked_lead_in(model,initial_a,initial_b,q[0],h[0],normal,offset,run_scale)
        # Validation may take time: reject a changed live pose before sending targets.
        if lease:lease.check()
        if max(np.max(np.abs(arm.fresh()-initial_a)),np.max(np.abs(hand.fresh()-initial_b)))>.005:
            raise RuntimeError('Robot moved during start validation; retry from a stationary pose.')
        if lead:print(f'Smoothed live-pose handover: {lead:.1f} seconds.',flush=True)
        endpoint_time=t[-1]+lead
        armed=True;start=time.monotonic();last_progress=-1
        print('Physical replay started.',flush=True)
        while True:
            if lease:lease.check()
            elapsed=0. if not records else time.monotonic()-start
            if adaptive and elapsed>adaptive_timing.EXECUTION_BUDGET:
                raise SafetyStop('Eight-second execution budget exceeded; stopping and recording incomplete timing.')
            path_time=np.clip(elapsed-lead,0,t[-1])
            if elapsed<lead:
                u=np.clip(elapsed/lead,0,1);w=u**3*(10-15*u+6*u*u)
                target_a=initial_a+(q[0]-initial_a)*w
                target_b=initial_b+(h[0]-initial_b)*w
            elif curves:
                target_a,target_b=(curve(path_time) for curve in curves)
            else:
                target_a=np.array([np.interp(path_time,t,q[:,j]) for j in range(6)])
                target_b=np.array([np.interp(path_time,t,h[:,j]) for j in range(20)])
            loop_context={'elapsed_s':float(elapsed),'path_time_s':float(path_time),
                          'phase':'feedback_check',
                          'segment_index':None if elapsed<lead else int(np.clip(np.searchsorted(t,path_time,side='right')-1,0,len(t)-2)),
                          'candidate_arm_target':target_a.tolist(),
                          'candidate_hand_target':None if arm_only else target_b.tolist()}
            a,b=arm.fresh(),hand.fresh();check(a,b,target_a,target_b)
            if arm_only and np.max(np.abs(b-initial_b))>.03:
                raise SafetyStop('Hand moved during arm-only replay; collision geometry changed.')
            if int(elapsed)!=last_progress:
                last_progress=int(elapsed)
                moved=float(np.rad2deg(np.max(np.abs(a-initial_a))))
                tracking=float(np.rad2deg(np.max(np.abs(a-target_a))))
                print(f'Replaying {elapsed:.1f} / {t[-1]+lead:.1f} seconds | '
                      f'measured arm change {moved:.2f} deg | tracking error {tracking:.2f} deg',flush=True)
            if elapsed>=endpoint_time:
                if not (native or fixed_pose):break
                arm_residual=float(np.max(np.abs(a-target_a)))
                hand_residual=None if arm_only else float(np.max(np.abs(b-target_b)))
                completion={'targets_delivered':final_target_sent,'arm_residual_rad':arm_residual,
                            'hand_residual_rad':hand_residual,
                            'within_settle_tolerance':arm_residual<.01 and (arm_only or hand_residual<.05)}
                if final_target_sent and elapsed>=endpoint_time+.25 and completion['within_settle_tolerance']:break
                if final_target_sent and elapsed>endpoint_time+(adaptive_timing.settle_time(t[-1]) if adaptive else 2.):
                    # The existing 0.08/0.2 tracking guards above still stop unsafe
                    # deviations. Fine settling is an accuracy result, not a new fault.
                    print(f'Trajectory targets completed; residual arm error {arm_residual:.4f} rad, '
                          f'hand error {hand_residual if hand_residual is not None else 0.:.4f} rad. '
                          'Fine settling tolerance was not reached; saving the residual.',flush=True)
                    break
            command_a=(native_timing.servo_target(target_a,curves[0].derivative()(path_time),a,
                                                 float(config['path_tracking_p_gain']),ARM_LIMIT) if native else target_a)
            loop_context['phase']='send_arm'
            arm.command.send_json({'target_q':command_a.tolist()})
            sent_counts['arm']+=1;last_sent['arm']=command_a.tolist()
            command_records.append(np.r_[elapsed,command_a])
            if not arm_only:
                loop_context['phase']='send_hand'
                hand.send(target_b)
                sent_counts['hand']+=1;last_sent['hand']=target_b.tolist()
            if (native or fixed_pose) and path_time>=t[-1]:final_target_sent=True
            loop_context['phase']='waiting'
            records.append(np.r_[elapsed,a,b,target_a,target_b]);time.sleep(1/60)
        print('Replay finished; stopping arm.' if arm_only else 'Replay finished; stopping arm and holding measured fingers.',flush=True)
    except BaseException as error:
        if armed:
            failure={'exception_type':type(error).__name__,'message':str(error),
                     'layer':'application_safety_check' if isinstance(error,SafetyStop) else 'runtime_or_communication',
                     'arm_state':getattr(arm,'diagnostic_state',None),
                     'hand_state':getattr(hand,'diagnostic_state',None)}
        raise
    finally:
        if armed:
            confirmed=False
            try:
                arm.stop()
                deadline=time.monotonic()+1.5
                confirmed=False
                while time.monotonic()<deadline:
                    arm.fresh()
                    velocity=getattr(arm,'qd',np.array([]))
                    if velocity.shape==(6,) and np.isfinite(velocity).all() and np.max(np.abs(velocity))<.01:
                        confirmed=True;break
                    time.sleep(.02)
                print('Robot arm stop confirmed.' if confirmed else 'STOP UNCONFIRMED: use physical e-stop.',flush=True)
            except Exception as error:print(f'Arm stop failed; use e-stop: {error}',file=sys.stderr,flush=True)
            try:
                if not arm_only and hand.q is not None:hand.send(hand.q)
            except Exception as error:print(f'Finger hold failed: {error}',file=sys.stderr)
            log_path=folder/f'rollout_log_{time.time_ns()}.npz'
            np.savez_compressed(log_path,samples=np.asarray(records),arm_command_targets=np.asarray(command_records).reshape(-1,7),
                                hand_feedback=np.asarray(feedback).reshape(-1,61),
                                hand_feedback_columns=np.array(['monotonic_s']+
                                    [f'position_rad:{n}' for n in HAND_JOINT_NAMES]+
                                    [f'velocity_rad_s:{n}' for n in HAND_JOINT_NAMES]+
                                    [f'current_ma:{n}' for n in HAND_JOINT_NAMES]))
            # Persist after requesting stop. "Sent" means a local transport send,
            # never an acknowledgement that the hardware executed that target.
            outcome=('stopped_with_error' if failure else 'trajectory_finished_with_residual_error'
                     if completion and not completion['within_settle_tolerance'] else 'trajectory_finished')
            context={'outcome':outcome,'completion':completion,
                     'failure':failure,'loop':loop_context,'sent_counts':sent_counts,
                     'last_sent_targets':last_sent,'arm_stop_confirmed':confirmed,
                     'arm_only':arm_only,'start_policy':start_policy,
                     'native_timing':timing,
                     'resolved_rollout':str(resolved_path) if resolved_path else None,
                     'trajectory_log':str(log_path),
                     'note':'Targets in a feedback_check phase have not been sent. Counts exclude final stop/hold. Transport send is not hardware acknowledgement.'}
            if failure and loop_context:
                print(f"Stopped during {loop_context['phase']} at path {loop_context['path_time_s']:.3f} s; "
                      f"segment {loop_context['segment_index']}; targets sent: {sent_counts}.",flush=True)
            try:
                log_path.with_suffix('.json').write_text(json.dumps(context,indent=2)+'\n')
                print(f'Execution context: {log_path.with_suffix(".json")}',flush=True)
            except OSError as error:
                print(f'Could not save execution context: {error}',file=sys.stderr,flush=True)
        if hand is not None:hand.close()
        if arm is not None:arm.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    plan=commands.add_parser('prepare');plan.add_argument('--source',type=Path,required=True);plan.add_argument('--out',type=Path,required=True);plan.add_argument('--speed-scale',type=float,default=1.)
    plan.add_argument('--timing-policy',choices=['adaptive','native','legacy'],default='adaptive')
    run=commands.add_parser('replay');run.add_argument('--folder',type=Path,required=True);run.add_argument('--send',action='store_true');run.add_argument('--web-session',action='store_true',help=argparse.SUPPRESS)
    run.add_argument('--arm-only',action='store_true',help='Send arm commands only; measured hand feedback is required for surface checks.')
    run.add_argument('--speed',type=float,default=1.,help='Requested source-speed multiplier for native rollouts; additional time scaling for historical rollouts.')
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.source,args.out,args.speed_scale,args.timing_policy)
    else:
        try:replay(args.folder,args.send,args.web_session,args.arm_only,args.speed)
        except KeyboardInterrupt:
            print('Operator stopped replay.',flush=True)
            raise SystemExit(130)
        except SafetyStop as error:
            print(f'SAFETY STOP: {error}',file=sys.stderr,flush=True)
            raise SystemExit(2)

if __name__=='__main__':main()
