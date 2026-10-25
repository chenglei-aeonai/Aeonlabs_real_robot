"""Offline calibrated hand/contact replay; the cube is free, never attached to the hand."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from .placement import apply
from .retarget_scene import ROOT,scene_geometry,RetargetScene
from .h3_actions import rigid_transforms


def cube_position(capture,geometry,pixel,edge):
    camera=capture['camera'];u,v=np.asarray(pixel,dtype=float)
    if not np.isfinite([u,v,edge]).all() or not 0<=u<camera['width'] or not 0<=v<camera['height'] or not .005<=edge<=.2:
        raise ValueError('Select a cube center inside the first frame and an edge length between 5 and 200 mm.')
    k=np.asarray(camera['undistorted_intrinsics']['matrix'],float)
    base_camera=np.linalg.inv(rigid_transforms(capture['camera_from_robot_base'],1)[0])
    ray=base_camera[:3,:3]@np.linalg.solve(k,[u,v,1.]);origin=base_camera[:3,3]
    normal=np.asarray(geometry['table_plane']['normal']);offset=geometry['table_plane']['offset_m']
    denominator=float(normal@ray)
    if abs(denominator)<1e-8:raise ValueError('Selected camera ray is parallel to the table.')
    distance=(edge-offset-normal@origin)/denominator
    if distance<=0:raise ValueError('Selected point intersects the table behind the camera.')
    return origin+distance*ray-normal*edge/2


def build_scene(source,output,config):
    import mujoco
    from .robot import write_hand_urdf,HAND_JOINT_NAMES
    if not .001<=config['mass_kg']<=2 or not .05<=config['friction']<=3:
        raise ValueError('Cube mass or friction is outside the supported range.')
    capture=json.loads((source/'capture.json').read_text());geometry=scene_geometry(source)
    center=cube_position(capture,geometry,config['pixel'],config['edge_m'])
    assets=output/'assets';assets.mkdir()
    urdf=write_hand_urdf(assets);tree=ET.parse(urdf);root=tree.getroot()
    for link in root.findall('link'):
        for visual in link.findall('visual'):link.remove(visual)
    for mesh in root.findall('.//mesh'):
        path=Path(mesh.get('filename'));target=assets/path.name;shutil.copyfile(path,target);mesh.set('filename',str(target.resolve()))
    ET.SubElement(ET.SubElement(root,'mujoco'),'compiler',strippath='false',fusestatic='false',discardvisual='true')
    tree.write(urdf)
    spec=mujoco.MjSpec.from_file(str(urdf));spec.modelname='calibrated_dg5f_cube_replay';spec.compiler.discardvisual=False
    spec.option.timestep=1/480;spec.option.integrator=mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone=mujoco.mjtCone.mjCONE_ELLIPTIC;spec.option.iterations=100
    spec.worldbody.first_body().mocap=True
    for body in spec.bodies:
        if body.name!='world':body.gravcomp=1.
    for joint in spec.joints:joint.armature=.002
    for geom in spec.geoms:
        geom.friction=[config['friction'],.01,.001];geom.condim=4
        geom.rgba=[.75,.78,.8,1.];geom.solref=[.012,1.4]
    base_world=np.asarray(geometry['robot_base_from_world'])
    table=spec.worldbody.add_geom();table.name='table';table.type=mujoco.mjtGeom.mjGEOM_PLANE
    table.pos=apply(base_world,np.asarray(geometry['table_display']['center_world_m']))
    table.quat=RetargetScene.quaternion(base_world[:3,:3]);table.size=[.5,.35,.01];table.rgba=[.32,.37,.33,1.]
    table.friction=[config['friction'],.01,.001]
    # The printed pattern is visual context; the table plane handles contact.
    board=spec.worldbody.add_geom();board.name='board';board.type=mujoco.mjtGeom.mjGEOM_BOX
    w,h=geometry['board_size_m'];base_board=np.asarray(geometry['robot_base_from_board'])
    board.pos=apply(base_board,np.array([w/2,h/2,-.0005]));board.quat=RetargetScene.quaternion(base_board[:3,:3]@np.diag([1.,-1.,-1.]))
    board.size=[w/2,h/2,.0005];board.contype=0;board.conaffinity=0;board.rgba=[1,1,1,1]
    cube=spec.worldbody.add_body();cube.name='cube';cube.pos=center;yaw=np.deg2rad(config.get('yaw_deg',0.));c,s=np.cos(yaw),np.sin(yaw)
    cube.quat=RetargetScene.quaternion(base_world[:3,:3]@np.array([[c,-s,0],[s,c,0],[0,0,1.]]))
    cube.add_freejoint(name='cube_free')
    geom=cube.add_geom();geom.name='cube_geom';geom.type=mujoco.mjtGeom.mjGEOM_BOX
    geom.size=np.full(3,config['edge_m']/2);geom.mass=config['mass_kg'];geom.rgba=[.95,.2,.5,1.]
    geom.friction=[config['friction'],.01,.001];geom.condim=4
    # Gains follow the existing sibling physics_sim model; contact parameters remain estimates.
    kp=.116*np.array([42.9718,400,42.9718,42.9718]+[42.9718]*16)
    kd=np.array([.1,.9475,.3012,.1821,.7523,.4126,.2856,.1365,.7587,.4126,.2856,.1365,.7274,.4126,.2856,.1365,.2662,.4796,.3012,.1821])
    limits={j.get('name'):j.find('limit') for j in root.findall('joint') if j.find('limit') is not None}
    for index,name in enumerate(HAND_JOINT_NAMES):
        limit=limits[name];a=spec.add_actuator();a.name=name+'_position';a.target=name;a.trntype=mujoco.mjtTrn.mjTRN_JOINT
        a.ctrllimited=True;a.ctrlrange=[float(limit.get('lower')),float(limit.get('upper'))]
        a.forcelimited=True;effort=float(limit.get('effort'));a.forcerange=[-effort,effort]
        a.gaintype=mujoco.mjtGain.mjGAIN_FIXED;a.biastype=mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm[0]=kp[index];a.biasprm[1]=-kp[index];a.biasprm[2]=-kd[index]
    camera=spec.worldbody.add_camera();camera.name='calibrated_camera'
    base_camera=np.linalg.inv(np.asarray(capture['camera_from_robot_base']));camera.pos=base_camera[:3,3]
    camera.quat=RetargetScene.quaternion(base_camera[:3,:3]@np.diag([1.,-1.,-1.]))
    width,height=capture['camera']['width'],capture['camera']['height'];k=np.asarray(capture['camera']['undistorted_intrinsics']['matrix'])
    camera.resolution=[width,height];camera.sensor_size=[1,1];camera.focal_pixel=[k[0,0],k[1,1]]
    # Pixel-array coordinates use integer pixel centers; MuJoCo's raster centers are half-integers.
    camera.principal_pixel=[width/2-k[0,2]-.5,height/2-k[1,2]-.5]
    light=spec.worldbody.add_light();light.pos=center+[0,0,1.5];light.dir=[0,0,-1]
    light.type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    spec.visual.global_.offwidth=width;spec.visual.global_.offheight=height
    spec.compile();xml=ET.fromstring(spec.to_xml());asset=xml.find('asset')
    texture=assets/'board.png';shutil.copyfile(ROOT/'calibration/boards/charuco_board.png',texture)
    ET.SubElement(asset,'texture',name='board_pattern',type='2d',file=str(texture.resolve()))
    ET.SubElement(asset,'material',name='board_material',texture='board_pattern',texuniform='false',texrepeat='1 1')
    xml.find(".//geom[@name='board']").set('material','board_material')
    text=ET.tostring(xml,encoding='unicode');model=mujoco.MjModel.from_xml_string(text)
    # Package paths relative to the scene XML, including the collision meshes.
    (output/'mujoco_scene.xml').write_text(text.replace(str(output.resolve())+'/', ''))
    return model,capture,geometry,center


def simulate(source,output,config):
    os.environ.setdefault('MUJOCO_GL','egl')
    import mujoco
    import imageio_ffmpeg
    from scipy.spatial.transform import Rotation,Slerp
    from .robot import HAND_JOINT_NAMES
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','mode':'mujoco_replay','hardware_executed':False,'execution_ready':False,
            'cube_parameters':config,'cube_parameters_measured':False,'cube_pixel_reference':'center of top face; cube assumed resting flat on table',
            'palm_motion':'prescribed mocap trajectory; no arm dynamics or IK',
            'scene_scope':'table plane, calibration board, DG5F and target cube; other objects are not reconstructed',
            'table_collision':'infinite plane; table footprint is not measured',
            'cube_dynamics':'free body with frictional contacts; no weld, attachment, or forced cube motion'}
    renderer=None;writers=[]
    def progress(stage,step=0,total=0):
        temp=output/'progress.json.tmp';temp.write_text(json.dumps({'stage':stage,'step':step,'total':total,'updated_at':time.time()}));temp.replace(output/'progress.json')
    try:
        progress('build_scene')
        model,capture,geometry,center=build_scene(source,output,config);data=mujoco.MjData(model)
        with np.load(source/'retargeted_hand.npz',allow_pickle=False) as trajectory:
            targets=trajectory['hand_q_rad'].copy();poses=trajectory['palm_pose_robot_base'].copy();fps=float(trajectory['fps'])
        indices=np.asarray([model.joint(name).qposadr[0] for name in HAND_JOINT_NAMES])
        controls=np.asarray([model.actuator(name+'_position').id for name in HAND_JOINT_NAMES])
        time_grid=np.arange(len(targets))/fps;slerp=Slerp(time_grid,Rotation.from_matrix(poses[:,:3,:3]))
        def command(t):
            t=float(np.clip(t,0,time_grid[-1]));fraction=t*fps;i=min(int(fraction),len(targets)-2);alpha=np.clip(fraction-i,0,1)
            data.ctrl[controls]=(1-alpha)*targets[i]+alpha*targets[i+1]
            data.mocap_pos[0]=(1-alpha)*poses[i,:3,3]+alpha*poses[i+1,:3,3]
            data.mocap_quat[0]=np.roll(slerp(t).as_quat(),1)
        data.qpos[indices]=targets[0];command(0);mujoco.mj_forward(model,data)
        for _ in range(240):mujoco.mj_step(model,data)
        start=data.time
        width,height=capture['camera']['width'],capture['camera']['height'];renderer=mujoco.Renderer(model,height,width)
        overview=mujoco.MjvCamera();overview.lookat=(center+poses[0,:3,3])/2;overview.distance=.95;overview.azimuth=135;overview.elevation=-35
        for name in ('mujoco_camera.mp4','mujoco_overview.mp4'):
            writer=imageio_ffmpeg.write_frames(str(output/name),(width,height),fps=fps);writer.send(None);writers.append(writer)
        cube_id=model.body('cube').id;cube_geom=model.geom('cube_geom').id;table_id=model.geom('table').id
        normal=np.asarray(geometry['table_plane']['normal']);offset=geometry['table_plane']['offset_m'];initial_height=float(data.xpos[cube_id]@normal+offset)
        cube_poses=[];angles=[];contacts=[];clearances=[];tracking=[];gaps=[];max_penetration=0.
        hand_geoms=[i for i in range(model.ngeom) if model.body(model.geom_bodyid[i]).name.startswith('rl_dg_')]
        for frame,t in enumerate(time_grid):
            while data.time-start<t-1e-9:
                command(data.time-start);mujoco.mj_step(model,data)
            command(t);mujoco.mj_forward(model,data)
            if not np.isfinite(data.qpos).all() or data.warning.number.sum():raise ValueError('MuJoCo reported an unstable simulation; see model and configuration.')
            hand_contact=False;table_contact=False
            for contact in data.contact:
                ids={int(contact.geom1),int(contact.geom2)}
                if cube_geom in ids:
                    table_contact|=table_id in ids
                    hand_contact|=any(model.body(model.geom_bodyid[g]).name.startswith('rl_dg_') for g in ids)
                max_penetration=max(max_penetration,float(max(0,-contact.dist)))
            gaps.append(min(mujoco.mj_geomDistance(model,data,g,cube_geom,2.,None) for g in hand_geoms))
            cube_poses.append(np.r_[data.xpos[cube_id],data.xquat[cube_id]].copy());angles.append(data.qpos[indices].copy())
            vertical_extent=config['edge_m']/2*np.abs(normal@data.xmat[cube_id].reshape(3,3)).sum()
            clearances.append(float(normal@data.xpos[cube_id]+offset-vertical_extent))
            contacts.append([hand_contact,table_contact]);tracking.append(data.qpos[indices]-targets[frame])
            for writer,camera in zip(writers,('calibrated_camera',overview)):
                renderer.update_scene(data,camera=camera);writer.send(renderer.render())
            progress('simulate',frame+1,len(targets))
        cube_poses=np.asarray(cube_poses);contacts=np.asarray(contacts);tracking=np.asarray(tracking)
        lifted=(np.asarray(clearances)>.02)&contacts[:,0]&~contacts[:,1]
        longest=run=0
        for value in lifted:run=run+1 if value else 0;longest=max(longest,run)
        np.savez_compressed(output/'mujoco_motion.npz',hand_q_rad=np.asarray(angles),target_hand_q_rad=targets,
            joint_names=np.asarray(HAND_JOINT_NAMES),coordinate_frame='robot_base',position_units='metres',angle_units='radians',
            camera_matrix=np.asarray(capture['camera']['undistorted_intrinsics']['matrix']),
            camera_from_robot_base=np.asarray(capture['camera_from_robot_base']),
            palm_pose_robot_base=poses,cube_pose_robot_base_xyz_wxyz=cube_poses,hand_cube_contact=contacts[:,0],
            cube_table_contact=contacts[:,1],cube_bottom_clearance_m=np.asarray(clearances),hand_cube_surface_gap_m=np.asarray(gaps),fps=fps,time_s=time_grid)
        report.update(status='completed',frames=len(targets),fps=fps,geometry=geometry,cube_initial_center_robot_base_m=center.tolist(),
            max_cube_lift_m=float(np.max(cube_poses[:,:3]@normal+offset)-initial_height),
            hand_contact_frames=int(contacts[:,0].sum()),minimum_hand_cube_surface_gap_m=float(min(gaps)),lift_contact_duration_s=longest/fps,
            lift_observed=bool(longest/fps>=.25),lift_definition='Cube bottom >20 mm above table, with hand contact and no table contact for >=0.25 s.',
            joint_tracking_rms_rad=float(np.sqrt(np.mean(tracking**2))),max_contact_penetration_m=max_penetration,
            limitations='Approximate cube geometry, mass and friction; prescribed palm motion; collision meshes use MuJoCo convex approximations. This does not validate a real-arm grasp.',mujoco_version=mujoco.__version__)
        (output/'cube_config.json').write_text(json.dumps(config,indent=2)+'\n')
        with zipfile.ZipFile(output/'mujoco_scene.zip','w',zipfile.ZIP_DEFLATED) as archive:
            archive.write(output/'mujoco_scene.xml','mujoco_scene.xml')
            for path in (output/'assets').iterdir():
                if path.suffix.lower() in ('.stl','.png','.obj'):archive.write(path,'assets/'+path.name)
        progress('completed',len(targets),len(targets))
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}');progress('failed');raise
    finally:
        for writer in writers:writer.close()
        if renderer is not None:renderer.close()
        (output/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();simulate(args.source,args.out,json.loads(args.config.read_text()))


if __name__=='__main__':main()
