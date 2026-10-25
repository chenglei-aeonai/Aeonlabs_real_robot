import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hand_retarget.mujoco_replay import cube_position,build_scene
from hand_retarget.retarget_scene import scene_geometry


@pytest.fixture
def source(tmp_path):
    folder=tmp_path/'source';folder.mkdir()
    calibration=tmp_path/'world_from_base.npy';np.save(calibration,np.eye(4))
    camera=np.array([[1.,0,0,0],[0,-1,0,0],[0,0,-1,1],[0,0,0,1]])
    capture={'camera_from_robot_base':camera.tolist(),'camera':{'width':640,'height':480,'undistorted_intrinsics':{'matrix':[[550,0,310],[0,500,230],[0,0,1]],'dist_coeffs':[0]*5}},
             'calibration':{'world_from_base':{'path':str(calibration),'sha256':hashlib.sha256(calibration.read_bytes()).hexdigest()}}}
    (folder/'capture.json').write_text(json.dumps(capture))
    return folder


def test_cube_top_face_reprojects_to_selected_pixel(source):
    capture=json.loads((source/'capture.json').read_text());geometry=scene_geometry(source)
    center=cube_position(capture,geometry,[350,250],.04)
    normal=np.array(geometry['table_plane']['normal'])
    assert center@normal+geometry['table_plane']['offset_m']==pytest.approx(.02)
    point=center+normal*.02;transform=np.array(capture['camera_from_robot_base'])
    camera=transform[:3,:3]@point+transform[:3,3]
    uv=np.array(capture['camera']['undistorted_intrinsics']['matrix'])@camera
    np.testing.assert_allclose(uv[:2]/uv[2],[350,250])
    with pytest.raises(ValueError):cube_position(capture,geometry,[-1,200],.04)


def test_cube_is_free_and_falls_onto_calibrated_table(source,tmp_path):
    import mujoco
    out=tmp_path/'out';out.mkdir()
    model,_,geometry,center=build_scene(source,out,{'pixel':[350,250],'edge_m':.04,'mass_kg':.05,'friction':.8})
    assert model.nu==20 and model.nmocap==1 and model.neq==0
    joint=model.joint('cube_free');assert model.jnt_type[joint.id]==mujoco.mjtJoint.mjJNT_FREE
    data=mujoco.MjData(model);data.mocap_pos[0]=[2,2,2]
    address=joint.qposadr[0];data.qpos[address+2]+=.1
    for _ in range(480):mujoco.mj_step(model,data)
    mujoco.mj_forward(model,data)
    assert data.xpos[model.body('cube').id,2]==pytest.approx(.02,abs=.001)
    assert np.isfinite(data.qpos).all()
    portable=mujoco.MjModel.from_xml_path(str(out/'mujoco_scene.xml'))
    assert portable.nq==model.nq


def test_rendered_camera_matches_captured_intrinsics(source,tmp_path):
    import mujoco
    out=tmp_path/'out';out.mkdir()
    model,capture,_,_=build_scene(source,out,{'pixel':[350,250],'edge_m':.04,'mass_kg':.05,'friction':.8})
    # Replace the cube with a small sphere at a known base-space point to test
    # MuJoCo's actual rendering, including the optical/OpenGL axis conversion.
    geom=model.geom('cube_geom').id;model.geom_type[geom]=mujoco.mjtGeom.mjGEOM_SPHERE;model.geom_size[geom]=[.005,0,0]
    data=mujoco.MjData(model);data.mocap_pos[0]=[2,2,2]
    point=np.array([.05,.04,.3]);a=model.joint('cube_free').qposadr[0];data.qpos[a:a+3]=point
    mujoco.mj_forward(model,data)
    renderer=mujoco.Renderer(model,480,640)
    try:
        renderer.enable_segmentation_rendering();renderer.update_scene(data,camera='calibrated_camera');pixels=renderer.render()
    finally:renderer.close()
    y,x=np.where(pixels[:,:,0]==geom);assert len(x)>5
    transform=np.array(capture['camera_from_robot_base']);p=transform[:3,:3]@point+transform[:3,3]
    uv=np.array(capture['camera']['undistorted_intrinsics']['matrix'])@p
    np.testing.assert_allclose([x.mean(),y.mean()],uv[:2]/uv[2],atol=.65)
