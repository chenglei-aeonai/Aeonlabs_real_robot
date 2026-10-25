"""Render a saved retarget through Simone's full-arm physics simulator, offline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .mujoco_replay import cube_position
from .retarget_scene import ROOT, scene_geometry


def transform_points(transform, points):
    return np.asarray(points) @ transform[:3, :3].T + transform[:3, 3]


def table_pose(world_from_base, geometry):
    world_from_table = world_from_base @ np.asarray(geometry['robot_base_from_world'])
    center = transform_points(world_from_table, geometry['table_display']['center_world_m'])
    return world_from_table, center


def quaternion(matrix):
    return np.roll(Rotation.from_matrix(matrix).as_quat(), 1)


def build_scene(source, config, q0, upstream, temporary):
    import mujoco
    capture = json.loads((source/'capture.json').read_text())
    geometry = scene_geometry(source)
    reference = upstream.build_scene(with_ghost=False, rest_poses=(q0,))
    upstream.set_joints(reference, q0)
    base = reference.model.body('base').id
    world_from_base = np.eye(4)
    world_from_base[:3, :3] = reference.data.xmat[base].reshape(3, 3)
    world_from_base[:3, 3] = reference.data.xpos[base]
    world_from_table, center = table_pose(world_from_base, geometry)
    urdf = upstream._mujoco_urdf(upstream.DEFAULT_ASSETS, temporary)
    spec = mujoco.MjSpec.from_file(str(urdf))
    spec.compiler.discardvisual = False
    upstream._add_world(spec, upstream.DEFAULT_TABLE_TOP_Z)
    upstream._add_actuators(spec, urdf)
    for pair in reference.excluded_hand_pairs:
        exclude = spec.add_exclude()
        exclude.bodyname1, exclude.bodyname2 = sorted(pair)
    table = spec.body('table')
    table.quat = quaternion(world_from_table[:3, :3])
    table.pos = center - world_from_table[:3, 2]*.025
    spec.geom('table_geom').size = [.5, .35, .025]
    board_pose = world_from_base @ np.asarray(geometry['robot_base_from_board'])
    w, h = geometry['board_size_m']
    board = spec.worldbody.add_geom()
    board.name = 'calibration_board'
    board.type = mujoco.mjtGeom.mjGEOM_BOX
    board.pos = transform_points(board_pose, [w/2, h/2, -.0005])
    board.quat = quaternion(board_pose[:3, :3] @ np.diag([1., -1., -1.]))
    board.size = [w/2, h/2, .0005]
    board.contype = board.conaffinity = 0
    body = spec.worldbody.add_body()
    body.name = 'preview_cube'
    body.pos = transform_points(world_from_base, cube_position(capture, geometry, config['pixel'], config['edge_m']))
    body.quat = quaternion(world_from_table[:3, :3] @ Rotation.from_euler('z', config['yaw_deg'], degrees=True).as_matrix())
    body.add_freejoint(name='preview_cube_free')
    cube = body.add_geom()
    cube.name = 'preview_cube_geom'
    cube.type = mujoco.mjtGeom.mjGEOM_BOX
    cube.size = np.full(3, config['edge_m']/2)
    cube.mass = config['mass_kg']
    cube.rgba = [.95, .25, .5, 1.]
    cube.friction = [config['friction'], .01, .001]
    for body in spec.bodies:
        if body.name not in ('world', 'table', 'preview_cube'):
            body.gravcomp = 1.
    camera_pose = world_from_base @ np.linalg.inv(np.asarray(capture['camera_from_robot_base']))
    camera = spec.worldbody.add_camera()
    camera.name = 'captured_camera'
    camera.pos = camera_pose[:3, 3]
    camera.quat = quaternion(camera_pose[:3, :3] @ np.diag([1., -1., -1.]))
    width, height = capture['camera']['width'], capture['camera']['height']
    k = np.asarray(capture['camera']['undistorted_intrinsics']['matrix'])
    camera.resolution = [width, height]
    camera.sensor_size = [1., 1.]
    camera.focal_pixel = [k[0, 0], k[1, 1]]
    camera.principal_pixel = [width/2-k[0, 2]-.5, height/2-k[1, 2]-.5]
    spec.visual.global_.offwidth = width
    spec.visual.global_.offheight = height
    # Add the actual board texture before compiling the final scene.
    spec.compile()
    xml = ET.fromstring(spec.to_xml())
    asset = xml.find('asset')
    ET.SubElement(asset, 'texture', name='board_texture', type='2d',
                  file=str(ROOT/'calibration/boards/charuco_board.png'))
    ET.SubElement(asset, 'material', name='board_material', texture='board_texture', texuniform='false')
    xml.find(".//geom[@name='calibration_board']").set('material', 'board_material')
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding='unicode'))
    scene = upstream._configure(model, upstream.DEFAULT_TABLE_TOP_Z, False, upstream.Physics(),
                                reference.excluded_hand_pairs, None)
    cube_geom = model.geom('preview_cube_geom').id
    model.geom_contype[cube_geom] = upstream.TUBE_BIT
    model.geom_conaffinity[cube_geom] = upstream.TABLE_BIT | upstream.HAND_BIT | upstream.ARM_BIT
    model.geom_condim[cube_geom] = 4
    model.geom_solref[cube_geom] = scene.physics.solref
    model.geom_solimp[cube_geom] = scene.physics.solimp
    scene.tube_bodies = {model.body('preview_cube').id}
    return scene, capture, geometry, world_from_base


def preview(source, output, physics_source, config_path):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import imageio_ffmpeg
    from PIL import Image
    sys.path.insert(0, str(physics_source.resolve()))
    from physics_sim import scene as upstream, replay as playback
    config = json.loads(config_path.read_text())
    if (not .005 <= config['edge_m'] <= .2 or not .001 <= config['mass_kg'] <= 2
            or not .05 <= config['friction'] <= 3):
        raise ValueError('Invalid estimated object parameters.')
    paired_capture = config_path.parent/'capture.json'
    if paired_capture.exists() and paired_capture.read_bytes() != (source/'capture.json').read_bytes():
        raise ValueError('Object estimate belongs to a different capture.')
    demo = playback.RobotDemonstration.load(source/'robot_demonstration.npz')
    if not np.isfinite(demo.q).all() or demo.q.shape[1] != 26:
        raise ValueError('Invalid arm/hand trajectory.')
    if not np.allclose(np.diff(demo.timestamp), 1/60, atol=1e-6):
        raise ValueError('Upstream physics replay requires 60 Hz samples.')
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='full-arm-physics-') as directory:
        scene, capture, geometry, world_from_base = build_scene(source, config, demo.q[0], upstream, Path(directory))
        width, height = capture['camera']['width'], capture['camera']['height']
        renderer = mujoco.Renderer(scene.model, height, width)
        overview = mujoco.MjvCamera()
        bounds = []
        for index in np.linspace(0, len(demo.q)-1, 12).astype(int):
            upstream.set_joints(scene, demo.q[index])
            bounds.extend(scene.data.xpos[1:].copy())
        table_id = scene.model.body('table').id
        table_rotation = scene.data.xmat[table_id].reshape(3, 3)
        corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.35, .35) for z in (-.025, .025)])
        bounds.extend(corners @ table_rotation.T + scene.data.xpos[table_id])
        bounds = np.asarray(bounds)
        overview.lookat = (bounds.min(axis=0)+bounds.max(axis=0))/2
        overview.distance = float(np.linalg.norm(bounds.max(axis=0)-bounds.min(axis=0))*1.55)
        overview.azimuth, overview.elevation = 125., -25.
        writers = []
        for name in ('overview.mp4', 'camera.mp4'):
            writer = imageio_ffmpeg.write_frames(str(output/name), (width, height), fps=30)
            writer.send(None)
            writers.append(writer)
        objects = []
        class Recorder:
            step = -int(round(playback.SETTLE_SECONDS/scene.physics.control_dt))
            def is_running(self): return True
            def sync(self):
                if self.step >= 0:
                    cube = scene.model.body('preview_cube').id
                    objects.append(np.r_[scene.data.xpos[cube], scene.data.xquat[cube]])
                    if self.step % 2 == 0:
                        for view, writer, name in zip((overview, 'captured_camera'), writers, ('overview', 'camera')):
                            renderer.update_scene(scene.data, camera=view)
                            frame = renderer.render()
                            writer.send(frame)
                            if self.step == 0:
                                Image.fromarray(frame).save(output/f'{name}_first.png')
                            if self.step >= len(demo.q)-2:
                                Image.fromarray(frame).save(output/f'{name}_last.png')
                    if self.step % 60 == 0:
                        print(f'Simulating/rendering frame {self.step+1}/{len(demo.q)}', flush=True)
                self.step += 1
        try:
            log = playback.replay(scene, demo, viewer=Recorder(), realtime=False)
        finally:
            for writer in writers: writer.close()
            renderer.close()
        text = playback.report(log).replace('tube', 'object')
        contacts = {category.replace('tube', 'object'): dict(pairs) for category, pairs in log.contacts.items()}
        hand_object = contacts.get('hand-object', {})
        text += '\nHand-object contacts: ' + ('; '.join(
            f'{name}: {value[0]}/{len(log.q)} steps, penetration {value[1]*1000:.2f} mm'
            for name, value in hand_object.items()) or 'none')
        objects = np.asarray(objects)
        normal = (world_from_base @ np.asarray(geometry['robot_base_from_world']))[:3, 2]
        lift = (objects[:, :3]-objects[0, :3]) @ normal
        np.savez_compressed(output/'simulated_motion.npz', timestamp=demo.timestamp[:len(log.q)],
                            arm_hand_measured=log.q, arm_hand_targets=log.target, object_pose_world=objects,
                            world_from_robot_base=world_from_base)
        report = dict(status='completed', hardware_executed=False, source=str(source.resolve()),
            simulator=str(Path(upstream.__file__).resolve()),
            simulator_sha256=hashlib.sha256(Path(upstream.__file__).read_bytes()).hexdigest(),
            report=text, frames=len(log.q), source_duration_s=float(demo.timestamp[-1]-demo.timestamp[0]),
            contacts=contacts, object_maximum_lift_m=float(lift.max()),
            object_final_displacement_m=float(np.linalg.norm(objects[-1, :3]-objects[0, :3])),
            geometry=geometry, object_parameters=config, object_parameters_measured=False,
            source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (source/'robot_demonstration.npz', source/'capture.json', config_path)},
            world_from_robot_base=world_from_base.tolist(),
            note='Full-arm physics; original retarget, not live-rebased execution. Table extent and object dynamics are estimates.')
        (output/'manifest.json').write_text(json.dumps(report, indent=2))
        print(text)
        print(f'Saved full-arm physics videos to {output}', flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--physics-source', type=Path, required=True, help='Folder containing physics_sim package')
    parser.add_argument('--object-config', type=Path, required=True)
    args = parser.parse_args()
    preview(args.source, args.out, args.physics_source, args.object_config)


if __name__ == '__main__': main()
