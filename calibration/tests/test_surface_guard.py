import hashlib
import json
from pathlib import Path
import sys
import tempfile
import struct
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_collision_model import build, stl_vertices
from surface_guard import RobotSurfaceModel, SurfaceError, MODEL_PATH
from motion_safety import SafetyLimits, SafetyError, preflight, plan_poses
from overlay_utils import HandOverlay, ur5e_transforms
from safety_fixture import settings, start_state


class SurfaceGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = RobotSurfaceModel()

    def setUp(self):
        self.limits = SafetyLimits(settings())
        self.state = start_state()

    def test_all_arm_and_hand_collision_meshes_present(self):
        self.assertEqual(len(self.model.mesh_links),35)
        self.assertEqual(sum(len(points) for _,points in self.model.mesh_links),21232)
        report=self.model.check(self.state['Q'],np.zeros(20),self.limits,self.state['pos'])
        self.assertGreater(report['remaining_clearance_m'],0)
        self.assertLess(report['max_modeled_hand_radius_from_tcp_m'],self.limits.data['tool_radius_m'])

    def test_urdf_and_nominal_arm_frames_agree_at_arbitrary_angles(self):
        rng=np.random.default_rng(42)
        for _ in range(10):
            q=rng.uniform(-2,2,6)
            frames=self.model.transforms(q,np.zeros(20))
            np.testing.assert_allclose(frames['tool0'],ur5e_transforms(q)[-1],atol=1e-9)
            expected=np.eye(4)
            expected[:3,:3]=R.from_euler('z',60,degrees=True).as_matrix()
            np.testing.assert_allclose(frames['rl_dg_mount'],frames['tool0']@expected,atol=1e-9)

    def test_support_vertices_preserve_raw_hand_mesh_plane_minima(self):
        root=Path(__file__).resolve().parents[2]
        data=json.loads(MODEL_PATH.read_text())
        rng=np.random.default_rng(25)
        normals=rng.normal(size=(12,3))
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        checked=0
        for mesh in data['meshes']:
            if not mesh['link'].startswith('rl_dg_'):
                continue
            path=root/'hand/ws/src/dg5f_ros2/dg5f_description/meshes/dg5f_right/collision'/Path(mesh['source']).name
            raw=path.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(),mesh['sha256'])
            # Independent read of the original binary triangle vertices.
            triangles=np.frombuffer(raw,dtype=np.dtype([('normal','<f4',(3,)),
                          ('vertices','<f4',(3,3)),('attribute','<u2')]),offset=84)
            vertices=triangles['vertices'].reshape(-1,3).astype(float)
            support=self.model.links[mesh['link']]
            np.testing.assert_allclose(np.min(vertices@normals.T,axis=0),
                                       np.min(support@normals.T,axis=0),atol=1e-12)
            checked+=1
        self.assertEqual(checked,28)

    def test_model_builder_applies_collision_origins_and_mesh_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            points=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
            faces=[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
            payload=b'0'*80+struct.pack('<I',4)
            for indices in faces:
                payload+=struct.pack('<12fH',0.,0.,0.,*points[list(indices)].ravel(),0)
            (root/'tetra.stl').write_bytes(payload)
            (root/'robot.urdf').write_text(
                '<robot name="test"><link name="base"><collision>'
                '<origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>'
                '<geometry><mesh filename="tetra.stl" scale="2 3 4"/></geometry>'
                '</collision></link></robot>')
            with patch('builtins.print'):
                build(root/'robot.urdf',root/'model.json')
            model=json.loads((root/'model.json').read_text())
            actual=np.asarray(model['links'][0]['vertices'])
            expected=(points*[2,3,4])@R.from_euler('z',90,degrees=True).as_matrix().T+[1,2,3]
            for point in expected:
                self.assertLess(np.min(np.linalg.norm(actual-point,axis=1)),1e-12)

    def test_bad_geometry_or_missing_finger_state_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            damaged=Path(directory)/'damaged.json'
            damaged.write_bytes(MODEL_PATH.read_bytes()+b' ')
            with self.assertRaisesRegex(SurfaceError,'checksum'):
                RobotSurfaceModel(damaged)
        for q in (None,np.zeros(19),np.full(20,np.nan)):
            with self.assertRaises(SurfaceError):
                self.model.check(self.state['Q'],q,self.limits,self.state['pos'])
        with self.assertRaises(SurfaceError):
            self.model.check(self.state['Q'],np.zeros(20),self.limits,[np.nan]*7)

    def test_mesh_penetration_and_understated_tool_radius_refused(self):
        collision=SafetyLimits(dict(settings(),table_top_z_m=.01,
                                   table_plane={'frame':'robot_base','normal':[0.,0.,1.],'offset_m':-.01}))
        report=self.model.clearance(self.state['Q'],np.zeros(20),collision,self.state['pos'])
        self.assertLess(report['clearance_m'],0)
        with self.assertRaisesRegex(SurfaceError,'collision surface'):
            self.model.check(self.state['Q'],np.zeros(20),collision,self.state['pos'])
        undersized=SafetyLimits(dict(settings(),tool_radius_m=.1))
        with self.assertRaisesRegex(SurfaceError,'does not enclose'):
            self.model.check(self.state['Q'],np.zeros(20),undersized,self.state['pos'])

    def test_collision_aborts_preflight_before_any_motion(self):
        collision=SafetyLimits(dict(settings(),table_top_z_m=.01,
                                   table_plane={'frame':'robot_base','normal':[0.,0.,1.],'offset_m':-.01}))
        with self.assertRaisesRegex(SafetyError,'URDF surface'):
            preflight([('test',np.asarray(self.state['pos']),True)],self.state,collision,
                      self.model,np.zeros(20))

    def test_all_preflight_segments_include_limiting_surface(self):
        path=preflight(plan_poses(self.state['pos'],self.limits),self.state,self.limits,
                       self.model,np.zeros(20))
        for segment in path:
            self.assertGreaterEqual(segment['closest_urdf_surface_on_segment']['remaining_clearance_m'],0)

    def test_custom_tcp_hand_mount_is_applied(self):
        override=np.eye(4)
        override[:3,3]=[.01,.02,.03]
        frames=self.model.transforms(self.state['Q'],np.zeros(20),[0,0,1,0,0,0,1],override)
        np.testing.assert_allclose(frames['rl_dg_mount'][:3,3],[.01,.02,1.03],atol=1e-12)


class HandSurfaceFeedbackTests(unittest.TestCase):
    def test_poll_is_nonblocking_and_rejects_expired_measurements(self):
        with patch('socket.socket'):
            sensor=HandOverlay()
        packet=json.dumps({'type':'hand_state','positions':[.1]*20,'measured_age_s':.01}).encode()
        sensor.socket.recvfrom.side_effect=[(packet,('127.0.0.1',1)),BlockingIOError]
        with patch('time.monotonic',return_value=10.):
            np.testing.assert_allclose(sensor.poll_positions(),.1)
        sensor.socket.recvfrom.side_effect=BlockingIOError
        with patch('time.monotonic',return_value=11.):
            self.assertIsNone(sensor.poll_positions())


if __name__=='__main__':
    unittest.main()
