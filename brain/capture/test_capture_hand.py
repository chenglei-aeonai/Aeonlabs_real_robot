"""Hardware-free verification of calibration selection and camera-space export."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import capture_hand as capture
from mano_mapping import KEYPOINT_NAMES, corrected_wrist, load_wrist_conversion, map_mano_joints


def model():
    # Construct only the geometry; no UDP socket or camera is opened.
    hand = capture.HandOverlay.__new__(capture.HandOverlay)
    hand.joints = list(ET.parse(capture.ROOT / 'hand/ws/src/dg5f_ros2/dg5f_description/urdf/dg5f_right.urdf').getroot().findall('joint'))
    hand.tcp_from_hand = np.eye(4)
    hand.tcp_from_hand[0, 3] = .1
    return hand


def state():
    return {'Q': [0.] * 6, 'Qd': [0.] * 6, 'pos': [0, 0, 0, 0, 0, 0, 1],
            'timestamp_ms': time.time() * 1000}


class CaptureTests(unittest.TestCase):
    def test_latest_complete_calibration_pair_is_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for stamp in ('20261023_130201', '20261024_095538'):
                base = np.eye(4); base[0, 3] = int(stamp[:8])
                camera = np.eye(4); camera[2, 3] = -2
                np.save(directory / f'{stamp}_base_pose_robot_ur5e.npy', base)
                np.save(directory / f'{stamp}_camera_extrinsic_123.npy', camera)
            np.save(directory / '20261025_000000_base_pose_robot_ur5e.npy', np.eye(4))
            transform, source = capture.load_calibration(directory, 'ur5e', '123')
            self.assertEqual(transform[0, 3], 20261024)
            self.assertEqual(transform[2, 3], 2)
            self.assertIn('20261024_095538', source['world_from_camera']['path'])

    def test_camera_coordinates_pixels_and_finger_motion(self):
        hand = model()
        transform = np.eye(4); transform[2, 3] = 2
        matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
        joints, camera_from_hand = capture.camera_hand_joints(hand, np.zeros(20), state(), transform, matrix, 640, 480)
        self.assertEqual(len(joints), 26)
        self.assertEqual(len({j['name'] for j in joints}), 26)
        np.testing.assert_allclose(joints[0]['position_camera_m'], [.1, 0, 2])
        np.testing.assert_allclose(joints[0]['pixel_undistorted'], [345, 240])
        fingertip = hand.skeleton(np.zeros(20))[6, 1]
        expected = camera_from_hand[:3, :3] @ fingertip + camera_from_hand[:3, 3]
        np.testing.assert_allclose(joints[5]['position_camera_m'], expected)
        angles = np.zeros(20); angles[1] = .3
        bent, _ = capture.camera_hand_joints(hand, angles, state(), transform, matrix, 640, 480)
        self.assertFalse(np.allclose(joints[5]['position_camera_m'], bent[5]['position_camera_m']))
        transform[2, 3] = -2
        behind, _ = capture.camera_hand_joints(hand, angles, state(), transform, matrix, 640, 480)
        self.assertIsNone(behind[0]['pixel_undistorted'])
        self.assertFalse(behind[0]['in_image'])

    def test_outputs_have_matching_intrinsics_and_3d_points(self):
        hand = model()
        hand.tcp_from_hand = None
        hand.tool0_from_hand = np.eye(4)
        image = np.full((480, 640, 3), 100, np.uint8)
        matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
        distortion = np.array([.1, -.01, 0., 0., 0.])
        transform = np.eye(4); transform[2, 3] = 2
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'capture'
            capture.save_capture(output, image, state(), hand, np.zeros(20), transform, matrix,
                                 distortion, {}, {'serial_number': '123'})
            data = json.loads((output / 'hand_joints.json').read_text())
            self.assertEqual(len(data['hand_joints']), 26)
            self.assertEqual(len(data['mano']['joints']), 21)
            self.assertEqual([j['name'] for j in data['mano']['joints']], list(KEYPOINT_NAMES))
            self.assertEqual([j['index'] for j in data['mano']['joints']], list(range(21)))
            self.assertEqual([j['index'] for j in data['mano']['joints'] if j['synthetic']], [0, 9])
            self.assertIsNotNone(cv2.imread(str(output / data['images']['mano_overlay'])))
            self.assertEqual(data['camera']['undistorted_intrinsics']['dist_coeffs'], [0] * 5)
            corrected = cv2.imread(str(output / 'undistorted.png'))
            np.testing.assert_array_equal(corrected, cv2.undistort(image, matrix, distortion, None, matrix))
            for joint in data['hand_joints'] + data['mano']['joints']:
                xyz = np.array(joint['position_camera_m'])
                if xyz[2] > 1e-4:
                    np.testing.assert_allclose(joint['pixel_undistorted'], (matrix @ xyz)[:2] / xyz[2])

    def test_mano_mapping_uses_camera_space_3d_interpolation_before_projection(self):
        # Distinct depths make averaging pixels differ from projecting the 3D midpoint.
        robot = [{'name': 'wrist', 'position_camera_m': [0., 0., 2.]}]
        for finger in range(1, 6):
            for number, joint in enumerate(('1', '2', '3', '4', 'tip'), 1):
                robot.append({'name': f'finger_{finger}_{joint}',
                              'position_camera_m': [finger * .1, number * .1, float(number)]})
        matrix = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
        mapped = map_mano_joints(robot, matrix, 640, 480)
        lookup = {j['name']: j['position_camera_m'] for j in robot}
        expected_sources = ['wrist', 'finger_2_2', 'finger_2_3', 'finger_2_4',
                            'finger_3_2', 'finger_3_3', 'finger_3_4',
                            'finger_5_3', 'finger_5_4', None,
                            'finger_4_2', 'finger_4_3', 'finger_4_4',
                            'finger_1_2', 'finger_1_3', 'finger_1_4',
                            'finger_2_tip', 'finger_3_tip', 'finger_5_tip', 'finger_4_tip', 'finger_1_tip']
        for joint, source in zip(mapped, expected_sources):
            if source:
                np.testing.assert_array_equal(joint['position_camera_m'], lookup[source])
        midpoint = (np.array(lookup['finger_5_4']) + lookup['finger_5_tip']) / 2
        np.testing.assert_allclose(mapped[9]['position_camera_m'], midpoint)
        projected = matrix @ midpoint
        np.testing.assert_allclose(mapped[9]['pixel_undistorted'], projected[:2] / projected[2])
        pixel_midpoint = (np.array(mapped[8]['pixel_undistorted']) + mapped[18]['pixel_undistorted']) / 2
        self.assertFalse(np.allclose(mapped[9]['pixel_undistorted'], pixel_midpoint))
        robot[0]['position_camera_m'] = [0., 0., -1.]
        behind = map_mano_joints(robot, matrix, 640, 480)[0]
        self.assertIsNone(behind['pixel_undistorted'])
        self.assertFalse(behind['in_image'])

    def test_corrected_wrist_moves_only_on_axis_and_preserves_reference_proportion(self):
        lookup = {'wrist': np.array([0., 0., 0.])}
        for key, x in zip(('finger_2_2', 'finger_3_2', 'finger_4_2', 'finger_5_3'), [-.04, -.015, .015, .04]):
            lookup[key] = np.array([x, .01, .16])
        config = load_wrist_conversion()
        point, details = corrected_wrist(lookup, [0, 0, 1], config)
        np.testing.assert_allclose(point[:2], [0, 0])
        center = np.array(details['knuckle_center_camera_m'])
        self.assertAlmostEqual(np.linalg.norm(center - point), .08 * config['wrist_to_knuckle_center_over_index_pinky_width'])
        self.assertLess(point[2], center[2])
        rotation = capture.Rotation.from_euler('xyz', [.4, -.2, 1.]).as_matrix()
        translation = np.array([.1, -.4, 2.])
        rotated = {key: rotation @ value + translation for key, value in lookup.items()}
        moved, _ = corrected_wrist(rotated, rotation[:, 2], config)
        np.testing.assert_allclose(moved, rotation @ point + translation)
        with self.assertRaises(ValueError):
            corrected_wrist(lookup, [0, 0, 0], config)

    def test_stale_moving_and_faulted_arm_states_are_refused(self):
        socket = Mock(); socket.poll.return_value = True
        socket.recv_json.return_value = state()
        capture.read_arm(socket)
        for field, value in [('timestamp_ms', (time.time() - 1) * 1000),
                             ('Qd', [.1] * 6), ('is_truncated', True)]:
            invalid = state(); invalid[field] = value
            socket.recv_json.return_value = invalid
            with self.assertRaises(RuntimeError):
                capture.read_arm(socket)


if __name__ == '__main__':
    unittest.main()
