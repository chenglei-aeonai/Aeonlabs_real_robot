"""Read-only RealSense capture with measured DG5F joints in camera coordinates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import zmq

if __package__:
    from .mano_mapping import draw_mano_overlay, load_wrist_conversion, map_mano_joints
else:
    from mano_mapping import draw_mano_overlay, load_wrist_conversion, map_mano_joints

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'calibration'))
from overlay_utils import HandOverlay, draw_robot_overlay, undistort_for_overlay, ur5e_transforms


if __package__:
    from .calibration_selection import load_calibration, rigid_matrix
else:
    from calibration_selection import load_calibration, rigid_matrix


def read_arm(socket):
    if not socket.poll(1500):
        raise RuntimeError('No arm feedback; start ./arm/run_controller.sh.')
    state = socket.recv_json()
    age = time.time() - state.get('timestamp_ms', 0) / 1000.
    q, qd, tcp = [np.asarray(state.get(key, []), dtype=float) for key in ('Q', 'Qd', 'pos')]
    if (not -.05 <= age <= .25 or q.shape != (6,) or qd.shape != (6,)
            or tcp.shape != (7,) or not all(np.isfinite(x).all() for x in (q, qd, tcp))
            or abs(np.linalg.norm(tcp[3:]) - 1) > .001):
        raise RuntimeError('Arm feedback is stale or invalid; capture refused.')
    if state.get('is_truncated') or state.get('protective_stopped') or state.get('emergency_stopped'):
        raise RuntimeError('Arm reports a stop/fault; capture refused.')
    if np.max(np.abs(qd)) > .01:
        raise RuntimeError('Keep the arm stationary for capture (|Qd| <= 0.01 rad/s).')
    return state


def camera_hand_joints(hand, hand_q, state, camera_from_base, matrix, width, height):
    """26 URDF points: mount/wrist plus four joints and one tip per finger."""
    if hand.tcp_from_hand is None:
        base_from_hand = ur5e_transforms(state['Q'])[-1] @ hand.tool0_from_hand
    else:
        base_from_tcp = np.eye(4)
        base_from_tcp[:3, :3] = Rotation.from_quat(state['pos'][3:]).as_matrix()
        base_from_tcp[:3, 3] = state['pos'][:3]
        base_from_hand = base_from_tcp @ hand.tcp_from_hand
    camera_from_hand = camera_from_base @ base_from_hand
    # skeleton() emits child origins in topological traversal order.
    pending = hand.joints.copy()
    known, names = {'rl_dg_mount'}, []
    while pending:
        remaining = []
        for joint in pending:
            if joint.find('parent').get('link') not in known:
                remaining.append(joint)
                continue
            known.add(joint.find('child').get('link'))
            names.append(joint.get('name'))
        if len(remaining) == len(pending):
            raise ValueError('Unresolved hand URDF tree.')
        pending = remaining
    origins = dict(zip(names, hand.skeleton(hand_q)[:, 1]))
    labels = [('wrist', None, np.zeros(3))]
    for finger in range(1, 6):
        for joint in ('1', '2', '3', '4', 'tip'):
            urdf_name = f'rj_dg_{finger}_{joint}'
            labels.append((f'finger_{finger}_{joint}', urdf_name, origins[urdf_name]))
    joints = []
    for label, urdf_name, point in labels:
        xyz = camera_from_hand[:3, :3] @ point + camera_from_hand[:3, 3]
        uv = None
        if xyz[2] > 1e-4:
            pixel = (matrix @ xyz)[:2] / xyz[2]
            if np.isfinite(pixel).all():
                uv = pixel.tolist()
        joints.append({'name': label, 'urdf_joint': urdf_name,
                       'position_camera_m': xyz.tolist(), 'pixel_undistorted': uv,
                       'in_image': bool(uv is not None and 0 <= uv[0] < width and 0 <= uv[1] < height)})
    return joints, camera_from_hand


def save_capture(folder, image, state, hand, hand_q, camera_from_base, matrix, distortion,
                 provenance, capture_info):
    corrected, projection_matrix, zero = undistort_for_overlay(image, matrix, distortion)
    overlay = draw_robot_overlay(corrected, state, camera_from_base, projection_matrix, zero)
    overlay = hand.draw(overlay, hand_q, state, camera_from_base, projection_matrix, zero)
    height, width = image.shape[:2]
    joints, camera_from_hand = camera_hand_joints(
        hand, hand_q, state, camera_from_base, projection_matrix, width, height)
    wrist_conversion = load_wrist_conversion()
    arm_axis_camera = camera_from_base[:3, :3] @ ur5e_transforms(state['Q'])[-1, :3, 2]
    mano_joints = map_mano_joints(joints, projection_matrix, width, height,
                                wrist_conversion, arm_axis_camera)
    mano_overlay = draw_mano_overlay(corrected, mano_joints)
    for joint in joints:
        if joint['name'] == 'wrist' and joint['in_image']:
            point = tuple(np.asarray(joint['pixel_undistorted'], dtype=int))
            cv2.circle(overlay, point, 5, (255, 255, 255), -1)
            cv2.putText(overlay, 'hand wrist', point, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
    document = {
        'schema_version': 1, 'capture': capture_info, 'handedness': 'right',
        'coordinate_frame': 'RealSense color optical: x right, y down, z forward',
        'position_units': 'metres', 'angle_units': 'radians',
        'camera': {'serial_number': capture_info['serial_number'], 'width': width, 'height': height,
                   'raw_intrinsics': {'matrix': matrix.tolist(), 'dist_coeffs': distortion.tolist()},
                   'undistorted_intrinsics': {'matrix': projection_matrix.tolist(), 'dist_coeffs': zero.tolist()}},
        'calibration': provenance, 'camera_from_robot_base': camera_from_base.tolist(),
        'camera_from_hand_mount': camera_from_hand.tolist(),
        'wrist_definition': 'rl_dg_mount origin; hand mount, not the measured UR TCP',
        'mount_method': 'nominal UR5e tool0 FK + Rz(60deg)' if hand.tcp_from_hand is None else 'configured TCP-to-hand transform',
        'joint_model': 'DG5F right URDF joint origins and fixed fingertips; not anatomical landmarks',
        'hand_joints': joints,
        'mano': {
            'handedness': 'right',
            'convention': 'hand_retarget dpl/MANO 21-point order; fingertips last',
            'coordinate_frame': 'RealSense color optical', 'position_units': 'metres',
            'joints': mano_joints,
            'mapping_note': 'Robot-derived landmark approximation, not a fitted MANO mesh or pose.',
            'wrist_note': 'Hand-mount origin shifted along tool0 Z to match reference wrist/knuckle proportions.',
            'wrist_conversion': wrist_conversion,
            'pinky_dip_note': 'Synthetic 3D midpoint: 0.5 * pinky PIP + 0.5 * pinky tip, in camera coordinates.',
            'projection': 'K_undistorted @ position_camera_m, followed by perspective division; no extrinsic transform.',
        },
        'robot_state': {'arm_q_rad': state['Q'], 'tcp_pose_base_xyzw': state['pos'],
                        'hand_q_rad': np.asarray(hand_q).tolist(), 'arm_timestamp_ms': state['timestamp_ms']},
        'images': {'raw': 'raw.png', 'undistorted': 'undistorted.png', 'overlay': 'overlay.png',
                   'mano_overlay': 'mano_overlay.png'},
        'timing_note': 'Stationary snapshots with fresh feedback; streams are not hardware synchronized.',
    }
    # Serialize first, disallowing nonfinite JSON values.
    encoded = json.dumps(document, indent=2, allow_nan=False) + '\n'
    folder.mkdir(parents=True, exist_ok=False)
    for name, pixels in [('raw.png', image), ('undistorted.png', corrected),
                         ('overlay.png', overlay), ('mano_overlay.png', mano_overlay)]:
        if not cv2.imwrite(str(folder / name), pixels):
            raise OSError(f'Cannot write {folder / name}')
    (folder / 'hand_joints.json').write_text(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--serial_number', default='138322253845')
    parser.add_argument('--name', default='ur5e')
    parser.add_argument('--calibration_dir', type=Path, default=ROOT / 'calibration/results/original_legacy')
    parser.add_argument('--output_dir', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--robot_config_path', type=Path, default=ROOT / 'arm/pc_ur_new.json')
    parser.add_argument('--hand_state_port', type=int, default=5563)
    parser.add_argument('--tcp_to_hand_path', help='Optional rigid T_tcp_hand .npy, matching calibration overlay convention.')
    args = parser.parse_args()
    camera_from_base, provenance = load_calibration(args.calibration_dir, args.name, args.serial_number)
    config = json.loads(args.robot_config_path.read_text())
    import pyrealsense2 as rs

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt_string(zmq.SUBSCRIBE, '')
    socket.connect(f"tcp://127.0.0.1:{config['publisher_port']}")
    hand, pipeline, started = None, rs.pipeline(), False
    try:
        hand = HandOverlay(args.tcp_to_hand_path, args.hand_state_port)
        camera_config = rs.config()
        camera_config.enable_device(args.serial_number)
        camera_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = pipeline.start(camera_config)
        started = True
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        matrix = np.array([[intrinsics.fx, 0., intrinsics.ppx], [0., intrinsics.fy, intrinsics.ppy], [0., 0., 1.]])
        distortion = np.asarray(intrinsics.coeffs, dtype=float)
        for _ in range(30):
            pipeline.wait_for_frames()
        before = read_arm(socket)
        hand_before = hand.capture()
        color = pipeline.wait_for_frames().get_color_frame()
        if not color:
            raise RuntimeError('No camera color frame.')
        image = np.asanyarray(color.get_data()).copy()
        captured_at = datetime.now(timezone.utc).isoformat()
        state = read_arm(socket)
        hand_q = hand.capture()
        if hand_before is None or hand_q is None:
            raise RuntimeError('No fresh measured hand joints; start hand driver and bridge.')
        if (np.max(np.abs(np.asarray(before['Q']) - state['Q'])) > .01
                or np.max(np.abs(hand_before - hand_q)) > .01):
            raise RuntimeError('Arm or fingers moved during capture; keep both stationary and retry.')
        folder = args.output_dir / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        save_capture(folder, image, state, hand, hand_q, camera_from_base, matrix, distortion,
                     provenance, {'captured_at_utc': captured_at, 'serial_number': args.serial_number,
                                  'camera_timestamp_ms': color.get_timestamp(),
                                  'camera_timestamp_domain': str(color.get_frame_timestamp_domain()),
                                  'hand_feedback_age_s': time.monotonic() - hand.last_measured_at})
        print(f'Saved {folder}/hand_joints.json and raw/undistorted/robot/MANO overlay PNG images.')
        print('No robot or hand commands were sent.')
    finally:
        if started:
            pipeline.stop()
        if hand:
            hand.close()
        socket.close(linger=0)
        context.term()


if __name__ == '__main__':
    main()
