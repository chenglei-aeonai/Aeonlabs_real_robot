"""Nominal UR5e joint origins in camera space, matching calibration overlays."""
import numpy as np


def arm_joints(data, matrix, width, height):
    if 'robot_state' not in data or 'camera_from_robot_base' not in data:
        return None
    state = data['robot_state']
    q = np.asarray(state['arm_q_rad'], dtype=float)
    tcp = np.asarray(state['tcp_pose_base_xyzw'], dtype=float)
    camera_from_base = np.asarray(data['camera_from_robot_base'], dtype=float)
    if (q.shape != (6,) or tcp.shape != (7,) or camera_from_base.shape != (4, 4)
            or not all(np.isfinite(value).all() for value in (q, tcp, camera_from_base))):
        raise ValueError('Invalid captured robot geometry for arm alignment.')
    # Same standard-DH parameters as calibration/overlay_utils.py.
    a = (0, -.425, -.3922, 0, 0, 0)
    d = (.1625, 0, 0, .1333, .0997, .0996)
    alpha = (np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0)
    transform = np.eye(4)
    origins = [np.zeros(3)]
    for theta, length, offset, twist in zip(q, a, d, alpha):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(twist), np.sin(twist)
        transform = transform @ np.array([
            [c, -s*ca, s*sa, length*c], [s, c*ca, -c*sa, length*s],
            [0, sa, ca, offset], [0, 0, 0, 1]])
        origins.append(transform[:3, 3].copy())
    origins.append(tcp[:3])
    output = []
    for name, point in zip(('base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3', 'tool0', 'TCP'), origins):
        xyz = camera_from_base[:3, :3] @ point + camera_from_base[:3, 3]
        pixel = None
        if xyz[2] > 1e-4:
            uv = matrix @ xyz
            if np.isfinite(uv).all():
                pixel = (uv[:2] / uv[2]).tolist()
        output.append({'name': name, 'position_camera_m': xyz.tolist(), 'pixel_undistorted': pixel,
                       'in_image': bool(pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height)})
    return {'model': 'nominal UR5e DH origins + measured TCP; matches calibration overlays',
            'coordinate_frame': 'RealSense color optical', 'joints': output,
            'human_forearm_reference': {'elbow_camera_m': output[2]['position_camera_m'],
                                       'wrist_camera_m': data['mano']['joints'][0]['position_camera_m'],
                                       'wrist_joint_index': 0,
                                       'definition': 'The displayed human forearm ends at MANO joint 0; robot tool/TCP origins are diagnostic only.'},
            'projection_rule': 'Arm positions were transformed into camera space once; project with K only.'}
