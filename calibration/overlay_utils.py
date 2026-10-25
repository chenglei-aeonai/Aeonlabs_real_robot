"""Hardware-independent robot skeleton projection helpers."""
import os
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


def ur5e_transforms(joint_angles):
    """Nominal standard-DH origins in UR base coordinates, in metres.

    Source: Universal Robots' published UR5e DH parameters. These are a
    kinematic skeleton, not mesh surface points or factory-calibrated links.
    https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    """
    q = np.asarray(joint_angles, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("Overlay requires six finite UR5e joint angles (Q).")
    a = (0, -0.425, -0.3922, 0, 0, 0)
    d = (0.1625, 0, 0, 0.1333, 0.0997, 0.0996)
    alpha = (np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0)
    transform = np.eye(4)
    transforms = [transform.copy()]
    for theta, length, offset, twist in zip(q, a, d, alpha):
        c, s = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(twist), np.sin(twist)
        transform = transform @ np.array([
            [c, -s * ca, s * sa, length * c],
            [s, c * ca, -c * sa, length * s],
            [0, sa, ca, offset], [0, 0, 0, 1],
        ])
        transforms.append(transform.copy())
    return np.array(transforms)


def ur5e_skeleton(joint_angles):
    return ur5e_transforms(joint_angles)[:, :3, 3]


def undistort_for_overlay(frame, camera_matrix, dist_coeffs):
    """Full-size undistorted image and its matching pinhole projection model.

    Preserve the original intrinsic matrix and image extent, without cropping.
    Reproject with returned ZERO distortion to avoid applying distortion twice.
    Original frames and calibration observations remain in their raw image space.
    """
    matrix = np.asarray(camera_matrix, dtype=float).copy()
    distortion = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError('Invalid overlay camera model.')
    if len(distortion) not in (4, 5, 8, 12, 14):
        raise ValueError('Unsupported OpenCV distortion coefficient count.')
    image = (cv2.undistort(frame, matrix, distortion, None, matrix)
             if np.any(distortion) else frame.copy())
    return image, matrix, np.zeros_like(distortion)


def draw_robot_overlay(frame, robot_state, camera_from_base, camera_matrix, dist_coeffs):
    """Project nominal arm origins and measured TCP using the final calibration."""
    points = ur5e_skeleton(robot_state['Q'])
    points = np.vstack((points, np.asarray(robot_state['pos'][:3], dtype=float)))
    camera_points = points @ camera_from_base[:3, :3].T + camera_from_base[:3, 3]
    visible = np.isfinite(camera_points).all(axis=1) & (camera_points[:, 2] > 1e-4)
    pixels = np.zeros((len(points), 2), dtype=float)
    if visible.any():
        pixels[visible] = cv2.projectPoints(
            camera_points[visible], np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs
        )[0].reshape(-1, 2)
    # Guard OpenCV's integer coordinates for pathological near-camera projections.
    visible &= np.isfinite(pixels).all(axis=1) & (np.abs(pixels) < 1e7).all(axis=1)
    pixels = np.nan_to_num(np.clip(pixels, -1e7, 1e7)).astype(int)
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    for index in range(1, len(points)):
        if visible[index - 1] and visible[index]:
            ok, start, end = cv2.clipLine((0, 0, width, height),
                                         tuple(pixels[index - 1]), tuple(pixels[index]))
            if ok:
                cv2.line(overlay, start, end, (0, 255, 0), 3, cv2.LINE_AA)
    labels = ['base', 'shoulder', 'elbow', 'wrist1', 'wrist2', 'wrist3', 'tool0', 'TCP']
    for index, label in enumerate(labels):
        x, y = pixels[index]
        if visible[index] and 0 <= x < width and 0 <= y < height:
            color = (255, 0, 255) if label == 'TCP' else (0, 255, 255)
            cv2.circle(overlay, (x, y), 6, color, -1, cv2.LINE_AA)
            cv2.putText(overlay, label, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
    cv2.putText(overlay, 'Final calibration | nominal UR5e skeleton | magenta: measured TCP',
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    return overlay



class HandOverlay:
    """Read-only measured DG5F skeleton from the bundled right-hand URDF."""

    def __init__(self, tcp_to_hand_path=None, state_port=5563):
        import socket

        self.tcp_from_hand = (np.load(tcp_to_hand_path, allow_pickle=False)
                              if tcp_to_hand_path else None)
        # hand_retarget/assets/ur5e_right_dg5f.urdf (eff2810),
        # ur5e_dg5f_mount: wrist_3_link -> rl_dg_mount, Rz(60deg), no offset.
        # In that URDF wrist_3_link and tool0 have identical transforms.
        self.tool0_from_hand = np.eye(4)
        self.tool0_from_hand[:3, :3] = R.from_euler('z', np.pi / 3).as_matrix()
        t = self.tcp_from_hand if self.tcp_from_hand is not None else self.tool0_from_hand
        if (t.shape != (4, 4) or not np.isfinite(t).all()
                or not np.allclose(t[3], [0, 0, 0, 1])
                or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(t[:3, :3]), 1)):
            raise ValueError("T_tcp_hand must be a rigid 4x4 homogeneous transform.")
        urdf = os.path.join(os.path.dirname(__file__), '..', 'hand', 'ws', 'src',
                            'dg5f_ros2', 'dg5f_description', 'urdf', 'dg5f_right.urdf')
        self.joints = list(ET.parse(urdf).getroot().findall('joint'))
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(('127.0.0.1', state_port))
        self.socket.setblocking(False)

    def capture(self):
        """Discard queued packets, then wait briefly for a new measured state."""
        import json
        import select
        import time

        while True:
            try:
                self.socket.recvfrom(65535)
            except BlockingIOError:
                break
        deadline = time.monotonic() + 0.15
        while time.monotonic() < deadline:
            if not select.select([self.socket], [], [], max(0, deadline - time.monotonic()))[0]:
                break
            try:
                message = json.loads(self.socket.recvfrom(65535)[0])
                q = np.asarray(message.get('positions', []), dtype=float)
                age = float(message.get('measured_age_s', float('inf')))
                if (message.get('type') == 'hand_state' and q.shape == (20,)
                        and np.isfinite(q).all() and 0 <= age <= 0.25):
                    self.last_q = q.copy()
                    self.last_measured_at = time.monotonic() - age
                    return q
            except (ValueError, TypeError, AttributeError):
                continue
        return None

    def poll_positions(self):
        """Nonblocking fresh feedback for motion supervision; never sends commands."""
        import json
        import time

        while True:
            try:
                payload, _ = self.socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                message = json.loads(payload)
                q = np.asarray(message.get('positions', []), dtype=float)
                age = float(message.get('measured_age_s', float('inf')))
                if message.get('type') == 'hand_state' and q.shape == (20,) and np.isfinite(q).all() and 0 <= age <= .25:
                    self.last_q = q.copy()
                    self.last_measured_at = time.monotonic() - age
            except (ValueError, TypeError, AttributeError):
                continue
        if not hasattr(self, 'last_q') or time.monotonic() - self.last_measured_at > .25:
            return None
        return self.last_q.copy()

    def skeleton(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            raise ValueError("Hand overlay requires 20 finite measured right-hand angles.")
        angles = {f'rj_dg_{finger}_{joint}': q[(finger - 1) * 4 + joint - 1]
                  for finger in range(1, 6) for joint in range(1, 5)}
        transforms = {'rl_dg_mount': np.eye(4)}
        segments = []
        pending = self.joints.copy()
        while pending:
            remaining = []
            for joint in pending:
                parent = joint.find('parent').get('link')
                child = joint.find('child').get('link')
                if parent not in transforms:
                    remaining.append(joint)
                    continue
                origin = joint.find('origin')
                transform = np.eye(4)
                if origin is not None:
                    transform[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
                    transform[:3, :3] = R.from_euler(
                        'xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')).as_matrix()
                if joint.get('type') in ('revolute', 'continuous'):
                    axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                    rotation = R.from_rotvec(axis / np.linalg.norm(axis) * angles[joint.get('name')])
                    transform[:3, :3] = transform[:3, :3] @ rotation.as_matrix()
                elif joint.get('type') != 'fixed':
                    raise ValueError(f"Unsupported hand joint type: {joint.get('type')}")
                transforms[child] = transforms[parent] @ transform
                segments.append((transforms[parent][:3, 3], transforms[child][:3, 3]))
            if len(remaining) == len(pending):
                raise ValueError("Hand URDF tree cannot be resolved from rl_dg_mount.")
            pending = remaining
        return np.asarray(segments)

    def draw(self, frame, q, robot_state, camera_from_base, camera_matrix, dist_coeffs):
        base_from_tcp = np.eye(4)
        base_from_tcp[:3, 3] = robot_state['pos'][:3]
        base_from_tcp[:3, :3] = R.from_quat(robot_state['pos'][3:]).as_matrix()
        if self.tcp_from_hand is None:
            base_from_hand = ur5e_transforms(robot_state['Q'])[-1] @ self.tool0_from_hand
        else:
            base_from_hand = base_from_tcp @ self.tcp_from_hand
        camera_from_hand = camera_from_base @ base_from_hand
        segments = self.skeleton(q)
        points = segments.reshape(-1, 3)
        points = points @ camera_from_hand[:3, :3].T + camera_from_hand[:3, 3]
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        colors = [(255, 150, 0), (255, 0, 100), (0, 180, 255), (180, 0, 255), (255, 255, 0)]
        # All segments include the fixed palm/mount and the five fingertip joints.
        for index, segment in enumerate(points.reshape(-1, 2, 3)):
            if not np.isfinite(segment).all() or (segment[:, 2] <= 1e-4).any():
                continue
            uv = cv2.projectPoints(segment, np.zeros(3), np.zeros(3),
                                   camera_matrix, dist_coeffs)[0].reshape(2, 2)
            if not np.isfinite(uv).all() or (np.abs(uv) >= 1e7).any():
                continue
            start, end = [tuple(point.astype(int)) for point in uv]
            color = (180, 180, 180) if index < 2 else colors[(index - 2) // 5]
            ok, clipped_start, clipped_end = cv2.clipLine((0, 0, width, height), start, end)
            if ok:
                cv2.line(overlay, clipped_start, clipped_end, color, 2, cv2.LINE_AA)
            if 0 <= end[0] < width and 0 <= end[1] < height:
                cv2.circle(overlay, end, 3, color, -1, cv2.LINE_AA)
        cv2.putText(overlay, 'DG5F: measured right-hand joints + URDF fingertips',
                    (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        return overlay

    def close(self):
        self.socket.close()
