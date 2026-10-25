"""Map already camera-space DG5F landmarks to the retargeter's 21-point order."""
import cv2
import numpy as np
import json
from pathlib import Path


# Matches hand_retarget/src/hand_retarget/skeleton.py, including tips-last order.
KEYPOINT_NAMES = (
    'wrist', 'index_mcp', 'index_pip', 'index_dip',
    'middle_mcp', 'middle_pip', 'middle_dip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip',
    'ring_mcp', 'ring_pip', 'ring_dip',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip',
    'index_tip', 'middle_tip', 'pinky_tip', 'ring_tip', 'thumb_tip',
)
FINGER_CHAINS = ((13, 14, 15, 20), (1, 2, 3, 16), (4, 5, 6, 17),
                 (10, 11, 12, 19), (7, 8, 9, 18))
SOURCES = (
    'wrist', 'finger_2_2', 'finger_2_3', 'finger_2_4',
    'finger_3_2', 'finger_3_3', 'finger_3_4',
    'finger_5_3', 'finger_5_4', None,
    'finger_4_2', 'finger_4_3', 'finger_4_4',
    'finger_1_2', 'finger_1_3', 'finger_1_4',
    'finger_2_tip', 'finger_3_tip', 'finger_5_tip', 'finger_4_tip', 'finger_1_tip',
)


def load_wrist_conversion(path=None):
    path = Path(path) if path else Path(__file__).with_name('mano_wrist_conversion.json')
    config = json.loads(path.read_text())
    ratio = config['wrist_to_knuckle_center_over_index_pinky_width']
    if (config['version'] != 1 or config['method'] != 'knuckle_width_scaled_wrist_distance_on_tool0_axis'
            or not np.isfinite(ratio) or ratio <= 0):
        raise ValueError('Unsupported or invalid wrist conversion.')
    return config


def corrected_wrist(lookup, arm_axis_camera, config):
    """Keep the wrist on its tool-axis line and match reference palm proportions."""
    axis = np.asarray(arm_axis_camera, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8:
        raise ValueError('Wrist conversion requires a finite nonzero camera-space arm axis.')
    axis = axis / np.linalg.norm(axis)
    original = lookup['wrist']
    knuckles = [lookup[key] for key in ('finger_2_2', 'finger_3_2', 'finger_4_2', 'finger_5_3')]
    center = np.mean(knuckles, axis=0)
    width = np.linalg.norm(knuckles[0] - knuckles[3])
    target = config['wrist_to_knuckle_center_over_index_pinky_width'] * width
    delta = center - original
    longitudinal = np.dot(delta, axis)
    lateral_squared = np.dot(delta, delta) - longitudinal ** 2
    discriminant = target ** 2 - max(0., lateral_squared)
    if width < 1e-6 or discriminant < -1e-10:
        raise ValueError('Reference wrist distance cannot be achieved along the tool axis for this hand geometry.')
    # Proximal root: the corrected wrist remains behind the knuckle center.
    shift = float(longitudinal - np.sqrt(max(0., discriminant)))
    point = original + shift * axis
    return point, {'axis_camera': axis.tolist(), 'shift_m': shift,
                   'original_position_camera_m': original.tolist(),
                   'knuckle_center_camera_m': center.tolist(),
                   'knuckle_width_m': float(width), 'target_wrist_to_knuckles_m': float(target)}


def map_mano_joints(robot_joints, camera_matrix, width, height,
                    wrist_conversion=None, arm_axis_camera=None):
    """Interpolate in 3D camera space, then project with K only (zero distortion)."""
    lookup = {joint['name']: np.asarray(joint['position_camera_m'], dtype=float)
              for joint in robot_joints}
    matrix = np.asarray(camera_matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError('Invalid undistorted camera intrinsics.')
    output = []
    for index, (name, source) in enumerate(zip(KEYPOINT_NAMES, SOURCES)):
        weights = {source: 1.0} if source else {'finger_5_4': .5, 'finger_5_tip': .5}
        point = sum(weight * lookup[key] for key, weight in weights.items())
        correction = None
        if index == 0 and wrist_conversion is not None:
            point, correction = corrected_wrist(lookup, arm_axis_camera, wrist_conversion)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f'Invalid camera-space point for {name}.')
        pixel = None
        if point[2] > 1e-4:
            homogeneous = matrix @ point
            projected = homogeneous[:2] / homogeneous[2]
            if np.isfinite(projected).all():
                pixel = projected.tolist()
        output.append({'index': index, 'name': name, 'position_camera_m': point.tolist(),
                       'pixel_undistorted': pixel,
                       'in_image': bool(pixel is not None and 0 <= pixel[0] < width and 0 <= pixel[1] < height),
                       'synthetic': source is None or correction is not None,
                       'source_weights': weights})
        if correction is not None:
            output[-1]['wrist_correction'] = correction
    return output


def draw_mano_overlay(undistorted_image, joints):
    """Draw exactly the exported camera-space mapping's projected pixels."""
    overlay = undistorted_image.copy()
    height, width = overlay.shape[:2]
    colors = ((255, 150, 0), (255, 0, 100), (0, 180, 255), (180, 0, 255), (255, 255, 0))

    def drawable(index):
        pixel = joints[index]['pixel_undistorted']
        return pixel is not None and np.isfinite(pixel).all() and np.max(np.abs(pixel)) < 1e7

    for chain, color in zip(FINGER_CHAINS, colors):
        chain = (0,) + chain
        for start, end in zip(chain[:-1], chain[1:]):
            if drawable(start) and drawable(end):
                a = tuple(np.asarray(joints[start]['pixel_undistorted'], dtype=int))
                b = tuple(np.asarray(joints[end]['pixel_undistorted'], dtype=int))
                visible, a, b = cv2.clipLine((0, 0, width, height), a, b)
                if visible:
                    cv2.line(overlay, a, b, color, 2, cv2.LINE_AA)
        for index in chain:
            joint = joints[index]
            if joint['in_image'] and drawable(index):
                pixel = tuple(np.asarray(joint['pixel_undistorted'], dtype=int))
                marker = (255, 255, 255) if joint['synthetic'] else color
                cv2.circle(overlay, pixel, 5 if joint['synthetic'] else 3, marker, -1, cv2.LINE_AA)
                cv2.putText(overlay, str(index), (pixel[0] + 5, pixel[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .4, marker, 1, cv2.LINE_AA)
    cv2.putText(overlay, 'Mapped MANO 21 | white: corrected wrist / synthetic pinky DIP',
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
    return overlay
