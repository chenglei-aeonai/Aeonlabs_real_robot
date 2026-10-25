"""Validate and preserve already-camera-space capture data for image editing."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .skeleton import KEYPOINT_NAMES, FINGER_CHAINS
from .arm_guidance import arm_joints
from .mano_convention import wrist_rotation, validate_mano_right


def load_guidance(path: Path, image: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get('position_units') != 'metres' or data['mano'].get('coordinate_frame') != 'RealSense color optical':
        raise ValueError('Capture pose must be in camera optical coordinates, in metres.')
    for metadata in (data, data['mano']):
        if metadata.get('handedness', 'right') != 'right':
            raise ValueError('This robot is right-handed; left-hand capture metadata is not supported.')
    camera = data['camera']
    model = camera['undistorted_intrinsics']
    matrix = np.asarray(model['matrix'], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or np.any(np.asarray(model['dist_coeffs'], dtype=float) != 0)):
        raise ValueError('Capture requires valid undistorted intrinsics and zero distortion.')
    with Image.open(image) as source:
        if source.size != (camera['width'], camera['height']):
            raise ValueError('Captured image and intrinsics have different dimensions.')
    joints = data['mano']['joints']
    if len(joints) != 21:
        raise ValueError('Capture must contain 21 mapped MANO landmarks.')
    for index, (joint, name) in enumerate(zip(joints, KEYPOINT_NAMES)):
        point = np.asarray(joint['position_camera_m'], dtype=float)
        if joint['index'] != index or joint['name'] != name or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('Invalid MANO index, name, or camera-space position.')
        pixel = joint['pixel_undistorted']
        if point[2] > 1e-4:
            projected = matrix @ point
            if pixel is None or not np.allclose(pixel, projected[:2] / projected[2], atol=1e-4):
                raise ValueError('MANO projection does not match camera-space XYZ and intrinsics.')
        elif pixel is not None:
            raise ValueError('Landmarks behind the camera must not have projected pixels.')
    points = np.asarray([joint['position_camera_m'] for joint in joints])
    rotation = wrist_rotation(points)
    convention = validate_mano_right(points, rotation)
    return {'handedness': 'right', 'hand_order': ['left', 'right'], 'valid': [False, True],
            'wrist_R_right': rotation.tolist(), 'wrist_convention_validation': convention,
            'wrist_origin_camera_m': points[0].tolist(),
            'orientation_anchors': {joints[i]['name']: {
                'index': i, 'pixel_undistorted': joints[i]['pixel_undistorted'],
                'position_camera_m': joints[i]['position_camera_m']}
                for i in (0, 1, 4, 7, 10, 13, 16, 17, 18, 19, 20)},
            'schema_version': 1, 'image_space': 'undistorted',
            'pose_source': 'capture_ground_truth', 'pose_is_authoritative': True,
            'coordinate_frame': 'RealSense color optical: x right, y down, z forward',
            'position_units': 'metres', 'camera': camera,
            'crop': data.get('crop'),
            'mano': data['mano'], 'capture': data['capture'],
            'arm': arm_joints(data, matrix, camera['width'], camera['height']),
            'source_capture_json': str(path.resolve()),
            'source_capture_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
            'projection_rule': 'Points are already in camera space. Use K @ XYZ and perspective division only; do not apply camera extrinsics again.'}


def overlay_on_edited(image: Path, guidance: dict, output: Path,
                      title='Captured GT MANO pose on converted image'):
    """Reproject the unchanged captured GT onto the final converted photograph."""
    with Image.open(image) as source:
        overlay = source.convert('RGB')
    camera = guidance['camera']
    if overlay.size != (camera['width'], camera['height']):
        raise ValueError('Converted image dimensions changed; cannot overlay captured pose.')
    matrix = np.asarray(camera['undistorted_intrinsics']['matrix'])
    pixels = []
    for joint in guidance['mano']['joints']:
        point = np.asarray(joint['position_camera_m'])
        pixel = None
        if point[2] > 1e-4:
            homogeneous = matrix @ point
            value = homogeneous[:2] / homogeneous[2]
            if np.isfinite(value).all() and np.max(np.abs(value)) < 1e7:
                pixel = tuple(value.tolist())
        pixels.append(pixel)
    draw = ImageDraw.Draw(overlay)
    if guidance.get('arm'):
        # The human forearm and hand share MANO joint 0. Robot tool0/TCP and
        # UR5e wrist axes remain diagnostic geometry, not human wrist landmarks.
        reference = guidance['arm'].get('human_forearm_reference')
        if reference and pixels[0] is not None:
            elbow = np.asarray(reference['elbow_camera_m'], dtype=float)
            if elbow.shape == (3,) and np.isfinite(elbow).all() and elbow[2] > 1e-4:
                value = matrix @ elbow
                uv = value[:2] / value[2]
                if np.isfinite(uv).all() and np.max(np.abs(uv)) < 1e7:
                    draw.line([tuple(uv.tolist()), pixels[0]], fill='#42de7b', width=3)
    colors = ('#2089dd', '#249b42', '#ddb128', '#a24cbe', '#159eae')
    for chain, color in zip(FINGER_CHAINS, colors):
        chain = (0,) + chain
        for start, end in zip(chain[:-1], chain[1:]):
            if pixels[start] is not None and pixels[end] is not None:
                draw.line([pixels[start], pixels[end]], fill=color, width=3)
        for index in chain:
            pixel = pixels[index]
            if pixel is not None and 0 <= pixel[0] < overlay.width and 0 <= pixel[1] < overlay.height:
                x, y = pixel
                marker = 'white' if guidance['mano']['joints'][index]['synthetic'] else color
                draw.ellipse((x-4, y-4, x+4, y+4), fill=marker, outline='black')
                draw.text((x+6, y-10), str(index), fill=marker, stroke_width=1, stroke_fill='black')
    # Anatomical labels disambiguate the two sides of the palm for image editing.
    for index, label in ((0, 'WRIST'), (13, 'THUMB BASE'), (20, 'THUMB TIP'),
                         (16, 'INDEX TIP'), (17, 'MIDDLE TIP'),
                         (19, 'RING TIP'), (18, 'PINKY TIP')):
        pixel = pixels[index]
        if pixel is not None and 0 <= pixel[0] < overlay.width and 0 <= pixel[1] < overlay.height:
            draw.text((pixel[0]+6, pixel[1]+7), label, fill='white', stroke_width=1, stroke_fill='black')
    draw.text((15, 15), title, fill='yellow', stroke_width=1, stroke_fill='black')
    overlay.save(output, format='PNG')
