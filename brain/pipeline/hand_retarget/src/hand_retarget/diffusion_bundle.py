"""Converted RGB + immutable camera-frame GT -> aeon custom-frame bundle."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .capture_guidance import load_guidance
from .skeleton import KEYPOINT_NAMES

from .mano_convention import RIGHT_SIGNATURE, wrist_rotation, validate_mano_right


def build_bundle(image: Path, capture_json: Path, output: Path):
    guidance = load_guidance(capture_json, image)
    with Image.open(image) as source:
        frame = np.asarray(source.convert('RGB')).copy()
    height, width = frame.shape[:2]
    if min(height, width) < 32 or max(height, width) > 2048 or height % 32 or width % 32:
        raise ValueError('Diffusion input dimensions must be multiples of 32 between 32 and 2048; adjust the capture crop output.')
    right = np.array([j['position_camera_m'] for j in guidance['mano']['joints']], dtype=np.float64)
    if not np.isfinite(right).all() or np.any(right[:, 2] <= 0):
        raise ValueError('GT hand must have finite positive optical depth.')
    joints = np.zeros((2, 21, 3), dtype=np.float64); joints[1] = right
    rotations = np.tile(np.eye(3), (2, 1, 1)); rotations[1] = wrist_rotation(right)
    convention=validate_mano_right(right,rotations[1])
    matrix = np.array(guidance['camera']['undistorted_intrinsics']['matrix'], dtype=np.float64)
    np.savez_compressed(output, frame=frame, joints3d=joints, wrist_R=rotations,
                        valid=np.array([False, True]), K=matrix)
    report = {'handedness': 'right', 'valid': [False, True], 'pose_source': 'capture_ground_truth', 'pose_reestimated': False,
              'units': 'metres', 'coordinate_frame': 'optical camera', 'right_hand_index': 1,
              'hand_3d_definition': {
                  'representation':'21 Cartesian landmarks in DPL/MANO order; not MANO pose parameters or a mesh',
                  'shape':[2,21,3], 'hand_order':['left','right'],
                  'joint_names':list(KEYPOINT_NAMES),
                  'axes':{'x':'image right','y':'image down','z':'forward from camera'},
                  'origin':'camera optical centre', 'units':'metres',
                  'coordinates':'absolute camera-space positions, not wrist-relative or normalized',
                  'valid':[False,True], 'dtype':'float64; runtime may convert to float32',
                  'provenance':'robot forward kinematics mapped to human landmark labels; includes synthetic landmarks, not independently measured human anatomy',
                  'synthetic_joint_indices':[j['index'] for j in guidance['mano']['joints'] if j.get('synthetic')],
                  'projection':'q = K @ J_camera; pixel = q[:2] / q[2]; zero distortion',
              },
              'image_size_px': [width, height], 'K': matrix.tolist(),
              'wrist_rotation': 'Derived from captured MCP geometry and dpl MANO-right signature, nearest SO(3); no joint-position fitting.',
              'wrist_convention_validation':convention,
              'wrist_R_right':rotations[1].tolist(),
              'dpl_revision': '48575bd9dbec73c7180cc430fa6cee31c62ec187',
              'image_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
              'bundle_sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
