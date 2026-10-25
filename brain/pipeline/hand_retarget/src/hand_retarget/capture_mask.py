"""Deterministic edit envelope from captured camera-space joints; no detector."""
import json
import numpy as np
from PIL import Image, ImageDraw
from .capture_guidance import load_guidance
from .skeleton import FINGER_CHAINS


def projected_capture_mask(image, capture_json, output):
    guidance = load_guidance(capture_json, image)
    data = json.loads(capture_json.read_text())
    camera = guidance['camera']
    k = np.asarray(camera['undistorted_intrinsics']['matrix'])
    mask = Image.new('L', (camera['width'], camera['height']))
    draw = ImageDraw.Draw(mask)

    def stroke(points, radius_m):
        previous = None
        for point in points:
            xyz = np.asarray(point, dtype=float)
            if xyz.shape != (3,) or not np.isfinite(xyz).all() or xyz[2] <= .01:
                previous = None
                continue
            uv = k @ xyz; uv = uv[:2]/uv[2]
            if np.max(np.abs(uv)) > 1e6:
                previous = None
                continue
            radius = min(max(camera['width'], camera['height']), max(3., max(k[0, 0], k[1, 1])*radius_m/xyz[2]))
            x, y = uv
            if previous is not None:
                draw.line([previous[0], tuple(uv)], fill=255, width=int(2*max(previous[1], radius)))
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
            previous = (tuple(uv), radius)

    joints = [j['position_camera_m'] for j in guidance['mano']['joints']]
    for chain in FINGER_CHAINS:
        stroke([joints[i] for i in (0,)+tuple(chain)], .018)
    stroke([joints[i] for i in (0, 1, 4, 10, 7, 0)], .035)
    robot = data.get('hand_joints', [])
    if len(robot) == 26:
        for finger in range(5):
            indices = [0]+list(range(1+5*finger, 6+5*finger))
            stroke([robot[i]['position_camera_m'] for i in indices], .022)
    if guidance.get('arm'):
        stroke([j['position_camera_m'] for j in guidance['arm']['joints']], .065)
        stroke([guidance['arm']['joints'][-1]['position_camera_m'], joints[0]], .05)
    if not mask.getbbox():
        raise ValueError('Captured skeleton does not intersect the image; adjust the crop.')
    mask.save(output)
    return {'provider': 'captured_joint_projection', 'detector_used': False,
            'mask': str(output), 'coverage_fraction': float(np.mean(np.asarray(mask)>0)),
            'occlusion_verified': False,
            'note': 'Geometric edit envelope, not object segmentation. Preserve table and object occlusions within it; manual mask can override.'}
