"""Non-destructive crop/resize of undistorted capture bundles and camera models."""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from .capture_guidance import load_guidance, overlay_on_edited


def crop_bundle(source: Path, destination: Path, *, x: float, y: float,
                width: float, height: float, output_width: int, output_height: int):
    data = json.loads((source / 'hand_joints.json').read_text())
    load_guidance(source / 'hand_joints.json', source / 'undistorted.png')
    camera = data['camera']
    sw, sh = camera['width'], camera['height']
    if (not np.isfinite([x, y, width, height]).all() or x < 0 or y < 0
            or width < 2 or height < 2 or x + width > sw + 1e-6 or y + height > sh + 1e-6
            or not 32 <= output_width <= 4096 or not 32 <= output_height <= 4096):
        raise ValueError('Crop must fit inside the source image; output dimensions must be 32–4096 pixels.')
    if not np.isclose(width / height, output_width / output_height, rtol=1e-5):
        raise ValueError('Crop and output must have the same aspect ratio; stretching is not allowed.')
    destination.mkdir(parents=True, exist_ok=False)
    # PIL EXTENT samples the exact float rectangle used for intrinsic adjustment.
    with Image.open(source / 'undistorted.png') as image:
        cropped = image.convert('RGB').transform((output_width, output_height), Image.Transform.EXTENT,
                    (x, y, x + width, y + height), resample=Image.Resampling.BICUBIC)
        cropped.save(destination / 'undistorted.png')
    sx, sy = output_width / width, output_height / height
    # Pixel-center coordinates: u' = sx*(u - x + 0.5) - 0.5, likewise v.
    pixel_transform = np.array([[sx, 0, sx * (.5 - x) - .5],
                                [0, sy, sy * (.5 - y) - .5], [0, 0, 1.]])
    original_camera = copy.deepcopy(camera)
    matrix = pixel_transform @ np.asarray(camera['undistorted_intrinsics']['matrix'])
    camera['width'], camera['height'] = output_width, output_height
    camera['undistorted_intrinsics']['matrix'] = matrix.tolist()
    camera['undistorted_intrinsics']['width'] = output_width
    camera['undistorted_intrinsics']['height'] = output_height
    if 'raw_intrinsics' in camera:
        camera['raw_intrinsics'].setdefault('width', sw)
        camera['raw_intrinsics'].setdefault('height', sh)
    data['crop'] = {'source_capture_json': str((source / 'hand_joints.json').resolve()),
                    'source_camera': original_camera, 'rectangle_source_px': [x, y, width, height],
                    'output_size_px': [output_width, output_height],
                    'source_to_crop_pixels': pixel_transform.tolist(),
                    'pixel_convention': 'pixel centers; PIL EXTENT resampling',
                    'raw_image': 'raw.png remains the original full-frame camera image',
                    'gt_camera_coordinates_unchanged': True}
    for joints in (data['hand_joints'], data['mano']['joints']):
        for joint in joints:
            point = np.asarray(joint['position_camera_m'])
            pixel = None
            if point[2] > 1e-4:
                projected = matrix @ point
                if np.isfinite(projected).all():
                    pixel = (projected[:2] / projected[2]).tolist()
            joint['pixel_undistorted'] = pixel
            joint['in_image'] = bool(pixel is not None and 0 <= pixel[0] < output_width and 0 <= pixel[1] < output_height)
    # These are the original camera-frame extrinsics; only the image model changed.
    shutil.copyfile(source / 'raw.png', destination / 'raw.png')
    with Image.open(source / 'overlay.png') as image:
        image.convert('RGB').transform((output_width, output_height), Image.Transform.EXTENT,
                    (x, y, x + width, y + height), resample=Image.Resampling.BICUBIC).save(destination / 'overlay.png')
    (destination / 'hand_joints.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    guidance = load_guidance(destination / 'hand_joints.json', destination / 'undistorted.png')
    overlay_on_edited(destination / 'undistorted.png', guidance, destination / 'mano_overlay.png',
                      title='Captured GT MANO pose on cropped image')
    return data
