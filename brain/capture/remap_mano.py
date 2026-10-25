"""Regenerate mapped MANO outputs from a saved capture without accessing hardware."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

if __package__:
    from .capture_hand import ur5e_transforms
    from .mano_mapping import draw_mano_overlay, load_wrist_conversion, map_mano_joints
else:
    from capture_hand import ur5e_transforms
    from mano_mapping import draw_mano_overlay, load_wrist_conversion, map_mano_joints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('capture_dir', type=Path)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    source = args.capture_dir.resolve()
    output = args.output_dir or source.with_name(source.name + '_mano_wrist')
    document = json.loads((source / 'hand_joints.json').read_text())
    model = document['camera']['undistorted_intrinsics']
    matrix = np.array(model['matrix'])
    image = cv2.imread(str(source / document['images']['undistorted']))
    if image is None:
        raise ValueError('Cannot read saved undistorted image.')
    if np.any(model['dist_coeffs']):
        raise ValueError('Expected zero distortion for the saved undistorted image.')
    rotation = np.array(document['camera_from_robot_base'])[:3, :3]
    axis = rotation @ ur5e_transforms(document['robot_state']['arm_q_rad'])[-1, :3, 2]
    config = load_wrist_conversion()
    joints = map_mano_joints(document['hand_joints'], matrix, image.shape[1], image.shape[0], config, axis)
    document['mano']['joints'] = joints
    document['mano']['wrist_conversion'] = config
    document['mano']['wrist_note'] = 'Hand-mount origin shifted along tool0 Z to match reference wrist/knuckle proportions.'
    document['derived_from_capture'] = str(source)
    encoded = json.dumps(document, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    for key in ('raw', 'undistorted', 'overlay'):
        filename = document['images'][key]
        shutil.copy2(source / filename, output / filename)
    if not cv2.imwrite(str(output / document['images']['mano_overlay']), draw_mano_overlay(image, joints)):
        raise OSError('Could not save mapped overlay.')
    (output / 'hand_joints.json').write_text(encoded)
    print(f'Saved {output}; wrist shift: {joints[0]["wrist_correction"]["shift_m"] * 100:.2f} cm along tool0 +Z.')


if __name__ == '__main__':
    main()
