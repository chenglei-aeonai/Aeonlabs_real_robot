import json
import sys
from PIL import Image
import numpy as np
from hand_retarget import image_edit
from hand_retarget.capture_mask import projected_capture_mask
from test_capture_bridge import bundle


def test_projection_mask_and_stage2_skip_detector(tmp_path, monkeypatch):
    source = tmp_path/'source'
    bundle(source)
    mask = tmp_path/'mask.png'
    report = projected_capture_mask(source/'undistorted.png', source/'hand_joints.json', mask)
    pixels = np.asarray(Image.open(mask))
    assert report['detector_used'] is False
    assert pixels.shape == (48, 64) and pixels.max() == 255 and pixels.min() == 0
    assert pixels[32, 36] == 255  # Provided camera-space wrist projected by K.
    def forbidden(*args, **kwargs):
        raise AssertionError('Captured frame must never invoke independent segmentation')
    monkeypatch.setattr(image_edit, 'segment_hands', forbidden)
    monkeypatch.setattr(image_edit, 'replace_hands', lambda *a, **kw: {'test': True})
    output = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['image_edit', '--image', str(source/'undistorted.png'),
                         '--capture-json', str(source/'hand_joints.json'), '--out', str(output)])
    image_edit.main()
    manifest = json.loads((output/'manifest.json').read_text())
    assert manifest['segmentation']['provider'] == 'captured_joint_projection'
    assert manifest['status'] == 'completed'
