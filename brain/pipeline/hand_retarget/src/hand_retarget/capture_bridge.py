"""Run the repository's read-only capture in its separate camera environment."""
import json
from pathlib import Path
import re
import subprocess
import threading
import uuid
import math

from fastapi import HTTPException
from .calibration_status import status as calibration_status

FILES = {'raw.png', 'undistorted.png', 'overlay.png', 'mano_overlay.png', 'hand_joints.json'}


class Captures:
    def __init__(self, root: Path, runner: Path):
        self.root, self.runner = root, runner
        self.lock = threading.Lock()
        self.defaults_path = root.parent / 'crop_defaults.json'

    def saved_defaults(self):
        if self.defaults_path.is_file():
            return json.loads(self.defaults_path.read_text())
        # Recover the last applied crop from captures made before persistence.
        for index in sorted(self.root.glob('*/index.json'), key=lambda p: p.stat().st_mtime, reverse=True):
            data = json.loads(index.read_text())
            if data.get('artifact_dir') != 'crop':
                continue
            path = index.parent / 'crop/hand_joints.json'
            if path.is_file():
                crop = json.loads(path.read_text()).get('crop')
                if crop:
                    camera = crop['source_camera']
                    x, y, width, height = crop['rectangle_source_px']
                    ow, oh = crop['output_size_px']
                    defaults = self.derive_defaults(camera, x, y, width, height, ow, oh)
                    self.save_defaults(defaults)
                    return defaults
        return None

    @staticmethod
    def derive_defaults(camera, x, y, width, height, ow, oh):
        sw, sh = camera['width'], camera['height']
        return {'version': 1, 'width': ow, 'height': oh,
                'zoom': min(sh, sw/(ow/oh))/height,
                'x': x/(sw-width)*100 if sw-width > 1e-6 else 50,
                'y': y/(sh-height)*100 if sh-height > 1e-6 else 50}

    def save_defaults(self, defaults):
        self.defaults_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.defaults_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        temp.write_text(json.dumps(defaults, indent=2) + '\n')
        temp.replace(self.defaults_path)

    def locate(self, identifier):
        if not re.fullmatch(r'[0-9a-f]{32}', identifier):
            raise HTTPException(404, 'Unknown capture')
        path = self.root / identifier
        index = path / 'index.json'
        if not index.is_file():
            raise HTTPException(404, 'Unknown capture')
        folder = (path / json.loads(index.read_text())['artifact_dir']).resolve()
        if not folder.is_relative_to(path.resolve()):
            raise HTTPException(404, 'Invalid capture path')
        return folder

    def snapshot(self, identifier):
        folder = self.locate(identifier)
        data = json.loads((folder / 'hand_joints.json').read_text())
        index = json.loads((self.root / identifier / 'index.json').read_text())
        return {'id': identifier, 'source_capture_id': index.get('source_capture_id'),
                'camera': data['camera'], 'mano': data['mano'],
                'crop': data.get('crop'), 'calibration_status': calibration_status(data),
                'capture': data['capture'], 'files': {
                    name: f'/api/captures/{identifier}/files/{name}' for name in sorted(FILES)}}

    def latest(self):
        indices = sorted(self.root.glob('*/index.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        if not indices:
            raise HTTPException(404, 'No saved capture yet.')
        return self.snapshot(indices[0].parent.name)

    def crop(self, identifier, parameters):
        from .capture_crop import crop_bundle
        source = self.locate(identifier)
        parameters = dict(parameters)
        defaults = parameters.pop('defaults', None)
        data = json.loads((source / 'hand_joints.json').read_text())
        if defaults is not None:
            camera = data['camera']; sw, sh = camera['width'], camera['height']
            height = min(sh, sw/(defaults['width']/defaults['height']))/defaults['zoom']
            width = height*defaults['width']/defaults['height']
            expected = {'x': (sw-width)*defaults['x']/100, 'y': (sh-height)*defaults['y']/100,
                        'width': width, 'height': height, 'output_width': defaults['width'],
                        'output_height': defaults['height']}
            if any(not math.isclose(parameters[key], value, abs_tol=1e-6) for key,value in expected.items()):
                raise ValueError('Saved defaults must match the applied crop.')
        new_id = uuid.uuid4().hex
        root = self.root / new_id
        artifact = root / 'crop'
        crop_bundle(source, artifact, **parameters)
        (root / 'index.json').write_text(json.dumps({'artifact_dir': 'crop', 'source_capture_id': identifier}))
        if defaults is None:
            defaults = self.derive_defaults(data['camera'], parameters['x'], parameters['y'],
                       parameters['width'], parameters['height'], parameters['output_width'], parameters['output_height'])
        self.save_defaults(defaults)
        return self.snapshot(new_id)

    def capture(self):
        if not self.lock.acquire(blocking=False):
            raise HTTPException(409, 'A robot-camera capture is already running.')
        identifier = uuid.uuid4().hex
        folder = self.root / identifier
        try:
            if not self.runner.is_file():
                raise HTTPException(503, f'Capture runner is missing: {self.runner}')
            folder.mkdir(parents=True)
            with (folder / 'capture.log').open('w') as log:
                result = subprocess.run([str(self.runner), '--output_dir', str(folder.resolve())],
                                        stdout=log, stderr=subprocess.STDOUT, timeout=90)
            if result.returncode:
                detail = (folder / 'capture.log').read_text()[-2500:]
                raise HTTPException(503, 'Capture failed. Keep arm/fingers stationary; start the arm '
                                    'controller and hand bridge, and close other camera/overlay listeners.\n' + detail)
            documents = list(folder.glob('*/hand_joints.json'))
            if len(documents) != 1:
                raise HTTPException(503, 'Capture produced no unique joint/image bundle.')
            artifact = documents[0].parent
            if not all((artifact / name).is_file() for name in FILES):
                raise HTTPException(503, 'Capture bundle is incomplete.')
            from .capture_guidance import load_guidance
            load_guidance(artifact / 'hand_joints.json', artifact / 'undistorted.png')
            (folder / 'index.json').write_text(json.dumps({'artifact_dir': artifact.name}))
            return self.snapshot(identifier)
        except subprocess.TimeoutExpired:
            raise HTTPException(504, 'Camera capture timed out; no robot commands were sent.')
        finally:
            self.lock.release()
