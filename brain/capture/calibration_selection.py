"""Select and validate a complete legacy calibration pair without camera dependencies."""
import hashlib
import io
from pathlib import Path
import re
import numpy as np

def rigid_matrix(path):
    matrix = np.load(io.BytesIO(path) if isinstance(path, bytes) else path, allow_pickle=False)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1)):
        raise ValueError('Invalid rigid calibration transform')
    return matrix


def load_calibration(directory, name, serial):
    """Select the newest timestamp for which both legacy matrices exist."""
    suffix = f'_base_pose_robot_{name}.npy'
    for base_path in sorted(directory.glob(f'*{suffix}'), reverse=True):
        stamp = base_path.name[:-len(suffix)]
        if not re.fullmatch(r'\d{8}_\d{6}', stamp):
            continue
        camera_path = directory / f'{stamp}_camera_extrinsic_{serial}.npy'
        if camera_path.is_file():
            break
    else:
        base_path = directory / f'base_pose_robot_{name}.npy'
        camera_path = directory / f'camera_extrinsic_{serial}.npy'
    base_bytes, camera_bytes = base_path.read_bytes(), camera_path.read_bytes()
    world_from_base = rigid_matrix(base_bytes)
    world_from_camera = rigid_matrix(camera_bytes)
    provenance = {}
    for key, path, payload in [('world_from_base', base_path, base_bytes), ('world_from_camera', camera_path, camera_bytes)]:
        provenance[key] = {'path': str(path.resolve()),
                           'sha256': hashlib.sha256(payload).hexdigest()}
    return np.linalg.inv(world_from_camera) @ world_from_base, provenance
