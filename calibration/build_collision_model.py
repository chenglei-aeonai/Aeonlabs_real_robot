"""Build portable plane-distance support geometry from a fully expanded URDF.

Only collision geometry is used. Convex-hull vertices preserve the minimum of
any linear plane-distance function exactly; no mesh decimation/rounding is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as R


def origin_matrix(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
        result[:3, :3] = R.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return result


def stl_vertices(path):
    data = path.read_bytes()
    if len(data) >= 84 and len(data) == 84 + 50 * struct.unpack_from('<I', data, 80)[0]:
        triangles = np.frombuffer(data, dtype=np.dtype([('normal', '<f4', (3,)),
                    ('vertices', '<f4', (3, 3)), ('attribute', '<u2')]), offset=84)
        vertices = triangles['vertices'].reshape(-1, 3).astype(float)
    else:
        rows = [line.split()[1:] for line in data.decode('ascii').splitlines()
                if line.strip().startswith('vertex ')]
        vertices = np.asarray(rows, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError(f'Invalid/empty STL: {path}')
    vertices = np.unique(vertices, axis=0)
    try:
        return vertices[ConvexHull(vertices).vertices]
    except QhullError:
        return vertices  # Planar/degenerate geometry: retaining all vertices is exact.


def build(urdf_path, output):
    source = Path(urdf_path).resolve()
    root = ET.parse(source).getroot()
    model = {'version': 1, 'reference_link': 'base', 'urdf_name': root.get('name'),
             'source_urdf': str(source), 'source_urdf_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
             'method': 'Collision-mesh convex-hull support vertices; exact signed-plane minimum for these meshes.',
             'links': [], 'joints': [], 'meshes': []}
    for link in root.findall('link'):
        entries = []
        for collision in link.findall('collision'):
            geometry = collision.find('geometry')
            mesh = geometry.find('mesh')
            if mesh is None:
                raise ValueError(f'Unsupported collision geometry in {link.get("name")}')
            filename = mesh.get('filename')
            if '://' in filename:
                raise ValueError('Resolve package mesh paths before building the model.')
            path = (source.parent / filename).resolve()
            vertices = stl_vertices(path)
            scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
            if scale.shape != (3,) or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('Invalid mesh scale')
            transform = origin_matrix(collision.find('origin'))
            vertices = (vertices * scale) @ transform[:3, :3].T + transform[:3, 3]
            entries.extend(vertices.tolist())
            model['meshes'].append({'link': link.get('name'), 'source': filename,
                                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        model['links'].append({'name': link.get('name'), 'vertices': entries})
    for joint in root.findall('joint'):
        axis = joint.find('axis')
        model['joints'].append({'name': joint.get('name'), 'type': joint.get('type'),
            'parent': joint.find('parent').get('link'), 'child': joint.find('child').get('link'),
            'origin': origin_matrix(joint.find('origin')).tolist(),
            'axis': np.fromstring(axis.get('xyz'), sep=' ').tolist() if axis is not None else None})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, separators=(',', ':')) + '\n')
    print(f'Saved {output}: {len(model["meshes"])} collision meshes, '
          f'{sum(len(link["vertices"]) for link in model["links"])} support vertices.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--urdf', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'models' / 'ur5e_dg5f_collision.json')
    args = parser.parse_args()
    build(args.urdf, args.output)
