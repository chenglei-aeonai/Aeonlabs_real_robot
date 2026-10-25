"""URDF collision-surface clearance against a fixed robot-base table plane.

Uses every plane-support vertex of each collision mesh, including collision
origins/scales. This is a model check, not a claim of real-world collision safety.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


class SurfaceError(RuntimeError):
    pass


MODEL_PATH = Path(__file__).parent / 'models' / 'ur5e_dg5f_collision.json'
MODEL_SHA256 = '92acf86aa21f99c19cfb970e2caeb3ecbda13ebba5885dd4464dbd7c9fbdd5d7'
ARM_NAMES = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
             'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
HAND_NAMES = tuple(f'rj_dg_{finger}_{joint}' for finger in range(1, 6) for joint in range(1, 5))


class RobotSurfaceModel:
    def __init__(self, path=MODEL_PATH, expected_sha256=MODEL_SHA256):
        raw = Path(path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        if self.sha256 != expected_sha256:
            raise SurfaceError('Collision model checksum mismatch; refusing unreviewed geometry.')
        data = json.loads(raw)
        if data.get('version') != 1 or data.get('reference_link') != 'base':
            raise SurfaceError('Unsupported collision model/reference frame.')
        self.links = {}
        for link in data['links']:
            name = link['name']
            points = np.asarray(link['vertices'], dtype=float).reshape(-1, 3)
            if name in self.links or not np.isfinite(points).all():
                raise SurfaceError('Duplicate link or nonfinite mesh support geometry.')
            self.links[name] = points
        self.joints = []
        children = set()
        for joint in data['joints']:
            if joint['child'] in children or joint['type'] not in ('fixed', 'revolute', 'continuous'):
                raise SurfaceError('Invalid URDF tree/joint type.')
            children.add(joint['child'])
            origin = np.asarray(joint['origin'], dtype=float)
            if origin.shape != (4, 4) or not np.isfinite(origin).all():
                raise SurfaceError('Invalid collision/joint transform.')
            self.joints.append((joint, origin))
        roots = set(self.links) - children
        if roots != {'base_link'} or not set(ARM_NAMES + HAND_NAMES).issubset({j[0]['name'] for j in self.joints}):
            raise SurfaceError('Collision model is not the expected UR5e + right DG5F.')
        dynamic = set()
        pending=list(self.joints)
        resolved={'base_link'}
        while pending:
            remaining=[]
            for joint,_ in pending:
                if joint['parent'] not in resolved:
                    remaining.append((joint,_));continue
                if joint['type'] != 'fixed' or joint['parent'] in dynamic:
                    dynamic.add(joint['child'])
                resolved.add(joint['child'])
            if len(remaining)==len(pending):raise SurfaceError('Invalid collision tree.')
            pending=remaining
        self.moving_links=dynamic
        self.mesh_links = [(name, points) for name, points in self.links.items() if len(points)]
        if len(self.mesh_links) != 35:
            raise SurfaceError('Missing arm/hand collision meshes.')

    def transforms(self, arm_q, hand_q, tcp_pose=None, tcp_from_hand=None):
        arm_q, hand_q = np.asarray(arm_q, dtype=float), np.asarray(hand_q, dtype=float)
        if arm_q.shape != (6,) or hand_q.shape != (20,) or not np.isfinite(arm_q).all() or not np.isfinite(hand_q).all():
            raise SurfaceError('Surface checks require six arm and twenty measured hand angles.')
        angles = dict(zip(ARM_NAMES + HAND_NAMES, np.r_[arm_q, hand_q]))
        frames = {'base_link': np.eye(4)}
        pending = list(self.joints)
        override = None
        if tcp_from_hand is not None:
            if tcp_pose is None:
                raise SurfaceError('Configured TCP pose missing for custom hand mount.')
            # root->base is the fixed pi rotation in the UR model. Resolve it
            # from the tree below rather than assuming the frame convention.
            tcp = np.eye(4)
            tcp[:3, :3] = R.from_quat(tcp_pose[3:]).as_matrix()
            tcp[:3, 3] = tcp_pose[:3]
            override = tcp @ np.asarray(tcp_from_hand)
        while pending:
            remaining = []
            for joint, origin in pending:
                if joint['parent'] not in frames or (joint['child'] == 'rl_dg_mount' and override is not None and 'base' not in frames):
                    remaining.append((joint, origin))
                    continue
                local = origin.copy()
                if joint['type'] != 'fixed':
                    axis = np.asarray(joint['axis'], dtype=float)
                    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                        raise SurfaceError('Invalid URDF joint axis.')
                    local[:3, :3] = local[:3, :3] @ R.from_rotvec(axis / np.linalg.norm(axis) * angles[joint['name']]).as_matrix()
                frames[joint['child']] = (frames['base'] @ override if joint['child'] == 'rl_dg_mount' and override is not None
                                           else frames[joint['parent']] @ local)
            if len(remaining) == len(pending):
                raise SurfaceError('Unresolved/cyclic URDF collision tree.')
            pending = remaining
        base_from_root = np.linalg.inv(frames['base'])
        return {name: base_from_root @ transform for name, transform in frames.items()}

    def clearance(self, arm_q, hand_q, limits, tcp_pose, tcp_from_hand=None):
        tcp_pose = np.asarray(tcp_pose, dtype=float)
        if tcp_pose.shape != (7,) or not np.isfinite(tcp_pose).all() or abs(np.linalg.norm(tcp_pose[3:]) - 1.) > .001:
            raise SurfaceError('Surface check requires a finite configured TCP pose and unit quaternion.')
        transforms = self.transforms(arm_q, hand_q, tcp_pose, tcp_from_hand)
        best = None
        excluded = []
        max_hand_radius = 0.
        padding = limits.reserve + limits.data['clearance_m']
        plane_only=getattr(limits,'plane_only',False)
        if not plane_only:
            lo = np.asarray(limits.data['table_min_xy_m']) - padding
            hi = np.asarray(limits.data['table_max_xy_m']) + padding
        fixed=[]
        for name, vertices in self.mesh_links:
            if plane_only and name not in self.moving_links:
                fixed.append(name);continue
            transform = transforms[name]
            points = vertices @ transform[:3, :3].T + transform[:3, 3]
            if name.startswith('rl_dg_'):
                max_hand_radius = max(max_hand_radius, float(np.max(np.linalg.norm(points - tcp_pose[:3], axis=1))))
            # A whole-link XY bounding box is conservative. If it overlaps the
            # expanded table footprint, use the minimum over the ENTIRE mesh.
            # This may refuse a clear pose, but cannot overstate its clearance.
            if not plane_only and ((points[:, :2].max(axis=0) < lo).any() or (points[:, :2].min(axis=0) > hi).any()):
                excluded.append(name)
                continue
            signed = points @ limits.plane_normal + limits.plane_offset
            index = int(np.argmin(signed))
            report = {'link': name, 'clearance_m': float(signed[index]), 'point_base_m': points[index].tolist()}
            if best is None or report['clearance_m'] < best['clearance_m']:
                best = report
        # Printed wrist target: 92mm square, using the existing measured
        # ADAPTER_TO_EE_POSE convention. Backing/cables remain model uncertainty.
        if plane_only:
            board=np.array([[x,-.05,z] for x in (-.046,.046) for z in (-.046,.046)])
            points=board @ R.from_quat(tcp_pose[3:]).as_matrix().T + tcp_pose[:3]
            signed=points @ limits.plane_normal + limits.plane_offset
            index=int(np.argmin(signed))
            if best is None or signed[index]<best['clearance_m']:
                best={'link':'wrist_calibration_board','clearance_m':float(signed[index]),'point_base_m':points[index].tolist()}
        if best is None:
            raise SurfaceError('No robot collision surfaces overlap the table footprint; check the geometry.')
        return dict(best, model_sha256=self.sha256, excluded_links_outside_table_xy=excluded,
                    fixed_mount_links_excluded=fixed,
                    max_modeled_hand_radius_from_tcp_m=max_hand_radius,
                    required_clearance_m=padding, remaining_clearance_m=best['clearance_m'] - padding)

    def check(self, arm_q, hand_q, limits, tcp_pose, tcp_from_hand=None):
        report = self.clearance(arm_q, hand_q, limits, tcp_pose, tcp_from_hand)
        if not getattr(limits,'plane_only',False) and report['max_modeled_hand_radius_from_tcp_m'] > limits.data['tool_radius_m'] + 1e-9:
            raise SurfaceError('Configured tool sphere does not enclose the modeled hand surfaces.')
        if report['remaining_clearance_m'] < 0:
            raise SurfaceError(f"URDF collision surface {report['link']} is only {report['clearance_m']:.4f}m above the table plane; requires {report['required_clearance_m']:.4f}m including reserves.")
        return report
