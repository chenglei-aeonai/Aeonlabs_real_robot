"""Synthetic geometry for tests only; never a physical robot configuration."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R
from overlay_utils import ur5e_transforms


def settings():
    data = json.loads((Path(__file__).resolve().parents[1] / 'safety.example.json').read_text())
    data.update(verified=True, robot_safety_setup_verified=True,
                verification_notes='SYNTHETIC TEST FIXTURE - NOT FOR HARDWARE',
                workspace_min_m=[-1.5, -1.5, -1.5], workspace_max_m=[1.5, 1.5, 3.],
                table_min_xy_m=[-2., -2.], table_max_xy_m=[2., 2.], table_top_z_m=-2.,
                tool_radius_m=.35, arm_link_radii_m=[.15] * 6, max_downward_offset_m=0.,
                table_plane={'frame':'robot_base','normal':[0.,0.,1.],'offset_m':2.})
    return data


def start_state():
    q = np.array([-1.5708, -1.05, 1.95, -.9, 1.571, -2.618])
    t = ur5e_transforms(q)[-1]
    return {'pos': np.r_[t[:3, 3], R.from_matrix(t[:3, :3]).as_quat()].tolist(),
            'Q': q.tolist(), 'Qd': [0.] * 6, 'vel': [0.] * 6,
            'is_truncated': False, 'protective_stopped': False, 'emergency_stopped': False}
