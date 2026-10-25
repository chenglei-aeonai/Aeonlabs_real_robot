"""Shared MANO-right wrist convention for image editing and diffusion."""
import numpy as np

# dpl/representations.py at 48575bd; same convention as aeon-webui's
# runtime/uplift/custom_frame.py::wrist_from_geometry. The measured rounded
# signature is projected onto SO(3), so the saved wrist_R is a proper rotation.
RIGHT_SIGNATURE = np.array([[-1., -.02, -.04], [-.04, .07, 1.], [-.02, 1., -.07]])


def wrist_rotation(joints):
    joints=np.asarray(joints,dtype=np.float64)
    if joints.shape!=(21,3) or not np.isfinite(joints).all():
        raise ValueError('Wrist rotation requires 21 finite camera-space MANO joints.')
    forward = joints[4] - joints[0]
    across = joints[7] - joints[1]
    if np.linalg.norm(forward) < 1e-7:
        raise ValueError('Degenerate wrist-to-middle-MCP vector.')
    forward /= np.linalg.norm(forward)
    normal = np.cross(across, forward)
    if np.linalg.norm(normal) < 1e-7:
        raise ValueError('Degenerate hand palm plane.')
    normal /= np.linalg.norm(normal)
    side = np.cross(normal, forward)
    geometric = np.stack([forward, side, normal])
    u, _, vt = np.linalg.svd(RIGHT_SIGNATURE)
    signature = u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt
    return geometric.T @ signature


def validate_mano_right(joints, rotation):
    """Check DPL's chirality-aware MANO-right axes and camera/wrist roundtrip."""
    if (not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-8)
            or not np.isclose(np.linalg.det(rotation),1,atol=1e-8)):
        raise ValueError('MANO wrist rotation is not a proper right-handed rotation.')
    local=(joints[1:]-joints[0])@rotation
    forward=local[3]/np.linalg.norm(local[3])  # dpl middle MCP index 4 -> local index 3
    across=local[6]-local[0]                 # pinky MCP minus index MCP
    normal=np.cross(across,forward);normal/=np.linalg.norm(normal)
    side=np.cross(normal,forward)
    observed=np.stack([forward,side,normal])
    u,_,vt=np.linalg.svd(RIGHT_SIGNATURE)
    signature=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
    angle=float(np.degrees(np.arccos(np.clip((np.trace(observed@signature.T)-1)/2,-1,1))))
    reconstructed=local@rotation.T+joints[0]
    error=float(np.max(np.linalg.norm(reconstructed-joints[1:],axis=-1)))
    if angle>1e-3 or error>1e-8:
        raise ValueError(f'MANO-right wrist convention failed: axis error {angle:.6f} deg; roundtrip {error:.3g} m.')
    return {'mano_right_axes_error_deg':angle,'camera_roundtrip_max_error_m':error,
            'metric_scope':'construction self-consistency; not physical pose or image alignment accuracy',
            'axes_error_definition':'acos(clip((trace(observed_axes @ reference_axes.T)-1)/2,-1,1)), degrees; reference is nearest SO(3) of DPL right signature',
            'roundtrip_error_definition':'max over non-wrist joints of ||reconstructed_camera_joint - input_camera_joint||_2, metres',
            'thresholds':{'mano_right_axes_error_deg':1e-3,'camera_roundtrip_max_error_m':1e-8},
            'rotation_direction':'MANO-right wrist frame to optical camera frame',
            'row_vector_equation':'J_camera = J_wrist @ wrist_R.T + wrist_position_camera',
            'chirality':'right; normal = (pinky_MCP - index_MCP) cross (middle_MCP - wrist)'}
