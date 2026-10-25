"""Conservative calibration geometry, deterministic planning, and supervised motion.

Application checks complement the robot's independently configured safety system.
No table geometry is inferred from the calibration being measured.
"""
import json
from pathlib import Path
import time
import uuid

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R, Slerp

from overlay_utils import ur5e_transforms
from surface_guard import RobotSurfaceModel, SurfaceError


class SafetyError(RuntimeError):
    pass


class SampleSkipped(SafetyError):
    """A soft planning boundary was approached; stop then reconsider next sample."""
    pass


class SafetyLimits:
    def __init__(self, data):
        self.data = data
        self.plane_only = data.get('geometry_mode', 'workspace') == 'plane_only'
        if data.get('geometry_mode', 'workspace') not in ('workspace', 'plane_only'):
            raise SafetyError('Unknown geometry_mode.')
        if self.plane_only and data.get('allow_estimated_plane') is not True:
            raise SafetyError('Plane-only mode requires explicit allow_estimated_plane acknowledgement.')
        if not self.plane_only and (data.get('verified') is not True or data.get('robot_safety_setup_verified') is not True):
            raise SafetyError('Safety configuration is not verified. Measure table/workspace/tool geometry and verify robot safety settings first.')
        if not self.plane_only and (not isinstance(data.get('verification_notes'), str) or not data['verification_notes'].strip()):
            raise SafetyError('Record the independent measurement and safety-setup verification in verification_notes.')
        for key, size in [('workspace_min_m', 3), ('workspace_max_m', 3),
                          ('table_min_xy_m', 2), ('table_max_xy_m', 2), ('arm_link_radii_m', 6)] if not self.plane_only else []:
            value = data.get(key)
            if (not isinstance(value, list) or len(value) != size
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in value)):
                raise SafetyError(f'{key} requires {size} independently verified finite numbers.')
        keys = ['table_top_z_m', 'tool_radius_m', 'clearance_m', 'uncertainty_m',
                'stopping_margin_m', 'max_linear_speed_m_s', 'max_angular_speed_deg_s',
                'max_linear_acceleration_m_s2', 'max_angular_acceleration_deg_s2',
                'command_timeout_s', 'state_timeout_s', 'motion_timeout_s',
                'max_joint_deviation_deg', 'max_joint_speed_deg_s', 'max_start_translation_m',
                'tilt_deg', 'position_tolerance_m', 'angle_tolerance_deg', 'settle_s',
                'max_reprojection_error_px']
        if self.plane_only:
            keys = [key for key in keys if key not in ('table_top_z_m','tool_radius_m')]
        for key in keys:
            value = data.get(key)
            if type(value) not in (int, float) or not np.isfinite(value):
                raise SafetyError(f'Missing or nonfinite safety value: {key}')
            if key != 'table_top_z_m' and value <= 0:
                raise SafetyError(f'{key} must be positive.')
        caps = {'max_linear_speed_m_s': .01, 'max_angular_speed_deg_s': 2.,
                'max_linear_acceleration_m_s2': .02, 'max_angular_acceleration_deg_s2': 4.,
                'command_timeout_s': .2, 'state_timeout_s': .2, 'max_start_translation_m': .08,
                'tilt_deg': 3., 'max_joint_deviation_deg': 20., 'max_joint_speed_deg_s': 5.,
                'position_tolerance_m': .002, 'angle_tolerance_deg': .3}
        for key, cap in caps.items():
            if data[key] > cap:
                raise SafetyError(f'{key} exceeds calibration cap {cap}.')
        self.tcp_speed_trip_factor = data.get('measured_tcp_speed_trip_factor', 2.)
        self.joint_speed_trip_factor = data.get('measured_joint_speed_trip_factor', 1.5)
        for name, value, maximum in [('TCP',self.tcp_speed_trip_factor,2.),('joint',self.joint_speed_trip_factor,1.5)]:
            if type(value) not in (int,float) or not np.isfinite(value) or not 1.25 < value <= maximum:
                raise SafetyError(f'{name} measured-speed trip factor must be above 1.25 and at most {maximum}.')
        self.joint_planning_margin = data.get('joint_planning_margin_deg', 2.)
        if (type(self.joint_planning_margin) not in (int,float) or not np.isfinite(self.joint_planning_margin)
                or self.joint_planning_margin < .5 or self.joint_planning_margin >= data['max_joint_deviation_deg']):
            raise SafetyError('Joint planning margin must be finite, at least 0.5deg, and smaller than the runtime joint limit.')
        if data['command_timeout_s'] < .08 or data['state_timeout_s'] < .08:
            raise SafetyError('Communication timeouts must be between 0.08 and 0.2 s.')
        if data['stopping_margin_m'] < .01 or data['uncertainty_m'] < .01 or data['clearance_m'] < .02:
            raise SafetyError('Minimum reserves: 10mm stopping, 10mm geometry uncertainty, 20mm clearance; increase after commissioning as needed.')
        if not self.plane_only and (data['tool_radius_m'] < .1 or min(data['arm_link_radii_m']) < .05):
            raise SafetyError('Tool radius must be >=0.1m and arm radii >=0.05m; all must enclose the actual geometry.')
        if data['settle_s'] < .5:
            raise SafetyError('Require at least 0.5s of measured stationarity.')
        if not 3 <= data.get('samples_per_pose', 0) <= 30 or type(data.get('samples_per_pose')) is not int:
            raise SafetyError('samples_per_pose must be an integer between 3 and 30.')
        if not self.plane_only:
            self.lower = np.array(data['workspace_min_m'], dtype=float)
            self.upper = np.array(data['workspace_max_m'], dtype=float)
            if (self.lower >= self.upper).any() or (np.array(data['table_min_xy_m']) >= data['table_max_xy_m']).any():
                raise SafetyError('Workspace/table minima must be below maxima.')
            if (self.lower[:2] < data['table_min_xy_m']).any() or (self.upper[:2] > data['table_max_xy_m']).any():
                raise SafetyError('Approved workspace XY must be inside the measured table footprint.')
        plane = data.get('table_plane')
        if not isinstance(plane, dict) or plane.get('frame') != 'robot_base':
            raise SafetyError('table_plane must be explicitly defined in the fixed robot_base frame.')
        normal = plane.get('normal')
        offset = plane.get('offset_m')
        if (not isinstance(normal, list) or len(normal) != 3
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in normal)
                or type(offset) not in (int, float) or not np.isfinite(offset)):
            raise SafetyError('table_plane requires three finite normal components and a finite offset_m.')
        self.plane_normal = np.asarray(normal, dtype=float)
        self.plane_offset = float(offset)
        if abs(np.linalg.norm(self.plane_normal) - 1.) > 1e-6 or self.plane_normal[2] <= 1e-6:
            raise SafetyError('table_plane normal must be unit length and point upward in robot_base.')
        # Preserve the existing higher conservative table/fixture box. A tilted
        # plane may never silently reduce that already-protected forbidden volume.
        if not self.plane_only:
            heights = [-(normal[0]*x + normal[1]*y + offset) / normal[2]
                       for x in (data['table_min_xy_m'][0], data['table_max_xy_m'][0])
                       for y in (data['table_min_xy_m'][1], data['table_max_xy_m'][1])]
            if max(heights) > data['table_top_z_m'] + 1e-9:
                raise SafetyError('table_top_z_m must enclose the table plane over the entire table footprint, including fixtures.')
        self.max_downward_offset = data.get('max_downward_offset_m', 0.)
        if type(self.max_downward_offset) not in (int, float) or not np.isfinite(self.max_downward_offset) or not 0 <= self.max_downward_offset <= .05:
            raise SafetyError('max_downward_offset_m must be between 0 and 0.05m.')
        self.reserve = data['uncertainty_m'] + data['stopping_margin_m']
        self.radius = (0. if self.plane_only else data['tool_radius_m']) + self.reserve
        if not self.plane_only and (self.lower + self.radius >= self.upper - self.radius).any():
            raise SafetyError('Workspace is too small for the conservative tool envelope.')

    @classmethod
    def load(cls, path):
        try:
            return cls(json.loads(Path(path).read_text()))
        except (OSError, ValueError) as exc:
            raise SafetyError(f'Cannot load verified safety configuration {path}: {exc}') from exc

    def tool_plane_clearance(self, pose):
        """Metres from the physical tool-sphere surface to the fixed table plane."""
        return float(self.plane_normal @ np.asarray(pose[:3], dtype=float)
                     + self.plane_offset - (0. if self.plane_only else self.data['tool_radius_m']))

    def check_pose(self, pose, anchor=None, measured=False):
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:]) - 1) > .001:
            raise SafetyError('Invalid TCP pose/quaternion.')
        p = pose[:3]
        if self.tool_plane_clearance(pose) < self.data['clearance_m'] + self.reserve:
            raise SafetyError('Full hand/board envelope violates the fixed robot-base table-plane clearance.')
        if not self.plane_only:
            if (p - self.radius < self.lower).any() or (p + self.radius > self.upper).any():
                raise SafetyError('Full tool envelope would leave the approved workspace/table area.')
            if p[2] - self.radius < self.data['table_top_z_m'] + self.data['clearance_m']:
                raise SafetyError('Insufficient table clearance for the entire hand/board envelope.')
        if anchor is not None:
            if np.linalg.norm(p - np.asarray(anchor[:3])) > self.data['max_start_translation_m']:
                raise SafetyError('TCP leaves the small approved neighbourhood of the starting position.')
            tolerance = self.data['position_tolerance_m'] if measured else 1e-9
            if p[2] < anchor[2] - self.max_downward_offset - tolerance:
                raise SafetyError('Calibration exceeds the configured downward offset from its initial TCP pose.')
            angle = (R.from_quat(anchor[3:]).inv() * R.from_quat(pose[3:])).magnitude()
            allowed = self.data['tilt_deg'] + (self.data['angle_tolerance_deg'] if measured else 0.)
            if angle > np.deg2rad(allowed) + 1e-8:
                raise SafetyError('Orientation exceeds the approved absolute tilt.')

    def check_arm(self, q, anchor_q=None):
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise SafetyError('Missing or invalid measured arm joints.')
        if anchor_q is not None and np.max(np.abs(q - anchor_q)) > np.deg2rad(self.data['max_joint_deviation_deg']):
            raise SafetyError('Arm joints leave the approved local neighbourhood.')
        if (np.abs(q) > np.array([2, 2, 1, 2, 2, 2]) * np.pi).any():
            raise SafetyError('Arm exceeds conservative URDF joint limits.')
        if self.plane_only:
            return  # Full moving-link surfaces are checked by the URDF guard.
        points = ur5e_transforms(q)[:, :3, 3]
        # Expanded AABB against each arm capsule is conservative. The table
        # footprint includes the real full table, independently of the smaller
        # region approved for calibration. The solid below the top is forbidden.
        for index, (a, b) in enumerate(zip(points[:-1], points[1:])):
            radius = self.data['arm_link_radii_m'][index] + self.reserve + self.data['clearance_m']
            lower = np.r_[np.asarray(self.data['table_min_xy_m']) - radius, -1e6]
            upper = np.r_[np.asarray(self.data['table_max_xy_m']) + radius,
                          self.data['table_top_z_m'] + radius]
            enter, leave = 0., 1.
            for axis in range(3):
                delta = b[axis] - a[axis]
                if abs(delta) < 1e-12:
                    if a[axis] < lower[axis] or a[axis] > upper[axis]:
                        enter, leave = 1., 0.
                        break
                else:
                    ends = sorted(((lower[axis] - a[axis]) / delta, (upper[axis] - a[axis]) / delta))
                    enter, leave = max(enter, ends[0]), min(leave, ends[1])
            if enter <= leave:
                raise SafetyError(f'Arm link {index + 1} capsule approaches/intersects the table volume.')


def pose_matrix(pose):
    t = np.eye(4)
    t[:3, :3] = R.from_quat(pose[3:]).as_matrix()
    t[:3, 3] = pose[:3]
    return t


def plan_poses(start, limits, locations=1):
    """First location is current TCP. Remaining locations match old non-downward offsets."""
    offsets = [(0, 0, 0), (.05, .05, 0), (.05, -.05, 0),
               (-.05, -.05, 0), (-.05, .05, 0), (.05, 0, .05), (-.05, 0, .05)]
    if not 1 <= locations <= len(offsets):
        raise SafetyError('locations must be between 1 and 7.')
    result = []
    for location, offset in enumerate(offsets[:locations], start=1):
        neutral = np.asarray(start, dtype=float).copy()
        neutral[:3] += offset
        result.append((f'location_{location}_neutral', neutral.copy(), True))
        for axis in ('x', 'y'):
            for sign in (1, -1):
                tilted = neutral.copy()
                tilted[3:] = (R.from_quat(neutral[3:]) * R.from_euler(axis, sign * limits.data['tilt_deg'], degrees=True)).as_quat()
                result.append((f'location_{location}_{axis}_{sign:+d}', tilted, True))
                # These neutral waypoints are motion-only: never travel tilted.
                result.append((f'location_{location}_return_neutral', neutral.copy(), False))
    for _, pose, _ in result:
        limits.check_pose(pose, start)
    return result


def preflight(plan, state, limits, surface_model=None, hand_q=None, tcp_from_hand=None, anchor_state=None):
    """Check a dense nominal IK path, arm capsules, and tool envelope before arming."""
    try:
        model = surface_model if surface_model is not None else RobotSurfaceModel()
        initial_surface = model.check(state['Q'], hand_q, limits, state['pos'], tcp_from_hand)
    except (OSError, ValueError, SurfaceError) as exc:
        raise SafetyError(f'Cannot approve URDF surface clearance: {exc}') from exc
    start = np.asarray(state['pos'], dtype=float)
    anchor_state = state if anchor_state is None else anchor_state
    anchor_pose = np.asarray(anchor_state['pos'], dtype=float)
    q0 = np.asarray(anchor_state['Q'], dtype=float)
    current_q = np.asarray(state['Q'], dtype=float)
    limits.check_pose(start, anchor_pose)
    limits.check_arm(current_q, q0)
    # Derive the installed TCP from measured pose and nominal FK; this is a
    # local kinematic approximation, covered by the verified uncertainty reserve.
    tool0_from_tcp = np.linalg.inv(ur5e_transforms(current_q)[-1]) @ pose_matrix(start)
    previous, seed = start.copy(), current_q.copy()
    path = []
    bound = np.deg2rad(limits.data['max_joint_deviation_deg'] - limits.joint_planning_margin)
    if np.max(np.abs(seed - q0)) > bound:
        raise SafetyError('Current joint configuration consumes the reserved planning headroom; restart planning from a suitable initial pose.')
    previous_surface = initial_surface
    for label, goal, capture in plan:
        closest_surface = previous_surface
        minimum_joint_headroom = limits.data['max_joint_deviation_deg'] - float(np.degrees(np.max(np.abs(seed-q0))))
        steps = max(1, int(np.ceil(np.linalg.norm(goal[:3] - previous[:3]) / .002)),
                    int(np.ceil(np.degrees((R.from_quat(previous[3:]).inv() * R.from_quat(goal[3:])).magnitude()) / .25)))
        rotations = Slerp([0, 1], R.from_quat([previous[3:], goal[3:]]))
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            pose = np.r_[previous[:3] + fraction * (goal[:3] - previous[:3]), rotations(fraction).as_quat()]
            desired = pose_matrix(pose)
            def residual(q):
                actual = ur5e_transforms(q)[-1] @ tool0_from_tcp
                return np.r_[actual[:3, 3] - desired[:3, 3],
                             (R.from_matrix(desired[:3, :3]).inv() * R.from_matrix(actual[:3, :3])).as_rotvec()]
            fit = least_squares(residual, seed, bounds=(q0 - bound, q0 + bound), max_nfev=80,
                                ftol=1e-10, xtol=1e-10, gtol=1e-10)
            singular_values = np.linalg.svd(fit.jac, compute_uv=False)
            if singular_values[-1] < 1e-4 or singular_values[0] / singular_values[-1] > 200:
                raise SafetyError(f'Near-singular local IK at {label}; refuse motion.')
            error = residual(fit.x)
            if not fit.success or np.linalg.norm(error[:3]) > .0005 or np.linalg.norm(error[3:]) > np.deg2rad(.1):
                raise SafetyError(f'Cannot preflight local IK for {label}; no motion permitted.')
            limits.check_pose(pose, anchor_pose)
            limits.check_arm(fit.x, q0)
            try:
                surface = model.check(fit.x, hand_q, limits, pose, tcp_from_hand)
            except SurfaceError as exc:
                raise SafetyError(f'Unsafe URDF surface along {label}: {exc}') from exc
            if surface['clearance_m'] < closest_surface['clearance_m']:
                closest_surface = surface
            seed = fit.x
            minimum_joint_headroom = min(minimum_joint_headroom, limits.data['max_joint_deviation_deg'] - float(np.degrees(np.max(np.abs(seed-q0)))))
        path.append({'label': label, 'tcp_pose': goal.tolist(), 'capture': capture, 'predicted_q': seed.tolist(),
                     'joint_planning_limit_deg': limits.data['max_joint_deviation_deg'] - limits.joint_planning_margin,
                     'predicted_joint_runtime_headroom_deg': minimum_joint_headroom,
                     'tool_plane_clearance_m': None if limits.plane_only else limits.tool_plane_clearance(goal),
                     'tcp_plane_distance_m': float(limits.plane_normal @ goal[:3] + limits.plane_offset),
                     'clearance_reference': 'tcp_point_and_urdf_surfaces' if limits.plane_only else 'tool_sphere',
                     'table_plane_reserve_remaining_m': limits.tool_plane_clearance(goal) - limits.reserve - limits.data['clearance_m'],
                     'closest_urdf_surface_on_segment': closest_surface})
        previous = goal
        previous_surface = surface
    return path


def translation_candidates(start, locations=9):
    """Current pose plus the original eight translation offsets; no random tilt."""
    offsets = [(0,0,0), (.05,.05,0), (.05,-.05,0), (-.05,-.05,0), (-.05,.05,0),
               (.05,0,.05), (.05,0,-.05), (-.05,0,-.05), (-.05,0,.05)]
    if not 1 <= locations <= len(offsets):
        raise SafetyError('Translation sampling supports 1 through 9 locations.')
    result=[]
    for index, offset in enumerate(offsets[:locations],start=1):
        pose=np.asarray(start,dtype=float).copy()
        pose[:3]+=offset
        result.append((f'translation_{index}',pose,True))
    return result


def select_valid_samples(candidates, state, limits, surface_model, hand_q, tcp_from_hand=None, minimum=5):
    """Filter fixed candidates offline, checking edges from the last accepted pose.

    Invalid initial geometry/feedback is fatal. No alternate positions are invented,
    limits never expand, and rejected poses never become a path's starting state.
    """
    preflight([],state,limits,surface_model,hand_q,tcp_from_hand,anchor_state=state)
    accepted, checked, rejected = [], [], []
    previous=dict(state)
    for candidate in candidates:
        label,goal,capture=candidate
        try:
            edge=preflight([candidate],previous,limits,surface_model,hand_q,tcp_from_hand,anchor_state=state)
        except SafetyError as exc:
            rejected.append({'label':label,'tcp_pose':np.asarray(goal).tolist(),'reason':str(exc)})
            continue
        accepted.append(candidate)
        checked.extend(edge)
        previous=dict(previous,pos=np.asarray(goal).tolist(),Q=edge[-1]['predicted_q'])
    if sum(item[2] for item in accepted) < minimum:
        reasons='; '.join(item['label']+': '+item['reason'] for item in rejected)
        raise SafetyError(f'Only {len(accepted)} valid samples remain; at least {minimum} required before motion. Rejected: {reasons}')
    return accepted,checked,rejected


class CalibrationMotion:
    """No background heartbeat: acquisition/review happens stopped, with guard armed."""
    def __init__(self, publisher, state_socket, limits, cancel=lambda: False):
        self.publisher, self.socket, self.limits, self.cancel = publisher, state_socket, limits, cancel
        self.session = None
        self.anchor = None
        self.anchor_q = None
        self.surface_check = None
        self._overspeed_since = None
        self._last_speed_warning_at = None

    def read(self, timeout_ms=100, check_cancel=True):
        if check_cancel and self.cancel():
            raise SafetyError('Cancelled by operator.')
        if not self.socket.poll(timeout_ms):
            raise SafetyError('Arm feedback missing; motion stopped.')
        state = self.socket.recv_json()
        age = time.time() - state.get('timestamp_ms', 0) / 1000
        if not -.05 <= age <= self.limits.data['state_timeout_s']:
            raise SafetyError('Stale arm feedback.')
        if any(type(state.get(key)) is not bool for key in ('is_truncated', 'protective_stopped', 'emergency_stopped')):
            raise SafetyError('Missing controller stop status.')
        if state.get('is_truncated') or state.get('protective_stopped') or state.get('emergency_stopped'):
            raise SafetyError('Robot reports a stop/fault.')
        guard = state.get('calibration_guard') or {}
        if not isinstance(guard, dict):
            raise SafetyError('Missing controller guard status.')
        if guard.get('version') != 5 or guard.get('config') != self.limits.data:
            raise SafetyError('Controller lacks the matching calibration guard; rebuild and use the calibration controller profile.')
        if guard.get('fault'):
            raise SafetyError('Controller guard latched: ' + guard['fault'])
        if self.session is not None and (guard.get('session') != self.session or not guard.get('armed')):
            raise SafetyError('Calibration controller session lost.')
        self.limits.check_pose(state['pos'], self.anchor, measured=True)
        self.limits.check_arm(state['Q'], self.anchor_q)
        velocity = np.asarray(state.get('vel'), dtype=float)
        qd = np.asarray(state.get('Qd'), dtype=float)
        if velocity.shape != (6,) or qd.shape != (6,) or not np.isfinite(velocity).all() or not np.isfinite(qd).all():
            raise SafetyError('Missing finite measured velocity feedback.')
        linear=float(np.linalg.norm(velocity[:3]))
        angular=float(np.degrees(np.linalg.norm(velocity[3:])))
        joint=float(np.degrees(np.max(np.abs(qd))))
        linear_cap=self.limits.data['max_linear_speed_m_s']
        angular_cap=self.limits.data['max_angular_speed_deg_s']
        joint_cap=self.limits.data['max_joint_speed_deg_s']
        if linear>linear_cap*self.limits.tcp_speed_trip_factor or angular>angular_cap*self.limits.tcp_speed_trip_factor:
            raise SafetyError(f'Severe measured TCP overspeed: {linear*1000:.1f}mm/s, {angular:.2f}deg/s; hard thresholds {linear_cap*self.limits.tcp_speed_trip_factor*1000:.1f}mm/s, {angular_cap*self.limits.tcp_speed_trip_factor:.2f}deg/s.')
        if joint>joint_cap*self.limits.joint_speed_trip_factor:
            raise SafetyError(f'Severe measured joint overspeed: {joint:.2f}deg/s; hard threshold {joint_cap*self.limits.joint_speed_trip_factor:.2f}deg/s.')
        warning=linear>linear_cap*1.25 or angular>angular_cap*1.25 or joint>joint_cap*1.25
        state['overspeed_duration_s']=0.
        if warning:
            now=time.monotonic()
            if self._overspeed_since is None:self._overspeed_since=now
            state['overspeed_duration_s']=now-self._overspeed_since
            if self._last_speed_warning_at is None or now-self._last_speed_warning_at>=1.:
                print(f'Warning: measured speed above nominal tolerance: TCP {linear*1000:.1f}mm/s, {angular:.2f}deg/s; max joint {joint:.2f}deg/s. Brief excursions do not abort calibration.')
                self._last_speed_warning_at=now
        else:
            self._overspeed_since=None
        if self.surface_check is not None:
            state['surface_clearance'] = self.surface_check(state)
        return state

    def arm(self):
        state = self.read()
        self.anchor, self.anchor_q = np.asarray(state['pos']), np.asarray(state['Q'])
        session = uuid.uuid4().hex
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            packet={'calibration_arm':session}
            if self.limits.plane_only:
                packet.update(hand_q=state['measured_hand_q'],hand_timestamp_ms=state['hand_timestamp_ms'])
            self.publisher.send_json(packet)
            state = self.read()
            guard = state['calibration_guard']
            if guard.get('session') == session and guard.get('armed'):
                self.session = session
                return
            time.sleep(.02)
        raise SafetyError('Controller did not acknowledge calibration arming.')

    def stop(self):
        for _ in range(3):
            self.publisher.send_json({'stop': True})
            time.sleep(.01)

    def wait_stopped(self, timeout=3.):
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline:
            state=self.read()
            if np.max(np.abs(state['Qd']))<.003 and np.linalg.norm(state['vel'][:3])<.001 and np.linalg.norm(state['vel'][3:])<.005:
                return state
            time.sleep(.02)
        raise SafetyError('Robot did not settle after skipping sample.')

    def move(self, target, predicted_q=None):
        self.limits.check_pose(target, self.anchor)
        deadline = time.monotonic() + self.limits.data['motion_timeout_s']
        settled_since = None
        try:
            while time.monotonic() < deadline:
                state = self.read()
                if state.get('overspeed_duration_s',0.)>=.5:
                    raise SampleSkipped('Measured overspeed persisted for 0.5s; stop this movement and consider the next sample.')
                if self.anchor_q is not None:
                    delta=np.abs(np.asarray(state['Q'])-self.anchor_q)
                    # Stop before the planning boundary, leaving room to stop
                    # inside both planning and hard runtime limits. A subsequent
                    # move toward the anchor is allowed after fresh preflight.
                    soft=np.deg2rad(self.limits.data['max_joint_deviation_deg']-self.limits.joint_planning_margin-.5)
                    hot=delta>=soft
                    if hot.any():
                        inward=(predicted_q is not None and np.all(np.abs(np.asarray(predicted_q)[hot]-self.anchor_q[hot])<delta[hot]-np.deg2rad(.1)))
                        if not inward:
                            raise SampleSkipped('Approaching joint planning boundary; skip this sample after a controlled stop.')
                packet={'target_ee_pose':list(target),'calibration_session':self.session}
                if self.limits.plane_only:
                    packet.update(hand_q=state['measured_hand_q'],hand_timestamp_ms=state['hand_timestamp_ms'])
                self.publisher.send_json(packet)
                error = np.linalg.norm(np.asarray(state['pos'][:3]) - target[:3])
                angle = (R.from_quat(state['pos'][3:]).inv() * R.from_quat(target[3:])).magnitude()
                stationary = np.max(np.abs(state['Qd'])) < .003 and np.linalg.norm(state['vel'][:3]) < .001 and np.linalg.norm(state['vel'][3:]) < .005
                if error < self.limits.data['position_tolerance_m'] and angle < np.deg2rad(self.limits.data['angle_tolerance_deg']) and stationary:
                    settled_since = settled_since or time.monotonic()
                    if time.monotonic() - settled_since >= self.limits.data['settle_s']:
                        return state
                else:
                    settled_since = None
                time.sleep(.02)
            raise SafetyError('Calibration motion deadline exceeded.')
        finally:
            self.stop()
