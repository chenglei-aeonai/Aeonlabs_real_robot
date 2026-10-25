"""Checks where the right answer is known in advance.

- Round trip: pose the robot hand, use its own fingertips as the "human" skeleton, and
  the fingertip retargeters must reproduce them.
- Joint mapping: a flat hand must give almost no flexion, a synthetic hand with known
  angles must give those angles back, and a fist must bend every finger.
"""

from pathlib import Path

import numpy as np
import pytest

from hand_retarget.pipeline import Settings, export_robot_demonstration, retarget_demonstration, retarget_hands
from hand_retarget.placement import PlacementAdjustment
from hand_retarget.placement import apply, default_placement
from hand_retarget.hand_optimizer import MAX_JOINT_SPEED
from hand_retarget.retargeters import JointMapping, human_joint_angles, make_retargeter
from hand_retarget.robot import FINGER_CHAIN_JOINTS, HAND_LOWER, HAND_UPPER, HOME_ARM_Q, RobotKinematics
from hand_retarget.scaling import HandScaling
from hand_retarget.skeleton import (
    FINGER_BASES, FINGER_CHAINS, FINGERTIPS, WRIST, hand_anchor, load_world_joints, palm_frame, to_frame,
)

DEMO = Path(__file__).resolve().parents[1] / "data" / "20260928T132405_scene0_world_joints.csv"


@pytest.fixture(scope="module")
def robot():
    return RobotKinematics()


@pytest.fixture(scope="module")
def demo():
    return load_world_joints(DEMO)


def unit_scaling(robot):
    return HandScaling(
        per_finger=np.ones(5),
        overall=1.0,
        robot_finger_bases=robot.finger_bases_in_palm,
        robot_hand_anchor=robot.hand_anchor_in_palm,
    )


def robot_as_skeleton(robot, hand_q):
    """A skeleton in the palm frame traced on the robot hand at ``hand_q``: every finger keypoint
    sits on the matching robot joint or fingertip."""
    skeleton = np.zeros((21, 3))
    anchor = robot.hand_anchor_in_palm
    tips = robot.fingertips_in_palm(hand_q)
    for finger, joints in enumerate(FINGER_CHAIN_JOINTS):
        points = np.vstack([robot.hand_points_in_palm(hand_q, joints), tips[finger]]) - anchor
        if len(points) == 3:  # the DG5F pinky has one phalanx fewer: put the DIP halfway
            points = np.vstack([points[:2], (points[1] + points[2]) / 2, points[2]])
        skeleton[list(FINGER_CHAINS[finger])] = points
    return skeleton


def random_hand_poses(count, seed=0):
    """Reachable, collision-free poses: flexion in the first half of each range, no spread,
    DIP equal to PIP (as the dex-retargeting variant couples them)."""
    rng = np.random.default_rng(seed)
    poses = []
    for _ in range(count):
        q = HAND_LOWER + rng.uniform(0.1, 0.5, 20) * (HAND_UPPER - HAND_LOWER)
        q[[4, 8, 12, 16, 17]] = 0.0
        q[[7, 11, 15]] = q[[6, 10, 14]]
        poses.append(q)
    return poses


def round_trip_errors(robot, retargeter, frames):
    """Hold each pose still for ``frames`` frames: the speed limit, and for dex-retargeting
    its pull toward the previous frame, need several frames to get there."""
    errors = []
    for hand_q in random_hand_poses(8):
        skeleton = robot_as_skeleton(robot, hand_q)
        retargeter.reset()
        for _ in range(frames):
            q = retargeter.retarget(skeleton)
        errors.append(np.linalg.norm(robot.fingertips_in_palm(q) - robot.fingertips_in_palm(hand_q), axis=1))
    return np.array(errors)


@pytest.mark.parametrize("optimizer", ["dex", "own"])
def test_fingertip_ik_reproduces_reachable_fingertips(robot, optimizer):
    retargeter = make_retargeter("fingertip_ik", unit_scaling(robot), "global", optimizer)
    errors = round_trip_errors(robot, retargeter, frames=60 if optimizer == "dex" else 15)
    assert errors.max() < 0.003, np.round(errors * 1000, 1)


@pytest.mark.parametrize("optimizer", ["dex", "own"])
def test_vector_retargeting_reproduces_reachable_fingertips(robot, optimizer):
    retargeter = make_retargeter("vector", unit_scaling(robot), "global", optimizer)
    errors = round_trip_errors(robot, retargeter, frames=60 if optimizer == "dex" else 15)
    # DexPilot deliberately pulls fingertip pairs closer than 3 cm into contact, so a
    # random pose with a near pinch is not reproduced exactly; the typical one is.
    assert np.median(errors) < 0.003, np.round(errors * 1000, 1)


def test_per_finger_scaling_hangs_each_fingertip_from_its_robot_finger(robot):
    skeleton = robot_as_skeleton(robot, np.zeros(20))
    targets = unit_scaling(robot).fingertip_targets(skeleton, "per_finger")
    np.testing.assert_allclose(targets, robot.fingertips_in_palm(np.zeros(20)), atol=1e-9)


def test_palm_ik_reaches_poses_the_arm_can_take(robot):
    rng = np.random.default_rng(1)
    for _ in range(10):
        arm_q = HOME_ARM_Q + rng.uniform(-0.3, 0.3, 6)
        target = robot.palm_pose(arm_q)
        _, position_error, rotation_error = robot.solve_palm_ik(target, HOME_ARM_Q)
        assert position_error < 1e-4 and rotation_error < 1e-3


def test_palm_frame_on_the_resting_hand(demo):
    frame = palm_frame(demo.skeletons[0])
    # Palm flat on the table in the first frame: the palm side faces down, the fingers are level.
    assert frame[2, 0] < -0.9
    assert abs(frame[2, 2]) < 0.3


def test_palm_frame_moves_with_the_hand(demo):
    skeleton = demo.skeletons[0]
    angle = 0.7
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = skeleton @ rotation.T + np.array([0.3, -0.1, 0.2])
    np.testing.assert_allclose(to_frame(palm_frame(moved), moved), to_frame(palm_frame(skeleton), skeleton), atol=1e-9)


def synthetic_hand(demo, spread_deg, flex_deg):
    """A skeleton in the palm frame with the first frame's bone lengths and chosen angles.

    Long fingers start from the first frame's MCPs pointing along the palm (+z), turned
    toward the thumb by ``spread_deg`` and bent toward the palm by ``flex_deg`` at each
    of the three joints.
    """
    first = to_frame(palm_frame(demo.skeletons[0]), demo.skeletons[0])
    skeleton = first.copy()
    for finger in range(1, 5):
        chain = FINGER_CHAINS[finger]
        lengths = np.linalg.norm(np.diff(first[list(chain)], axis=0), axis=1)
        spread, flex = np.deg2rad(spread_deg), np.deg2rad(flex_deg)
        lateral = np.array([0.0, np.cos(spread), -np.sin(spread)])  # bending axis, across the finger
        direction = np.array([0.0, np.sin(spread), np.cos(spread)])
        point = first[chain[0]]
        for bone, length in enumerate(lengths):
            bend = flex
            # Rodrigues rotation of the direction toward the palm (+x) about the lateral axis.
            direction = (
                direction * np.cos(bend) + np.cross(lateral, direction) * np.sin(bend)
                + lateral * lateral.dot(direction) * (1 - np.cos(bend))
            )
            point = point + length * direction
            skeleton[chain[bone + 1]] = point
    # Keep the metacarpals where they were: the wrist and MCPs are untouched.
    assert np.allclose(skeleton[WRIST], first[WRIST])
    return skeleton


@pytest.mark.parametrize("spread_deg, flex_deg", [(0, 0), (10, 20), (-8, 45), (5, 70)])
def test_joint_angles_are_measured_back(demo, spread_deg, flex_deg):
    angles = np.degrees(human_joint_angles(synthetic_hand(demo, spread_deg, flex_deg)))
    long_fingers = angles[1:]
    if flex_deg <= 60:  # beyond that, spread is faded out on purpose
        np.testing.assert_allclose(long_fingers[:, 0], spread_deg, atol=1.0)
    np.testing.assert_allclose(long_fingers[:, 2:], flex_deg, atol=1.0)
    # MCP flexion is measured against the metacarpal, which is not quite in the palm plane.
    np.testing.assert_allclose(long_fingers[:, 1], flex_deg, atol=8.0)


def test_joint_mapping_flat_hand_is_straight(demo):
    q = JointMapping().retarget(to_frame(palm_frame(demo.skeletons[0]), demo.skeletons[0])).reshape(5, 4)
    long_finger_flexion = np.concatenate([q[1:4, 1:].ravel(), q[4, 2:]])
    assert np.degrees(long_finger_flexion).max() < 20.0


def test_joint_mapping_fist_bends_every_finger(demo):
    q = JointMapping().retarget(synthetic_hand(demo, 0, 70)).reshape(5, 4)
    assert np.degrees(q[1:4, 1:]).min() > 60.0
    assert np.degrees(q[4, 2:]).min() > 60.0


VARIANTS = [("fingertip_ik", "dex"), ("fingertip_ik", "own"), ("vector", "dex"), ("vector", "own"), ("joint_mapping", "own")]


@pytest.fixture(scope="module")
def replays(demo, robot):
    return {v: retarget_demonstration(demo, robot, Settings(method=v[0], optimizer=v[1])) for v in VARIANTS}


@pytest.mark.parametrize("variant", VARIANTS)
def test_whole_demonstration_stays_within_the_real_hand(replays, variant, demo):
    replay = replays[variant]
    hand_q = replay.stack("hand_q")
    assert hand_q.shape == (demo.num_frames, 20)
    assert np.all(hand_q >= HAND_LOWER - 1e-9) and np.all(hand_q <= HAND_UPPER + 1e-9)
    # No joint moves faster than the DG5F or the UR5e can: no jumps from one frame to the next.
    assert np.abs(np.diff(hand_q, axis=0)).max() <= MAX_JOINT_SPEED / demo.fps + 1e-9
    arm_q = replay.stack("arm_q")
    assert np.all(np.abs(np.diff(arm_q, axis=0)) <= np.pi / demo.fps + 1e-9)
    # Where the human wrist turns faster than the arm can, the palm lags; elsewhere it is on target.
    palm_error = np.array([frame.palm_position_error for frame in replay.frames])
    assert palm_error[0] < 1e-3 and np.median(palm_error) < 1e-3


@pytest.mark.parametrize("method", ["fingertip_ik", "vector"])
def test_own_optimizer_keeps_fingers_from_passing_through_each_other(replays, method):
    # Checked on the real collision meshes, every pair of fingers including the thumb.
    # The optimizer's capsules are a penalty, so fingers may touch and squeeze a little.
    assert replays[method, "own"].worst_finger_penetration().max() < 0.0025


def test_first_frame_hand_starts_on_the_robot_home_hand(demo, robot):
    placement = default_placement(demo, robot)
    home_palm = robot.palm_pose(HOME_ARM_Q)
    robot_anchor = apply(home_palm, robot.hand_anchor_in_palm)
    np.testing.assert_allclose(apply(placement, hand_anchor(demo.skeletons[0])), robot_anchor, atol=1e-9)


def test_moving_the_placement_leaves_the_fingers_alone(demo, robot):
    # The viewer reuses the fingers when only the placement changes; that must give what a
    # full run gives.
    moved = Settings(method="joint_mapping", adjustment=PlacementAdjustment(x=0.05, z=0.03, yaw=0.4, pitch=0.1))
    reused = retarget_demonstration(demo, robot, moved, hands=retarget_hands(demo, robot, Settings(method="joint_mapping")))
    full = retarget_demonstration(demo, robot, moved)
    np.testing.assert_allclose(reused.stack("hand_q"), full.stack("hand_q"), atol=1e-9)
    np.testing.assert_allclose(reused.stack("arm_q"), full.stack("arm_q"), atol=1e-9)
    np.testing.assert_allclose(reused.stack("fingertip_error"), full.stack("fingertip_error"), atol=1e-9)


def test_saved_placement_round_trips(tmp_path):
    from hand_retarget.placement import SavedPlacement, load_placement, save_placement

    csv = tmp_path / "clip_world_joints.csv"
    assert load_placement(csv) == SavedPlacement()  # nothing saved: the default placement
    saved = SavedPlacement(PlacementAdjustment(x=-0.1, y=0.045, z=0.05, yaw=0.3, pitch=0.45, roll=-0.12), 0.02)
    save_placement(csv, saved)
    loaded = load_placement(csv)
    np.testing.assert_allclose(list(vars(loaded.adjustment).values()), list(vars(saved.adjustment).values()), atol=1e-4)
    assert loaded.table_offset == pytest.approx(0.02)


def test_frame_range_is_saved_and_cuts_the_replay(tmp_path, demo, robot):
    from hand_retarget.placement import SavedPlacement, load_placement, save_placement

    csv = tmp_path / "clip_world_joints.csv"
    save_placement(csv, SavedPlacement(first_frame=10, last_frame=95))
    loaded = load_placement(csv)
    assert (loaded.first_frame, loaded.last_frame) == (10, 95)

    replay = retarget_demonstration(demo, robot, Settings(method="joint_mapping"))
    cut = replay.cut(loaded.first_frame, loaded.last_frame)
    assert cut.demo.num_frames == len(cut.frames) == 86  # frames 10 to 95
    np.testing.assert_array_equal(cut.stack("arm_q"), replay.stack("arm_q")[10:96])
    exported = np.load(export_robot_demonstration(cut, tmp_path / "cut.npz"))
    assert exported["timestamp"][-1] == pytest.approx(85 / demo.fps, abs=1 / 60)


def test_grasp_strengthening_closes_the_base_joints_between_its_frames(tmp_path, demo, robot):
    from hand_retarget.grasp import GRASP_JOINT_INDEX, GraspStrengthening
    from hand_retarget.placement import SavedPlacement, load_placement, save_placement

    grasp = GraspStrengthening(tuple(np.deg2rad([10.0, 20.0, 20.0, 20.0, 15.0])), first_frame=30, last_frame=60, ramp_frames=10)
    csv = tmp_path / "clip_world_joints.csv"
    save_placement(csv, SavedPlacement(grasp=grasp))
    loaded = load_placement(csv).grasp
    assert (loaded.first_frame, loaded.last_frame, loaded.ramp_frames) == (30, 60, 10)
    np.testing.assert_allclose(loaded.extra, grasp.extra, atol=1e-4)

    plain = retarget_demonstration(demo, robot, Settings(method="joint_mapping"))
    closed = retarget_demonstration(demo, robot, Settings(method="joint_mapping", grasp=grasp))
    before, after = plain.stack("hand_q"), closed.stack("hand_q")
    np.testing.assert_array_equal(after[:21], before[:21])  # nothing until the ramp begins, 10 frames before the start
    np.testing.assert_allclose(after[71:], before[71:], atol=1e-9)  # and nothing once ramped out
    np.testing.assert_allclose(grasp.envelope(100)[[20, 25, 30, 60, 65, 70]], [0, 0.5, 1, 1, 0.5, 0], atol=1e-12)
    others = np.setdiff1d(np.arange(20), GRASP_JOINT_INDEX)
    np.testing.assert_array_equal(after[:, others], before[:, others])
    expected = np.minimum(before[45, GRASP_JOINT_INDEX] + grasp.extra, HAND_UPPER[GRASP_JOINT_INDEX])
    np.testing.assert_allclose(after[45, GRASP_JOINT_INDEX], expected, atol=1e-9)
    assert (after[45, GRASP_JOINT_INDEX] >= before[45, GRASP_JOINT_INDEX]).all()
    assert np.abs(np.diff(after, axis=0)).max() <= MAX_JOINT_SPEED / demo.fps + 1e-9
