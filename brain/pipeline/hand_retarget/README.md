# hand-retarget

The experimental [Robot Demonstration Studio](../docs/web_pipeline.md) adds an
independent browser UI and script API for robot photo → human demonstration →
UR5e/DG5F trajectory export, without a Codex runtime dependency.

Retarget a human hand skeleton (from video) onto a UR5e + Tesollo DG5F and replay it kinematically in a browser viewer.
Vocabulary is in [CONTEXT.md](CONTEXT.md); why the skeletons are never corrected is in
[docs/adr/0001-trajectory-kept-as-predicted.md](docs/adr/0001-trajectory-kept-as-predicted.md).

```bash
# viewer: robot table on the left, skeleton table on the right; open http://localhost:8080
uv run hand-retarget-view data/20260928T132405_scene0_world_joints.csv

# robot demonstration (arm_q, hand_q at 60 Hz, simtoolreal's layout) without the viewer
uv run hand-retarget-export data/20260928T132405_scene0_world_joints.csv --method vector
```

Input is a `*_world_joints.csv`; its `raw_x/y/z_m` columns (right hand) are used as they are.
The viewer's Start frame and End frame sliders choose which video frames the robot demonstration keeps; they are
saved with the placement in `*_placement.json` next to the CSV (`first_frame`, `last_frame`), which the export reads too.
The whole clip is always retargeted, and cut afterwards.

**Strengthen grasp** (viewer folder) closes the fingers more than the human did: an extra angle per finger on the
joint that closes it at its base (thumb `rj_dg_1_3`, index to ring `rj_dg_x_2`, pinky `rj_dg_5_3`), from a start
frame to an end frame. It grows linearly over the ramp frames before the start frame and goes away over as many
after the end frame, and stays inside the joint limits and the speed limit. It is saved with the placement
(`grasp_extra_deg`, thumb to pinky, `grasp_first_frame`, `grasp_last_frame`, `grasp_ramp_frames`).
Per frame: optional One-Euro filter → demonstration placement → arm IK puts `rl_dg_palm` on the human palm,
knuckle row on knuckle row → one of three finger retargeters works in the palm frame:

| method | how | hand scaling (default) |
|---|---|---|
| `fingertip_ik` | fingertip positions | global |
| `vector` | DexPilot: palm→tip and tip→tip vectors, near pinches snapped shut | per finger |
| `joint_mapping` | angles measured on the skeleton, mapped like the MANUS teleop's `manus_right_to_dg5f` | none |

`fingertip_ik` and `vector` run on one of two optimizers:

- `own` (default): our optimizer, which also follows every phalanx's direction and keeps every pair of
  fingers, thumb included, from passing through each other. About 7 s per clip.
- `dex`: dex-retargeting, which sees only the fingertips; DIP is coupled to PIP to remove the spare joint.
  Fast, but fingers can pass through each other and switch between equivalent solutions.

Every method is held to the DG5F's joint speed limit (π rad/s), so no joint jumps between frames.

The URDF, meshes, home pose and scene numbers are copied from simtoolreal_newton so replays line up with it.

Run the tests with `env -u PYTHONPATH uv run pytest` (unset PYTHONPATH so a sourced ROS install does not load its pytest plugins).

For H3 video/action inference and the `h3-retarget run` / `h3-retarget retarget`
commands, see the [pipeline integration guide](../docs/h3_pipeline.md).
