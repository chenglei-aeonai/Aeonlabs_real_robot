# Retargeted motion and supervised physical replay

The replay source is the retargeted arm and hand trajectory, not MuJoCo output.
Camera-space right-hand joints use the calibration saved with their capture.
ArmStage seeds sequential IK from the captured arm and preserves the calibrated
palm offset. `robot_demonstration.npz` follows the legacy replay format; that
format alone does not authorize physical execution.

## Deployment requirements

Run from this repository root. The pipeline uses the bundled sibling components:

- `calibration/surface_guard.py`, the checksum-verified UR5e/DG5F collision model,
  board definition, and the capture's calibration files;
- root `replay_real.py` for transports and `arm/pc_ur_new.json` for connection,
  acceleration and tracking-gain configuration;
- the normal arm controller accepting streamed `target_q` and `stop` messages,
  reporting actual `Q`, actual `Qd`, producer `timestamp_ms`, stop/fault flags and
  `joint_target_watchdog_s` in `(0, 0.25]`, `rtde_connected`, and
  `robot_state_age_s` derived from advances in the RTDE source timestamp;
- the hand bridge reporting actual positions/velocities and `measured_age_s`.
  Publishing a packet must not reset the age of an old sensor measurement.
  A missing velocity array is rejected rather than represented as zero speed.

**Build the controller and run the updated bridge from this branch.** The
required source changes and collision model are included in this repository.
An older installed binary still needs rebuilding. Unsupported feedback is
refused before motion.
The configured P gain must match the running arm controller: native replay uses
`q_command = q_reference + velocity_reference / P` for velocity feed-forward.
Use one command client at a time. The table model does not certify cable,
self-collision, other-object, or grasp safety.

## Prepare, inspect, execute

From the installed real-robot workspace root:

```bash
./brain/pipeline/run_robot_rollout.sh prepare \
  --source brain/pipeline/out/web-runs/RETARGET_ID/result \
  --out brain/pipeline/out/my-rollout --speed-scale 1
./brain/pipeline/run_robot_rollout.sh replay \
  --folder brain/pipeline/out/my-rollout
./brain/pipeline/run_robot_rollout.sh replay \
  --folder brain/pipeline/out/my-rollout --send
```

Preparation and replay without `--send` send no targets. Sending requires `SEND`.
The web interface exposes the same prepare/run flow and a Stop robot button;
a lost browser heartbeat interrupts its subprocess. Physical execution checks
fresh feedback, actual speed, tracking errors, joint ranges and table clearance.
The first reference matches the measured start. Translation is limited to 30 cm
from the source and the path remains within 30 cm of the live start.

Default `adaptive_cubic_v5` preserves the filtered waypoints and slows individual
intervals. It checks analytic cubic extrema independently of optimizer status.
Planning uses 80% of the arm acceleration budget (configured value capped at
3 rad/s²). Finger planning limits are 0.9 rad/s and 10 rad/s², except the thumb
distal joint's 0.5 rad/s and 4 rad/s². These are application planning constraints,
not manufacturer-certified limits. Actual arm/hand speed checks use the selected
demo profile, π rad/s; arm/hand tracking checks remain 0.08/0.2 rad.

The eight-second budget includes at least 0.5 seconds for final settling and
0.25 seconds stop margin. Settling can use up to 1.5 seconds when time permits.
A plan that cannot fit is rejected; motion is never truncated to report success.
A delivered endpoint with residual error is explicitly distinguished from precise
convergence. Fine tolerances are 0.01 rad arm and 0.05 rad hand.
`--timing-policy native` selects historical uniform 1×/0.5×/0.25× timing;
`--timing-policy legacy` selects the older conservative quintic policy.
Old prepared artifacts retain their saved policy.

## Return to the intended start

A reverse replay returns to that attempt's start, which may differ from the
original levelled pose. Do not accumulate relative replays from successive ends.
Use `hand_retarget.replay_return` for a completed logged path, or explicitly
review `hand_retarget.partial_return` for only the sent prefix of a stopped path.
Both validate source/model provenance and the expected return start.

For a fixed original reference, `hand_retarget.fixed_pose_return` takes a fresh
stopped-state JSON (`arm.Q/Qd/stop flags`, `hand.positions`), a successful
`level_hand.py` reference folder (`report.json`, `execution.npz`, `plan.npz`),
and a validated calibration rollout:

```bash
PY=brain/pipeline/hand_retarget/.venv/bin/python
$PY -m hand_retarget.fixed_pose_return \
  --state stopped_state.json --reference saved-levelled-pose \
  --calibration-rollout brain/pipeline/out/my-rollout \
  --out brain/pipeline/out/my-return
```

This is offline preparation. Execute the new folder through the same checked
replay command. The target joint pose stays absolute; current-position errors
cannot shift the destination. Reprepare after the robot moves. Return motion has
its own timing, separate from the eight-second forward-action budget.

`supervised_replay_probe.py` additionally records ROS joint, reference and PID
feedback while enforcing hardware connection/freshness. Run it with system
Python after sourcing `hand/env.sh`; it also requires `--send` and `SEND`.
The audit scripts only inspect recorded data. `level_hand.py` requires an explicit
`--calibration-rollout` folder rather than a machine-specific historical job.

## Validation

Tests use mocks or fake endpoints; they must not share ports with physical
controllers. Use `brain/pipeline/real_robot/fake_robot.py` for isolated protocol
checks, including the streamed-target watchdog. Tests requiring the collision
model use the installed sibling calibration/arm files described above.

```bash
PYTHONPATH=brain/pipeline/hand_retarget/src \
  brain/pipeline/hand_retarget/.venv/bin/python -m pytest \
  brain/pipeline/hand_retarget/tests
```

The [2026-10-25 hardware report](eight_second_replay_20261025.md) records the
successful deployed run: 6.311 seconds path, 7.765 seconds with settling,
all arm/hand targets delivered, no intermediate fault, and 3.309° maximum final
finger error. Logs named there are local evidence and are not bundled as datasets.
