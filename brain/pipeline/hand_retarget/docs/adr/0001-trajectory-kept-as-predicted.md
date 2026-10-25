# Skeletons are used exactly as predicted; only the demonstration placement is adjusted

We read the `raw_x/y/z_m` columns of `*_world_joints.csv`, the hand pose estimator's depth rotated to z-up with no correction, and never modify a skeleton: no table-contact shift, no lifting a frame out of the table, no clamping of robot targets. The estimator's depth is often several centimetres wrong (in the first clip the wrist sinks 6 cm and the fingertips up to 14 cm below where they rested), but any fix applied frame by frame changes the motion in ways we cannot check, and so hides what the retargeter is really being given. When the hand goes through the table we change the demonstration placement or the table height, one rigid transform for the whole human demonstration, and the robot may go through the table too.

## Considered Options

- **The file's corrected `x/y/z_m` columns**: they shift each frame along the camera ray so the wrist never goes below its frame-0 height. Rejected because the rule is ad hoc and only half works (fingertips stay 5–8 cm low during the grasp).
- **A table lift per frame** (raise the whole skeleton until no keypoint is below the table): rejected for the same reason; it rewrites the trajectory.
- **Clamping robot fingertips above the table**: rejected for v1, which is a kinematic replay only. Safety limits are a separate decision to take before anything runs on the real robot.
