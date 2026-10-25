# Camera-space hand capture

Capture a stationary RealSense image and measured UR5e/Tesollo DG5F state,
then save the right-hand joints in camera coordinates and an undistorted overlay.
This program sends no motion commands. It uses the calibration Python environment
(`./calibration/setup.sh`) and the newer calibration overlay/URDF helpers.

## Run

Keep the normal arm controller, hand driver, and bridge running in separate terminals:

```bash
./arm/run_controller.sh
./hand/run_driver.sh
./hand/run_bridge.sh
```

Keep the arm and fingers stationary, close other camera consumers and hand-overlay
listeners (which also bind UDP port 5563), then run from the repository root:

```bash
./brain/capture/run_capture.sh
```

Defaults: camera `138322253845`, robot `ur5e`, calibration directory
`calibration/results/original_legacy/`. The newest timestamp with both robot-base
and camera transforms is selected; generic filenames are a fallback when there
are no timestamped pairs. Recalibrate if the camera or robot base has moved.
Use `--help` for path, camera, and optional TCP-to-hand transform overrides.

## Outputs

Each capture creates `brain/capture/results/<UTC timestamp>/` containing:

- `raw.png`: original color frame.
- `undistorted.png`: full-size image retaining the factory intrinsic matrix.
- `overlay.png`: projected arm, TCP, wrist, finger joints, and fingertips.
- `mano_overlay.png`: separate 21-point mapped MANO skeleton, labeled by index,
  drawn on the undistorted image. Corrected wrist and synthetic pinky DIP are white.
- `hand_joints.json`: raw/undistorted intrinsics, calibration filenames/checksums,
  transforms, measured joint angles, timestamps, 26 named robot hand points, and
  `mano.joints` containing 21 mapped camera-space landmarks.

Camera coordinates use metres, x right, y down, z forward. `hand_joints` includes
the wrist (`rl_dg_mount` origin), four URDF joint origins per finger, and each
fixed fingertip. Finger indices follow URDF names `rj_dg_1_*` through `rj_dg_5_*`.
`pixel_undistorted` uses zero distortion; points behind the camera have null pixels.
`in_image` checks image bounds only and does not estimate occlusion.

The wrist is the hand mount, not the measured UR TCP or an anatomical landmark.
The default mount matches calibration overlays: nominal UR5e FK at tool0 followed
by Rz(60 degrees), using the bundled right-hand URDF. `--tcp_to_hand_path` can
replace this with a measured rigid TCP-to-hand `.npy` transform. Positions are
kinematic estimates, not independent camera detections. Fresh feedback and small
motion checks are enforced, but camera and robot streams are not hardware synchronized.

## Mapped MANO landmarks

`mano.joints` follows the order used by `hand_retarget`'s `skeleton.py`:
wrist; index MCP/PIP/DIP; middle MCP/PIP/DIP; pinky MCP/PIP/DIP;
ring MCP/PIP/DIP; thumb CMC/MCP/IP; index/middle/pinky/ring/thumb tips.
Each entry includes its index, name, camera-space position, undistorted pixel,
image-bounds flag, source weights, and a `synthetic` flag.

Thumb, index, middle, and ring map robot joint origins 2/3/4 and tip.
Pinky MCP/PIP map robot joints 3/4; its DIP (index 9) is the **3D midpoint**
between PIP and tip. This is a synthetic approximation, not a physical joint.
The mapped wrist is shifted along the robot tool0 longitudinal (+Z) axis to
match the reference hand's wrist-to-knuckle proportions. Original robot
`hand_joints` still retain the actual hand-mount origin.

All mapped points are computed from the exported robot **camera-frame** positions.
Projection applies only `K_undistorted` and perspective division, with zero
distortion; camera extrinsics are not applied again. These landmarks approximate
the MANO layout; they are not a fitted MANO mesh or MANO joint-angle parameters.

## Saved wrist conversion

`mano_wrist_conversion.json` stores the reusable conversion and reference
provenance. Its scale-free ratio is the median wrist-to-knuckle-center distance
divided by index-to-pinky MCP width over 107 right-hand frames in
`hand_retarget/data/20260928T132405_scene0_world_joints.csv` (MANO-style estimator
landmarks, not an official neutral MANO template). The ratio is approximately
1.2597. The reference path and checksum are saved; future captures need only this
JSON, not the external dataset.

For each capture, the four long-finger MCPs determine the knuckle center and
index/pinky MCPs determine width. The wrist moves only on the line through the
robot mount parallel to tool0 Z. Solve the desired wrist-to-center distance
using the proximal root, keeping the wrist behind the knuckles. The distance
scales with robot knuckle width; fingers are never repositioned. Geometry for
which the target distance is impossible on this line is rejected explicitly.

Each output embeds the conversion under `mano.wrist_conversion` and records
the original wrist, camera-space axis, signed shift, and target distance under
`mano.joints[0].wrist_correction`. Camera projection still uses only the final
camera-frame points and undistorted intrinsics.

To apply the current conversion to a saved capture without accessing hardware:

```bash
calibration/.venv/bin/python brain/capture/remap_mano.py brain/capture/results/<capture_timestamp>
```

It creates a separate `<capture_timestamp>_mano_wrist` folder with updated JSON
and MANO overlay, preserving the original capture. The normal
`./brain/capture/run_capture.sh` automatically uses the saved conversion for new captures.
