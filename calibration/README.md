# Camera + robot calibration (UR5e, RealSense)

Finds, in one run, the pose of the **UR5e base** and the pose of the **camera** in a common
**world frame** defined by a ChArUco board lying on the table. It uses an ArUco board mounted
on the robot's end-effector (EE): the robot moves to 8 poses, at each one the camera sees both
boards, and the robot's own TCP reading closes the loop.

This is a self-contained copy of `robohand/src/robot_ipc_control/pose_estimation/robot_base_calibration.py`
(state of 2026-08-27, the version with the live window and `q` to cancel). Only imports, default
paths and the output folder were changed; the maths is untouched.

```
calibration/
├── setup.sh                    creates calibration/.venv (run once per PC)
├── run_calibration.sh          runs the calibration with that venv
├── robot_base_calibration.py   the calibration
├── transform_utils.py          small helpers it imports
├── requirements.txt            Python deps installed by setup.sh
├── boards/
│   ├── aruco_board_DICT_4X4_250_3x3_28.0mm_2026-06-29.{json,pdf}   EE board (print at 100 %)
│   └── charuco_board.{json,pdf,png}                                 world board (print at 100 %)
└── results/                    outputs; the 2026-08-27 run is kept as reference
```

## Frames and what the files mean

| Symbol | Meaning | Saved as |
|---|---|---|
| `T_world_robot_base` | UR5e base pose in the world frame. Maps points from the base frame to the world frame. | `results/base_pose_robot_<name>.npy` |
| `T_world_camera` | Camera (colour sensor) pose in the world frame. | `results/camera_extrinsic_<serial>.npy` |

All `.npy` files are 4x4 homogeneous matrices. A timestamped copy and a
`<timestamp>_calibration_summary.txt` with the dispersion over the 8 observations are
written alongside.

**World frame**: the ChArUco board frame rotated by 180° about x then 90° about z
(`T_board_to_world = R_z(90°) · R_x(180°)` in the script), so z points up out of the table.

**EE frame**: whatever the UR reports as its TCP (`getActualTCPPose`), i.e. the TCP configured
in the pendant's installation when the arm controller is running. The controller publishes it
as `[x, y, z, qx, qy, qz, qw]`.

## Where the marker board sits relative to the EE frame

The script does **not** estimate this. It is hard-coded as `ADAPTER_TO_EE_POSE` and was
measured from the adapter geometry. It is the EE frame expressed in the board frame:

```
ADAPTER_TO_EE_POSE  (T_board_ee) =
 [[1  0  0  0.046]
  [0  0 -1  0.046]
  [0  1  0  0.050]
  [0  0  0  1    ]]
```

The board frame is the OpenCV ArUco board frame: origin at the first corner of marker 200,
x along the rows, y along the columns, z out of the printed face. The board is 92 x 92 mm
(3x3 markers of 28 mm with 4 mm gaps), so its centre is at (0.046, 0.046, 0).

Inverting gives the board in EE coordinates, which is usually what you want:

```
T_ee_board = inv(ADAPTER_TO_EE_POSE) =
 [[1  0  0 -0.046]
  [0  0  1 -0.050]
  [0 -1  0  0.046]
  [0  0  0  1    ]]
```

In words:

- Board **centre** is at `(0, -0.050, 0)` in the EE frame: 5 cm along the EE's **−y** axis,
  on the tool axis (no x or z offset).
- Board **normal** (printed face) points along the EE's **+y** axis. So the board is on the
  side of the adapter, its face parallel to the tool z-axis, looking sideways.
- Board **x** = EE **x**. Board **y** = EE **−z** (the columns run back toward the flange).
- Board **origin** (corner of marker 200) is at `(-0.046, -0.050, 0.046)` in the EE frame.

If the adapter is remounted on a different side or a different TCP is set in the pendant,
`ADAPTER_TO_EE_POSE` must be changed accordingly, otherwise `T_world_robot_base` is off by
exactly that error. (The original robohand repo also has `robot_hand_eye_calibration.py`, which
solves for the EE-to-board transform with `cv2.calibrateHandEye`; it is not needed here because it was
not the workflow used for this setup.)

## 1. Install (once per PC)

Everything lives in `calibration/.venv`, separate from the replay venv. Tested with Python 3.10
(Ubuntu 22.04); `pyrealsense2` 2.53 has no wheels for Python 3.12+.

```bash
sudo apt install python3-venv
./calibration/setup.sh
```

The pip `pyrealsense2` does not install the udev rules, so as a normal user it may see no camera.
If `./calibration/run_calibration.sh --list` prints nothing while the camera is plugged in (USB 3
port), install the rules once and replug the camera:

```bash
sudo curl -fsSL -o /etc/udev/rules.d/99-realsense-libusb.rules \
  https://raw.githubusercontent.com/IntelRealSense/librealsense/v2.53.1/config/99-realsense-libusb.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 2. Run

Hardware: UR5e in Remote Control on 192.168.1.x, RealSense on a USB 3 port, both boards
printed at 100 % scale, the world board taped flat on the table in view, the EE board on the
adapter. Clear the workspace: the robot moves ±5 cm around its current pose with random
small rotations. Keep the e-stop in hand. Do not run the replay at the same time (both bind port 5555).

```bash
# Terminal 1: arm controller (same one the replay uses)
./arm/run_controller.sh

# Terminal 2: calibration. Defaults: boards from calibration/boards, ports from arm/pc_ur_new.json,
# outputs into calibration/results. Only the name and the camera serial are required.
./calibration/run_calibration.sh --list                     # serials of the connected cameras
./calibration/run_calibration.sh --name ur5e --serial_number <camera serial>
```

At each of the 8 poses a window shows the detections and both frame axes. Press `y` to accept,
`n` to retry the same pose, `q` to cancel (nothing is written). At the end the robot returns to
its start pose and the files above are written. Check the dispersion in the summary: the
reference run had ~8 mm / 0.4° for the base and ~4 mm / 0.2° for the camera.

Options: `--world_board_path`, `--ee_board_path`, `--output_dir`, `--robot_config_path`,
`--camera_config_path <tag-pose-estimation camera json>` (updates its
`extrinsic_calibration_file` entry, as the original did), `--world_board_path legacy` for the
old built-in 4x5 board.

## Cameras

The reference run in `results/` was done with the **D455, serial 138322253845**. The D435I
whose intrinsics are in `../intrinsic_calibration.txt` is serial **242322072500**; pass that
serial to calibrate it. Intrinsics are read live from the device, not from a file.

## Geometry used by generated-motion replay

`surface_guard.py` loads `models/ur5e_dg5f_collision.json`, a checksum-verified
collision support model for the UR5e and right DG5F. `build_collision_model.py`
records the source URDF and mesh hashes and retains convex support vertices for
signed plane-distance checks. `overlay_utils.py` supplies camera projection and
nominal arm/hand transforms to `brain/capture/`.

The model is robot geometry, not an installation calibration. Supply the actual
capture's calibration results locally; table clearance does not certify other
objects, cables or self-collision. New replay plans validate the entire shifted
path and record their model and calibration provenance.
