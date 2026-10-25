# Aeonlabs real robot

Replays a robot demonstration (hand_retarget's `.npz` export, checked in physics_sim) on the real UR5e + Tesollo DG5F.
This folder is self-contained: it holds the hand driver and bridge (`hand/`), the arm controller (`arm/`) and the
replay script (`replay_real.py`). Paths below are relative to this folder; nothing depends on a particular user or home.

```
Aeonlabs_real_robot/
├── hand/            DG5F ROS 2 driver sources, build script, ZMQ bridge   -> hand/README.md for install
├── arm/             UR5e low-level controller (binary + sources + config)  -> arm/README.md for install
├── calibration/     camera + robot base calibration with the EE marker board       -> calibration/README.md
├── brain/           capture, generation, retargeting and supervised replay pipeline
├── replay_real.py   the replay (this file is the only thing that needs the venv below)
├── fake_robot.py    fake arm + hand with the same protocol, for trying without a robot
├── *.npz            recorded demonstrations (input)
└── logs/            one .npz per replay run (output)
```

## Install

1. Follow `hand/README.md` (ROS 2 Humble + workspace build) and `arm/README.md` (runtime libraries, network).
2. Python for the replay: only numpy and pyzmq.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

3. Only if you calibrate camera and robot base: `./calibration/setup.sh` (its own venv, see `calibration/README.md`).

## On the robot PC: four terminals, e-stop in hand

Hand powered (24 V / 10 A) and on 169.254.186.x; UR in Remote Control and on 192.168.1.x (see the sub-READMEs).

```bash
# 1. DG5F driver
./hand/run_driver.sh

# 2. hand bridge (after the driver reports its controllers active)
./hand/run_bridge.sh

# 3. arm controller
./arm/run_controller.sh

# 4. the replay
PY=.venv/bin/python
$PY replay_real.py demo.npz                         # dry run: connects, checks, prints; moves nothing
$PY replay_real.py demo.npz --send --arm-only --speed 0.25
$PY replay_real.py demo.npz --send --speed 0.5
$PY replay_real.py demo.npz --send
```

Do not run the MANUS teleop or anything else that commands the hand or binds ports 5555 / 5563 at the same time.

## What it does

1. Loads the demonstration and refuses it if it has NaNs, leaves the joint limits or is faster than π rad/s.
   Prints the start pose, the table height and **where to put the tube** (centre and axis in the UR base frame).
2. Connects and prints how far the robot is from the start pose. Without `--send` it stops here.
3. `SEND` arms it. Enter: the arm moves to the start pose on a slow spline (at least 5 s, 0.15 rad/s; refused
   from farther than 1.9 rad). Enter: the hand ramps to its start pose at 0.5 rad/s
   (accepted within 25 deg per joint: a stiff pinky can stop short).
4. Enter: the arm gets the whole trajectory as a timed path (the controller follows it with velocity
   feed-forward and holds its last point), and the hand targets are streamed at 60 Hz alongside.
   A demonstration starts mid-motion, so the first `--ramp-in` s (1) are eased in: the playback rate
   starts at a quarter and reaches full speed after that, adding 1 s to the replay. The path is unchanged.
5. At the end the hand **keeps its last target** (so the grasp does not relax) until Enter.

It stops the arm (`{"stop": true}`) and holds the hand where it is when: the arm is more than 10 deg off the plan,
a finger more than `--hand-error-limit-deg` (40) off it, either state is older than 0.25 s, or on Ctrl-C.
Once the arm has its path, it would finish it even if this script died: the e-stop is the backstop.

Each replay writes `logs/<demo>_<time>.npz` (time, measured and target arm and hand angles).

Other options: `--hand-only`, `--controller-config arm/pc_ur_new.json` (if you change the ports there),
`--hand-command-port` (5562), `--hand-state-port` (5563), `--log-dir`.

## Without a robot

```bash
.venv/bin/python fake_robot.py          # fake arm controller + hand bridge, same protocol
.venv/bin/python replay_real.py demo.npz --send
```

## Generated-motion replay

The integrated pipeline and actual robot controllers now live in this repository.
See [brain/pipeline/docs/physical_replay.md](brain/pipeline/docs/physical_replay.md)
for retargeting, local time allocation, fixed-pose resets, and the explicit
Prepare / Run / Stop workflow. The original `replay_real.py` demo commands above
remain available.

Build the updated controller from source with `./arm/build.sh`, and run the
updated hand bridge with `./hand/run_bridge.sh`. The new runner checks for the
controller watchdog and measured-state freshness before allowing motion. It
uses the saved capture calibration and the bundled collision support model;
installation-specific calibration results must be supplied locally.

The deployed test completed the original 4.4167-second arm-and-hand clip in
6.3109 seconds, 7.7652 including final settling. See the
[execution report](brain/pipeline/docs/eight_second_replay_20261025.md) for its
remaining 3.309-degree finger error and actual limits.
