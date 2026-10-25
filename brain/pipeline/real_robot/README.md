# real_robot

Replays a robot demonstration (hand_retarget's `.npz` export, checked in physics_sim) on the real UR5e + DG5F,
through the controllers that already run on the ur5 PC. One file, `replay_real.py`, needing only numpy and pyzmq.

## On the ur5 PC: four terminals, e-stop in hand

```bash
# 1. DG5F driver
cd /home/duplo/git/tesollo_ros2 && source /opt/ros/humble/setup.bash && source install_dg5f/setup.bash
ros2 launch dg5f_driver dg5f_right_pid_all_controller.launch.py

# 2. hand bridge (same two setups sourced)
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real && python3 dg5f_policy_ros_bridge.py

# 3. arm controller (UR in Remote Control)
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real && ./impedance_controller pc_ur_new.json

# 4. the replay, with simtoolreal_real's Python
cd /home/duplo/simone/startup_real_robot
PY=/home/duplo/simone/SimToolReal/deployment/simtoolreal_real/.venv/bin/python
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
   from farther than 1.9 rad). Enter: the hand ramps to its start pose at 0.5 rad/s.
4. Enter: the arm gets the whole trajectory as a timed path (the controller follows it with velocity
   feed-forward and holds its last point), and the hand targets are streamed at 60 Hz alongside.
5. At the end the hand **keeps its last target** (so the grasp does not relax) until Enter.

It stops the arm (`{"stop": true}`) and holds the hand where it is when: the arm is more than 10 deg off the plan,
a finger more than `--hand-error-limit-deg` (35) off it, either state is older than 0.25 s, or on Ctrl-C.
Once the arm has its path, it would finish it even if this script died: the e-stop is the backstop.

Each replay writes `logs/<demo>_<time>.npz` (time, measured and target arm and hand angles).

## Without a robot

```bash
uv run python fake_robot.py          # fake arm controller + hand bridge, same protocol
uv run python replay_real.py ../hand_retarget/out/<demo>.npz --send
env -u PYTHONPATH uv run pytest      # the tests run the script against the fakes
```
