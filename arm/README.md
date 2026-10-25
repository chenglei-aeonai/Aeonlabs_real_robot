# Arm: UR5e low-level controller

| Path | What it is |
|---|---|
| `impedance_controller` | prebuilt binary (Ubuntu 22.04, x86-64). Despite the name it is a joint-space velocity servo with a P term and feed-forward, driven over ZMQ |
| `pc_ur_new.json` | its configuration: robot IP, ports, gains, reset pose |
| `run_controller.sh` | starts it (terminal 3 of the replay procedure) |
| `src/` | C++ sources (from `robohand/src/robot_ipc_control/controller`) and `build.sh` to rebuild |

Protocol: JSON commands on TCP port 5555 (`socket_port`), state published on 5556 (`publisher_port`). `src/example.py`
shows the message format; `replay_real.py` at the top level is the real client.

## 1. Install

`librtde` is not in Ubuntu's own repositories: it comes from the ur_rtde PPA (add it once):

```bash
sudo add-apt-repository ppa:sdurobotics/ur-rtde
sudo apt update
```

Runtime libraries only (to use the prebuilt binary):

```bash
sudo apt install libzmq5 libboost-thread1.74.0 librtde
```

The binary was linked on Ubuntu 22.04 against Boost 1.74 and ur_rtde 1.6. On another Ubuntu version rebuild it:

```bash
sudo apt install build-essential cmake pkg-config libzmq3-dev libeigen3-dev libboost-all-dev python3-dev librtde-dev
./arm/build.sh
```

If `librtde-dev` is not in your distribution's apt, build ur_rtde from source as described at
https://sdurobotics.gitlab.io/ur_rtde/installation/installation.html (it installs into `/usr/local`, which the
CMakeLists already looks in).

Check it loads:

```bash
ldd arm/impedance_controller | grep "not found"    # must print nothing
```

## 2. Hardware and network

- Plug the robot's Ethernet cable into the PC. The UR controller box is at **192.168.1.100** (`robot_ip` in
  `pc_ur_new.json`); the PC must be on 192.168.1.0/24, conventionally **192.168.1.10**:

```bash
sudo nmcli con add type ethernet ifname <iface> con-name ur5 ipv4.method manual ipv4.addresses 192.168.1.10/24
sudo nmcli con up ur5
ping -c 2 192.168.1.100
```

- On the teach pendant: top-right menu, select **Remote Control**. The controller refuses to connect in Local mode.
- Keep the e-stop in hand. The controller streams velocity commands at 500 Hz and has no collision model.

## 3. Run

```bash
./arm/run_controller.sh                   # uses pc_ur_new.json
./arm/run_controller.sh other_config.json
```

It prints the connection status and then waits for commands. Stop it with Ctrl-C; the robot halts.
If it fails to bind port 5555, another controller instance (or a client that binds 5555) is still running.

## Configuration notes (`pc_ur_new.json`)

- `reset_pose`: joint angles (rad) used by the reset command. The copy here is the one used with the replay
  demonstrations; the original in `src/pc_ur_new.json` is identical.
- `frequency`: control loop rate (500 Hz).
- `max_joint_velocity`, `joint_acceleration`, `path_tracking_p_gain`: limits and gain for timed-path following,
  which is what `replay_real.py` uses. Lower `path_tracking_p_gain` for softer tracking.
- `kpd_values`, `position_delta`, `orientation_delta`, `exponential_filter`: Cartesian servo parameters, not used by
  the replay.

## Streamed retargeted replay

Rebuild from the sources in this repository before using the pipeline runner.
The controller starts idle, accepts explicit stop messages, reports actual joint
velocities and protective/emergency stop flags, and advertises its joint-target
watchdog. A missing target stream causes a stop. Protective stops require manual
inspection and recovery; reconnect does not automatically unlock them.

The sources also include the existing optional calibration guard and its
standalone test target. Building requires `libssl-dev` for SHA-256 validation of
collision geometry. No installation-specific safety configuration is included.
