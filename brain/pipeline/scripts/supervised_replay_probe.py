#!/usr/bin/env python3
"""Record ROS evidence around the existing guarded replay; stop on lost connection.

Use system Python after sourcing hand/env.sh. No commands are sent without
--send and an explicit SEND confirmation. Motion remains in robot_rollout.py.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from control_msgs.msg import DynamicJointState, MultiDOFCommand, MultiDOFStateStamped

ROOT = Path(__file__).resolve().parents[3]


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--send', action='store_true')
    parser.add_argument('--arm-only', action='store_true',
                        help='Keep fingers fixed while testing the retargeted arm path')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = rclpy.create_node('supervised_replay_diagnostics')
    cache = {}

    def save(key, values):
        cache[key] = {'received_monotonic_s': time.monotonic(), **values}

    def joint(message):
        save('joints', {'names': list(message.name), 'position': list(message.position),
                        'velocity': list(message.velocity), 'effort': list(message.effort)})

    def system(message):
        for name, values in zip(message.joint_names, message.interface_values):
            if name == 'system':
                save('system', dict(zip(values.interface_names, values.values)))

    def reference(message):
        save('reference', {'names': list(message.dof_names), 'values': list(message.values)})

    def pid(message):
        save('pid', {'joints': [{'name': s.name, 'reference': s.reference,
                                'feedback': s.feedback, 'feedback_dot': s.feedback_dot,
                                'error': s.error, 'error_dot': s.error_dot,
                                'time_step': s.time_step, 'output': s.output}
                               for s in message.dof_states]})

    for kind, topic, callback in (
        (JointState, 'joint_states', joint),
        (DynamicJointState, 'dynamic_joint_states', system),
        (MultiDOFCommand, 'rj_dg_pospid/reference', reference),
        (MultiDOFStateStamped, 'rj_dg_pospid/controller_state', pid),
    ):
        node.create_subscription(kind, '/dg5f_right/' + topic, callback, qos_profile_sensor_data)

    def health():
        now = time.monotonic()
        for key in ('joints', 'system'):
            if now - cache.get(key, {}).get('received_monotonic_s', 0) > .25:
                return f'Hand ROS {key} telemetry unavailable/stale (>0.25 s)'
        if cache['system'].get('connection_status') != 1.:
            return 'Hand hardware connection_status is not connected'
        return None

    process = None
    interrupted = False
    report = {'source_folder': str(args.folder.resolve()), 'motion_requested': args.send,
              'started_at_unix_s': time.time(), 'connection_guard': True,
              'arm_only': args.arm_only}
    try:
        until = time.monotonic() + 2.
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.02)
        reason = health()
        if reason:
            raise RuntimeError(reason + '; no motion started')
        (args.out / 'before.json').write_text(json.dumps(clean(cache), indent=2))
        print('ROS telemetry fresh; hand hardware connection_status=1.', flush=True)
        mode = 'arm-only' if args.arm_only else 'arm + hand'
        if args.send and input(f'Clear path, e-stop ready. Type SEND for {mode} replay: ').strip() != 'SEND':
            report['cancelled'] = True
            return
        # input() pauses callbacks. Refresh live telemetry after confirmation;
        # the pre-confirmation cache must never authorize or reject motion.
        until = time.monotonic() + .5
        while time.monotonic() < until:
            rclpy.spin_once(node, timeout_sec=.005)
        reason = health()
        if reason:
            raise RuntimeError(reason + '; no motion started')
        command = [str(ROOT / 'brain/pipeline/run_robot_rollout.sh'), 'replay',
                   '--folder', str(args.folder.resolve())]
        if args.send:
            command.append('--send')
        if args.arm_only:
            command.append('--arm-only')
        report['command'] = command
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        os.set_blocking(process.stdout.fileno(), False)
        # Forward the operator's explicit confirmation to the unchanged runner.
        if args.send:
            process.stdin.write(b'SEND\n')
            process.stdin.flush()
        next_sample = 0.
        with (args.out / 'ros.jsonl').open('w') as telemetry, (args.out / 'runner.log').open('wb') as log:
            def drain():
                try:
                    data = os.read(process.stdout.fileno(), 65536)
                except BlockingIOError:
                    return
                if data:
                    log.write(data)
                    log.flush()
                    print(data.decode(errors='replace'), end='', flush=True)

            while process.poll() is None:
                rclpy.spin_once(node, timeout_sec=.005)
                drain()
                now = time.monotonic()
                if now >= next_sample:
                    telemetry.write(json.dumps(clean({'monotonic_s': now, **cache})) + '\n')
                    next_sample = now + .02
                reason = health()
                if reason and not interrupted:
                    report['monitor_stop_reason'] = reason
                    print('MONITOR STOP: ' + reason, flush=True)
                    process.send_signal(signal.SIGINT)
                    interrupted = True
            drain()
            report['runner_exit_code'] = process.returncode
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                print('Runner stop unconfirmed; use physical e-stop.', flush=True)
        raise
    finally:
        report['finished_at_unix_s'] = time.time()
        (args.out / 'report.json').write_text(json.dumps(clean(report), indent=2))
        node.destroy_node()
        rclpy.shutdown()
    raise SystemExit(report.get('runner_exit_code', 1))


if __name__ == '__main__':
    main()
