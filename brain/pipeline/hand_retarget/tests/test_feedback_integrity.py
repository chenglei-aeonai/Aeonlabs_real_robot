"""Exercise hardware reader methods offline, without importing ROS or opening sockets."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import zmq
from hand_retarget import robot_rollout

ROOT=Path(robot_rollout.__file__).resolve().parents[5]


def reader_class():
    tree=ast.parse(Path(robot_rollout.__file__).read_text())
    replay=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
    node=next(n for n in replay.body if isinstance(n,ast.ClassDef) and n.name=='Arm')
    env={'np':np,'time':time,'zmq':zmq,'armed':True,'native':True,'fixed_pose':False,
         'transport':SimpleNamespace(Arm=object)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<real-arm-reader>','exec'),env)
    return env['Arm']


@pytest.mark.parametrize('change', [
    {'robot_state_age_s':.4}, {'rtde_connected':False}, {'robot_state_age_s':float('nan')},
    {'protective_stopped':None}, {'Qd':[float('nan')]*6},
])
def test_fresh_publisher_cannot_hide_invalid_hardware_feedback(change):
    packet={'timestamp_ms':time.time()*1000,'Q':[0.]*6,'Qd':[0.]*6,
            'is_truncated':False,'protective_stopped':False,'emergency_stopped':False,
            'joint_target_watchdog_s':.2,'robot_state_age_s':0.,'rtde_connected':True}
    packet.update(change)
    cls=reader_class();reader=cls()
    reader.state=SimpleNamespace(recv_json=lambda **kwargs:packet)
    with pytest.raises((RuntimeError,ValueError)):
        reader.poll()


def test_missing_hand_velocity_does_not_refresh_measurement_age():
    path=ROOT/'hand/dg5f_policy_ros_bridge.py'
    tree=ast.parse(path.read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Dg5fPolicyBridge')
    method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_on_joint_state')
    names=[f'rj_dg_{f}_{j}' for f in range(1,6) for j in range(1,5)]
    env={'np':np,'time':time,'JOINT_NAMES':names,'JointState':object}
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    reader=SimpleNamespace(last_state_at=-1.,measured_q=None,measured_qd=None)
    message=SimpleNamespace(name=names,position=[0.]*20,velocity=[],effort=[])
    env['_on_joint_state'](reader,message)
    assert reader.last_state_at == -1. and reader.measured_q is None
    message.velocity=[0.]*20
    env['_on_joint_state'](reader,message)
    assert reader.last_state_at > 0 and reader.measured_qd.shape == (20,)
