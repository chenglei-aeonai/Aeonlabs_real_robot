"""No ROS or hardware connection: validate missing-velocity rejection."""
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

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
