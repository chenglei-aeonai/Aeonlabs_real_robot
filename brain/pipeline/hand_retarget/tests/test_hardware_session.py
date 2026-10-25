from pathlib import Path
from types import SimpleNamespace
import signal
import time
from unittest.mock import Mock
import pytest
from hand_retarget.hardware_session import HardwareSession


def test_no_process_without_explicit_confirmation(tmp_path,monkeypatch):
    popen=Mock(side_effect=AssertionError('Must not launch hardware'))
    monkeypatch.setattr('hand_retarget.hardware_session.subprocess.Popen',popen)
    session=HardwareSession()
    for word,clear in [('',False),('SEND',False),('send',True)]:
        with pytest.raises(ValueError):session.start(tmp_path,word,clear)
    assert not popen.called


def test_stop_only_interrupts_owned_process_and_browser_timeout(monkeypatch):
    signals=[]
    monkeypatch.setattr('hand_retarget.hardware_session.os.killpg',lambda pid,sig:signals.append((pid,sig)))
    session=HardwareSession();session.stop();assert not signals
    session.process=SimpleNamespace(pid=123456,poll=lambda:None)
    session.heartbeat_at=time.monotonic()-4;session.state='running'
    session._tick()
    assert signals==[(123456,signal.SIGINT)]
    assert session.state=='stopping' and 'heartbeat' in session.stop_reason
    session.stop();assert len(signals)==1
    session.stop_at=time.monotonic()-4;session._tick()
    assert signals[-1]==(123456,signal.SIGKILL) and session.state=='stop_unconfirmed'


def test_heartbeat_does_not_rearm_stopped_or_other_session():
    session=HardwareSession()
    with pytest.raises(ValueError):session.heartbeat('a'*32)
    session.run_id='b'*32;session.process=SimpleNamespace(poll=lambda:None,stdin=SimpleNamespace(write=lambda _:None,flush=lambda:None))
    with pytest.raises(ValueError):session.heartbeat('a'*32)
    session.heartbeat('b'*32)
    assert time.monotonic()-session.heartbeat_at<.1


@pytest.mark.parametrize('arm_only',[False,True])
def test_confirmed_run_and_stop_with_dummy_subprocess_only(tmp_path,monkeypatch,arm_only):
    import subprocess,sys
    original_popen=subprocess.Popen
    script="""import signal,sys,time
assert input().strip()=='SEND'
def stopped(*args):
 print('Robot arm stop confirmed.',flush=True)
 sys.exit(130)
signal.signal(signal.SIGINT,stopped)
print('Physical replay started.',flush=True)
while True:time.sleep(.05)
"""
    def dummy(command,**kwargs):
        assert 'hand_retarget.robot_rollout' in command
        assert ('--arm-only' in command)==arm_only
        return original_popen([sys.executable,'-u','-c',script],**kwargs)
    monkeypatch.setattr('hand_retarget.hardware_session.subprocess.Popen',dummy)
    (tmp_path/'result').mkdir()
    session=HardwareSession()
    try:
        state=session.start(tmp_path,'SEND',True,arm_only=arm_only)
        deadline=time.monotonic()+3
        while session.state!='running' and time.monotonic()<deadline:
            session.heartbeat(state['run_id']);time.sleep(.02)
        assert session.state=='running'
        with pytest.raises(ValueError,match='already active'):session.start(tmp_path,'SEND',True)
        session.stop();session.thread.join(timeout=4)
        assert session.state=='stopped' and session.snapshot()['stop_confirmed']
    finally:session.close()
