"""One explicitly confirmed hardware subprocess, with browser lease and stop control."""
import collections
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid


class HardwareSession:
    def __init__(self):
        self.lock=threading.RLock();self.process=None;self.run_id=None;self.job_id=None
        self.state='idle';self.lines=collections.deque(maxlen=120);self.heartbeat_at=0.;self.stop_at=None
        self.stop_reason=None;self.thread=None

    @property
    def active(self):
        with self.lock:return self.process is not None and self.process.poll() is None

    def snapshot(self):
        with self.lock:
            return {'run_id':self.run_id,'job_id':self.job_id,'state':self.state,'active':self.active,
                    'stop_reason':self.stop_reason,'log':'\n'.join(self.lines),
                    'stop_confirmed':any('Robot arm stop confirmed.' in x for x in self.lines),
                    'browser_lease_s':3}

    def start(self,job,confirmation,workspace_clear,arm_only=False):
        if confirmation!='SEND' or workspace_clear is not True:
            raise ValueError('Confirm the clear workspace and type SEND before physical execution.')
        with self.lock:
            if self.active or (self.thread is not None and self.thread.is_alive()):
                raise ValueError('A hardware replay is already active or finishing its stop checks.')
            self.run_id=uuid.uuid4().hex;self.job_id=job.name;self.lines.clear();self.stop_at=None;self.stop_reason=None
            self.state='starting';self.heartbeat_at=time.monotonic()
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            self.process=subprocess.Popen([sys.executable,'-u','-m','hand_retarget.robot_rollout','replay','--folder',str(job/'result'),'--send','--web-session']+(['--arm-only'] if arm_only else []),
                                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          text=True,bufsize=1,start_new_session=True,env=env)
            process=self.process
            process.stdin.write('SEND\n');process.stdin.flush()
            self.thread=threading.Thread(target=self._monitor,args=(process,job/'result'/f'hardware_{self.run_id}.log'),daemon=True)
            self.thread.start()
            return self.snapshot()

    def heartbeat(self,run_id):
        with self.lock:
            if run_id!=self.run_id or not self.active:raise ValueError('This hardware session is no longer active.')
            try:
                self.process.stdin.write('HEARTBEAT\n');self.process.stdin.flush()
            except (OSError,ValueError) as error:
                self.stop('Hardware control pipe failed')
                raise ValueError('Hardware runner is no longer accepting heartbeats.') from error
            self.heartbeat_at=time.monotonic()
        return self.snapshot()

    def stop(self,reason='Operator pressed Stop robot'):
        with self.lock:
            if not self.active:return self.snapshot()
            if self.stop_at is None:
                self.stop_at=time.monotonic();self.stop_reason=reason;self.state='stopping'
                try:os.killpg(self.process.pid,signal.SIGINT)
                except ProcessLookupError:pass
        return self.snapshot()

    def _tick(self):
        with self.lock:
            if self.active and self.stop_at is None and time.monotonic()-self.heartbeat_at>3:
                self.stop('Browser heartbeat lost; replay interrupted')
            if self.active and self.stop_at is not None and time.monotonic()-self.stop_at>3:
                # Last resort: controller-side streaming watchdog must stop motion.
                self.state='stop_unconfirmed'
                try:os.killpg(self.process.pid,signal.SIGKILL)
                except ProcessLookupError:pass

    def _monitor(self,process,log_path):
        def read_output():
            with log_path.open('w') as log:
                for raw in process.stdout:
                    log.write(raw);log.flush()
                    with self.lock:
                        self.lines.append(raw.rstrip())
                        if 'Physical replay started.' in raw and self.stop_at is None:self.state='running'
        reader=threading.Thread(target=read_output,daemon=True);reader.start()
        while process.poll() is None:self._tick();time.sleep(.1)
        reader.join(timeout=2)
        try:process.stdin.close()
        except (OSError,ValueError):pass
        with self.lock:
            confirmed=any('Robot arm stop confirmed.' in x for x in self.lines)
            started=any('Physical replay started.' in x for x in self.lines)
            if self.state=='stop_unconfirmed' or (started and not confirmed):self.state='stop_unconfirmed'
            elif self.stop_at is not None:self.state='stopped' if confirmed else 'cancelled'
            else:self.state='completed' if process.returncode==0 and confirmed else 'failed'

    def close(self):
        self.stop('Web server shutting down')
        if self.thread is not None:self.thread.join(timeout=5)
