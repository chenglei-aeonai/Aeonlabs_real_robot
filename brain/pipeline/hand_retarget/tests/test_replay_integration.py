"""Publication regressions: UI/API wiring and offline diagnostics, no robot sends."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hand_retarget.webapp import Config, create_app

PIPELINE = Path(__file__).resolve().parents[2]
HEADERS = {'X-Retarget-Client':'1'}


def test_hardware_api_requires_prepared_result_and_confirmation(tmp_path, monkeypatch):
    app = create_app(Config(tmp_path/'runs'))
    jobs = app.state.jobs
    identifier = 'a'*32
    job = jobs.config.runs/identifier
    (job/'result').mkdir(parents=True)
    (job/'status.json').write_text(json.dumps({'status':'completed','mode':'robot_rollout'}))
    (job/'result/manifest.json').write_text(json.dumps({'execution_ready':True}))
    popen = Mock(side_effect=AssertionError('Must never start real hardware in test'))
    monkeypatch.setattr('hand_retarget.hardware_session.subprocess.Popen', popen)
    with TestClient(app) as client:
        assert client.get('/api/robot/status').json()['active'] is False
        response = client.post('/api/robot/run',headers=HEADERS,
            json={'source_job_id':identifier,'confirmation':'','workspace_clear':True})
        assert response.status_code == 400
        assert 'SEND' in response.json()['detail']
        assert not popen.called
        started = Mock(return_value={'active':True,'state':'starting'})
        monkeypatch.setattr(jobs.hardware, 'start', started)
        response = client.post('/api/robot/run',headers=HEADERS,
            json={'source_job_id':identifier,'confirmation':'SEND','workspace_clear':True})
        assert response.status_code == 202
        started.assert_called_once_with(job, 'SEND', True, arm_only=False)
        jobs.pending = 1
        assert client.post('/api/robot/run',headers=HEADERS,
            json={'source_job_id':identifier,'confirmation':'SEND','workspace_clear':True}).status_code == 409
        jobs.pending = 0


def test_ui_loads_execution_controls_and_exposes_local_timing(tmp_path):
    app = create_app(Config(tmp_path/'runs'))
    with TestClient(app) as client:
        html = client.get('/').text
        for control in ('rollout-prepare','rollout-source','rollout-result','robot-run',
                        'robot-stop','robot-confirm-dialog','robot-confirm-text','retarget-demo'):
            assert f'id="{control}"' in html
        script = client.get('/rollout.js')
        assert script.status_code == 200
        assert script.headers['cache-control'] == 'no-store'
        assert 'local_time_allocation' in script.text
        assert '/api/robot/heartbeat' in script.text
        assert '/api/robot/stop' in script.text


def test_recording_audit_uses_current_reader_without_opening_sockets():
    spec = importlib.util.spec_from_file_location('audit', PIPELINE/'scripts/audit_replay_guard.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, reader, stop = module.readers()
    packet = {'type':'hand_state','positions':[0.]*20,'velocities':[0.]*20,
              'measured_age_s':0.}
    np.testing.assert_array_equal(module.feed(reader,packet), np.zeros(20))
    packet['velocities'][3] = .8
    with pytest.raises(stop, match='Finger overspeed'):
        module.feed(reader,packet)
