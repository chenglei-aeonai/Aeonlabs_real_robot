import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import httpx
import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from hand_retarget.segmentation import polygon_mask, segment_hands
from hand_retarget.webapp import Config, JobInput, build_command, create_app, import_result, save_image, write_json
from hand_retarget.h3_actions import fixed_camera_transform
from hand_retarget.skeleton import load_world_joints

ROOT = Path(__file__).resolve().parents[2]
HEADERS = {"X-Retarget-Client": "1"}


def encoded_image(mask=False):
    image = Image.new("RGB", (64, 48), (100, 110, 120))
    if mask:
        image = Image.new("L", (64, 48), 0)
        image.paste(255, (10, 10, 30, 30))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def payload():
    return dict(image_b64=encoded_image(), instruction="Lift the cup.", hfov=70,
                fixed_camera_pitch=0, mask_b64=encoded_image(True),
                edited_image_b64=encoded_image())


@pytest.fixture
def config(tmp_path):
    return Config(tmp_path / "runs", uplift_python=sys.executable, uplift_script=__file__, image_provider='gemini')


def test_upload_and_origin_limits(config):
    with TestClient(create_app(config)) as client:
        assert client.get("/").status_code == 200
        assert "From observation to action" in client.get("/").text
        assert client.get("/app.js").status_code == 200
        assert client.get("/i18n.js").status_code == 200
        assert client.get("/aeonlabs-wordmark.png").headers["content-type"] == "image/png"
        assert client.get("/", headers={"Host": "untrusted.example"}).status_code == 400
        assert client.post("/api/jobs", json=payload()).status_code == 403
        request = payload()
        request["image_b64"] = "not an image"
        assert client.post("/api/jobs", json=request, headers=HEADERS).status_code == 400
        request = payload()
        request["camera_to_demo"] = np.eye(4).tolist()
        assert client.post("/api/jobs", json=request, headers=HEADERS).status_code == 400
        request = payload()
        request["instruction"] = " "
        assert client.post("/api/jobs", json=request, headers=HEADERS).status_code == 400
        assert client.get("/api/jobs").json() == []
        assert client.get("/api/jobs/not-a-job/files/manifest.json").status_code == 404


def test_missing_editor_never_queues(config, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    request = payload()
    request["edited_image_b64"] = None
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json=request, headers=HEADERS)
        assert response.status_code == 400
        assert "GEMINI_API_KEY" in response.json()["detail"]
        assert not client.get("/api/jobs").json()


def test_binary_mask_and_dimensions(tmp_path):
    with pytest.raises(ValueError, match="binary"):
        save_image(encoded_image(), tmp_path / "mask.png", mask=True)
    assert save_image(encoded_image(True), tmp_path / "mask.png", mask=True) == (64, 48)


def test_command_has_literal_user_prompt_and_calibration(config, tmp_path):
    request = JobInput(**{**payload(), "instruction": "$(touch /tmp/should-never-exist); test"})
    (tmp_path / "mask.png").touch()
    (tmp_path / "edited_input.png").touch()
    cmd = build_command(config, tmp_path, request)
    assert cmd[cmd.index("--instruction") + 1] == request.instruction
    assert "--mask" in cmd and "--edited-image" in cmd
    assert cmd[cmd.index("--seconds") + 1] == "4"


def test_restart_marks_unfinished_and_hides_commands(config):
    job = config.runs / ("a" * 32)
    (job / "result").mkdir(parents=True)
    write_json(job / "status.json", {"id": job.name, "status": "running"})
    (job / "result/robot_actions.npz").touch()
    with TestClient(create_app(config)) as client:
        result = client.get(f"/api/jobs/{job.name}").json()
        assert result["status"] == "interrupted"
        assert "robot_actions.npz" not in result["artifacts"]
        assert client.get(f"/api/jobs/{job.name}/files/robot_actions.npz").status_code == 404


def test_auto_region_contract_and_failure_modes(tmp_path, monkeypatch):
    image = tmp_path / "robot.png"
    save_image(encoded_image(), image)
    monkeypatch.setenv("GEMINI_API_KEY", "test-not-a-real-key")
    regions = {"hands": [{"polygon": [[100, 100], [400, 100], [400, 400], [100, 400]]}]}
    def handler(request):
        assert request.headers["x-goog-api-key"] == "test-not-a-real-key"
        return httpx.Response(200, json={"candidates": [{"content": {
            "parts": [{"text": json.dumps(regions)}]}}]})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = segment_hands(image, tmp_path / "mask.png", client=client, provider='gemini')
    assert result["geometrically_verified"] is False
    assert set(np.unique(Image.open(tmp_path / "mask.png"))) == {0, 255}
    for bad in ({}, {"hands": [{"polygon": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]}]},
                {"hands": [{"polygon": [[-1, 0], [1, 2], [2, 4]]}]}):
        with pytest.raises(ValueError):
            polygon_mask((64, 48), bad)


def test_web_queue_real_retarget_and_controller_compatibility(config, tmp_path, monkeypatch):
    """Real subprocess/solver/exporter; upstream GPU generation deliberately replaced."""
    from hand_retarget import webapp
    demo = load_world_joints(ROOT / "hand_retarget/data/20260928T132405_scene0_world_joints.csv")
    right = np.repeat(demo.skeletons[:1], 107, axis=0)
    right[:, :, 0] += np.linspace(0, .02, 107)[:, None]
    right = right @ fixed_camera_transform(0)[:3, :3] + [0, 0, 1]
    joints = tmp_path / "prediction.npz"
    np.savez(joints, joints=np.stack([np.zeros_like(right), right], axis=1),
             fps=24, K=np.array([[500, 0, 32], [0, 500, 24], [0, 0, 1]]))
    monkeypatch.setattr(webapp, "build_command", lambda cfg, job, request: [
        sys.executable, "-m", "hand_retarget.inference", "retarget", "--joints", str(joints),
        "--fixed-camera-pitch", "0", "--out", str(job / "result")])
    with TestClient(create_app(config)) as client:
        response = client.post("/api/jobs", json=payload(), headers=HEADERS)
        assert response.status_code == 202, response.text
        job_id = response.json()["id"]
        for _ in range(300):
            result = client.get(f"/api/jobs/{job_id}").json()
            if result["status"] in ("failed", "completed"):
                break
            time.sleep(.05)
        assert result["status"] == "completed", result
        assert result["metrics"]["exports"]["samples"] == 241
        files = result["artifacts"]
        trajectory = client.get(files["trajectory.json"]).json()
        assert len(trajectory["joint_names"]) == 26
        assert trajectory["points"][-1]["time_from_start"] == 4
        commands = client.get(files["controller_commands.json"]).json()
        assert len(commands["hand"]["scheduled_messages"]) == 241
        assert commands["hand"]["scheduled_messages"][-1]["payload"]["sequence"] == 241
        assert commands["arm"]["payload"]["time"][-1] == 4
        with zipfile.ZipFile(io.BytesIO(client.get(files["bundle.zip"]).content)) as bundle:
            assert "robot_actions.npz" in bundle.namelist()
        spec = importlib.util.spec_from_file_location("replay_web_check", ROOT / "real_robot/replay_real.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # This validates the real driver's file admission; no sockets or hardware are opened.
        replay = module.Demonstration(config.runs / job_id / "result/robot_actions.npz", 1)
        assert replay.arm.shape == (241, 6) and replay.hand.shape == (241, 20)
        np.testing.assert_allclose(trajectory["points"][0]["positions"], np.r_[replay.arm[0], replay.hand[0]])
        imported = import_result(config, config.runs / job_id / "result", "Saved regression")
        state = client.get(f"/api/jobs/{imported}").json()
        assert state["mode"] == "imported_result" and state["input_url"] is None
        assert client.get(f"/api/jobs/{imported}/input").status_code == 404
        assert client.get(state["artifacts"]["bundle.zip"]).status_code == 200


def test_worker_failure_is_persistent(config, monkeypatch):
    from hand_retarget import webapp
    monkeypatch.setattr(webapp, "build_command", lambda *_: [sys.executable, "-c", "raise SystemExit(9)"])
    with TestClient(create_app(config)) as client:
        job_id = client.post("/api/jobs", json=payload(), headers=HEADERS).json()["id"]
        for _ in range(100):
            result = client.get(f"/api/jobs/{job_id}").json()
            if result["status"] == "failed":
                break
            time.sleep(.02)
        assert result["status"] == "failed"
        assert "status 9" in result["error"]
        assert not result["artifacts"]


def test_shutdown_stops_process_group_and_does_not_launch_queued_jobs(config, tmp_path, monkeypatch):
    from hand_retarget import webapp
    started = tmp_path / "started"
    descendant_output = tmp_path / "descendant-output"
    queued_output = tmp_path / "queued-output"
    # A child process would write a marker after the service has shut down if
    # cancellation killed only the pipeline parent rather than its process group.
    child = ("import time; from pathlib import Path; time.sleep(1); "
             f"Path({str(descendant_output)!r}).touch()")
    parent = ("import subprocess, sys, time; from pathlib import Path; "
              f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
              f"Path({str(started)!r}).touch(); time.sleep(60)")
    commands = iter([
        [sys.executable, "-c", parent],
        [sys.executable, "-c", f"from pathlib import Path; Path({str(queued_output)!r}).touch()"],
    ])
    monkeypatch.setattr(webapp, "build_command", lambda *_: next(commands))
    app = create_app(config)
    with TestClient(app) as client:
        first = client.post("/api/jobs", json=payload(), headers=HEADERS).json()["id"]
        second = client.post("/api/jobs", json=payload(), headers=HEADERS).json()["id"]
        deadline = time.monotonic() + 5
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert started.exists()
        begin = time.monotonic()
    assert time.monotonic() - begin < 5
    time.sleep(1.1)
    assert not descendant_output.exists()
    assert not queued_output.exists()
    for job_id in (first, second):
        state = json.loads((config.runs / job_id / "status.json").read_text())
        assert state["status"] == "interrupted"
    assert app.state.jobs.pending == 0


@pytest.mark.parametrize("bad", ["bounds", "speed", "time", "nan"])
def test_controller_export_rejects_invalid_trajectories(tmp_path, bad):
    from hand_retarget.trajectory import export_trajectory
    from hand_retarget.robot import HAND_LOWER, HAND_UPPER
    arm = np.zeros((3, 6))
    hand = np.tile((HAND_LOWER + HAND_UPPER) / 2, (3, 1))
    t = np.arange(3) / 60
    if bad == "bounds":
        arm[:, 2] = 4
    elif bad == "speed":
        arm[1, 0] = .2
    elif bad == "time":
        t[:] = 0
    else:
        hand[0, 0] = np.nan
    np.savez(tmp_path / "robot_actions.npz", timestamp=t, arm_q=arm, hand_q_measured=hand,
             arm_dq=np.zeros_like(arm), hand_dq_measured=np.zeros_like(hand))
    with pytest.raises(ValueError):
        export_trajectory(tmp_path, {"placement_mode": "upstream_home_pose", "placement": np.eye(4).tolist()})
    assert not (tmp_path / "controller_commands.json").exists()
