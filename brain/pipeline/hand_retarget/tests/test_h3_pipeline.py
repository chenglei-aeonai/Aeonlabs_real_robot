import base64
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import httpx
import numpy as np
from PIL import Image
import pytest

from hand_retarget.h3_actions import (
    fixed_camera_transform, load_h3_actions, retarget_actions, write_world_csv,
)
from hand_retarget.inference import trim_video
from hand_retarget.inference_clients import H3Client, replace_hands
from hand_retarget.skeleton import load_world_joints

ROOT = Path(__file__).resolve().parents[2]
DEMO = ROOT / "hand_retarget/data/20260928T132405_scene0_world_joints.csv"


@pytest.fixture
def prediction(tmp_path):
    demo = load_world_joints(DEMO)
    # Known skeleton + motion transformed into optical camera coordinates.
    right = np.repeat(demo.skeletons[:1], 107, axis=0)
    right[:, :, 0] += np.linspace(0, .05, 107)[:, None]
    right = right @ fixed_camera_transform(0)[:3, :3] + [0, 0, 1]
    joints = np.stack([np.zeros_like(right), right], axis=1)
    K = np.array([[500, 0, 416], [0, 500, 240], [0, 0, 1]])
    path = tmp_path / "prediction.npz"
    np.savez(path, joints=joints, fps=24., K=K)
    return path, joints, K


def test_optical_axes_and_pitch():
    R = fixed_camera_transform(0)[:3, :3]
    np.testing.assert_allclose(R @ [0, 0, 1], [1, 0, 0])
    np.testing.assert_allclose(R @ [1, 0, 0], [0, -1, 0])
    R = fixed_camera_transform(30)[:3, :3]
    np.testing.assert_allclose(R @ [0, 0, 1], [np.sqrt(3)/2, 0, -.5], atol=1e-12)


def test_right_hand_order_motion_and_csv_roundtrip(prediction, tmp_path):
    path, joints, _ = prediction
    demo, meta = load_h3_actions(path, 4, fixed_camera_transform(0))
    expected = (joints[:96, 1] - joints[0, 1, 0]) @ fixed_camera_transform(0)[:3, :3].T
    np.testing.assert_allclose(demo.skeletons, expected, atol=1e-12)
    assert demo.skeletons[-1, 0, 0] > .04  # wrist displacement must not disappear
    assert meta["video_frames"] == 96
    csv_path = tmp_path / "human_world_joints.csv"
    write_world_csv(demo, csv_path)
    np.testing.assert_allclose(load_world_joints(csv_path).skeletons, expected)


@pytest.mark.parametrize("failure", ["missing", "nan", "collapsed", "short", "reflection", "fps"])
def test_reject_invalid_actions(prediction, failure):
    path, joints, K = prediction
    transform, fps = fixed_camera_transform(0), 24.
    if failure == "missing":
        joints[:, 1] = 0
    elif failure == "nan":
        joints[15, 1, 2, 0] = np.nan
    elif failure == "collapsed":
        joints[:, 1, 2] = joints[:, 1, 1]
    elif failure == "short":
        joints = joints[:80]
    elif failure == "reflection":
        transform[0] *= -1
    elif failure == "fps":
        fps = float("nan")
    np.savez(path, joints=joints, fps=fps, K=K)
    with pytest.raises(ValueError):
        load_h3_actions(path, 4, transform)


def test_moving_camera_translation_is_preserved(prediction):
    path, _, _ = prediction
    transforms = np.repeat(fixed_camera_transform(0)[None], 107, axis=0)
    transforms[:, 1, 3] = np.linspace(0, .1, 107)
    moving, _ = load_h3_actions(path, 4, transforms)
    fixed, _ = load_h3_actions(path, 4, fixed_camera_transform(0))
    np.testing.assert_allclose(moving.skeletons[-1, :, 1] - fixed.skeletons[-1, :, 1], 95/106*.1)


def test_calibrated_postprocess_preserves_scene_origin(prediction, tmp_path, monkeypatch):
    from hand_retarget import inference, trajectory
    path, joints, _ = prediction
    camera = fixed_camera_transform(20)
    camera[:3, 3] = [.2, -.3, .8]
    robot_from_demo = np.eye(4)
    robot_from_demo[:3, 3] = [.1, .6, .55]
    captured = {}
    def retarget(demo, out, method, placement):
        captured["world"] = demo.skeletons @ placement[:3, :3].T + placement[:3, 3]
        captured["camera_position"] = demo.camera_position
        return {}
    monkeypatch.setattr(inference, "retarget_actions", retarget)
    monkeypatch.setattr(trajectory, "export_trajectory", lambda *args: {})
    result = inference.postprocess(path, None, tmp_path, 4, camera, "vector", robot_from_demo)
    combined = robot_from_demo @ camera
    expected = joints[:96, 1] @ combined[:3, :3].T + combined[:3, 3]
    np.testing.assert_allclose(captured["world"], expected, atol=1e-12)
    np.testing.assert_allclose(captured["camera_position"], camera[:3, 3])
    assert result["human"]["origin_centered"] is False
    assert result["human"]["origin_removed"] == [0, 0, 0]


def test_actual_retargeter_exports_four_seconds(prediction, tmp_path):
    from hand_retarget.robot import HAND_LOWER, HAND_UPPER
    from hand_retarget.hand_optimizer import MAX_JOINT_SPEED
    demo, _ = load_h3_actions(prediction[0], 4, fixed_camera_transform(0))
    result = retarget_actions(demo, tmp_path, "vector")
    with np.load(result["robot_actions"], allow_pickle=False) as data:
        assert data["arm_q"].shape == (241, 6)
        assert data["hand_q_measured"].shape == (241, 20)
        assert data["timestamp"][-1] == 4.
        assert np.isfinite(data["arm_q"]).all()
        assert np.all(data["hand_q_measured"] >= HAND_LOWER - 1e-8)
        assert np.all(data["hand_q_measured"] <= HAND_UPPER + 1e-8)
        assert np.max(np.abs(data["hand_dq_measured"])) <= MAX_JOINT_SPEED + 1e-6
    assert result["placement_mode"] == "upstream_home_pose"


def test_image_edit_preserves_background_exactly(tmp_path):
    image, mask, edited = (tmp_path / n for n in ("original.png", "mask.png", "edited.png"))
    Image.new("RGB", (16, 16), "blue").save(image)
    Image.new("RGB", (16, 16), "red").save(edited)
    allowed = np.zeros((16, 16), np.uint8)
    allowed[4:12, 4:12] = 255
    Image.fromarray(allowed).save(mask)
    out = tmp_path / "out.png"
    replace_hands(image, mask, out, model="unused", edited_image=edited)
    actual = np.asarray(Image.open(out))
    np.testing.assert_array_equal(actual[allowed == 0], np.tile([0, 0, 255], (192, 1)))
    np.testing.assert_array_equal(actual[allowed == 255], np.tile([255, 0, 0], (64, 1)))


def test_h3_uses_actual_queue_contract_and_downloads(tmp_path):
    calls = []
    def handle(request):
        calls.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"ready": True, "checkpoint_profile": "chosen",
                                            "robot_custom_frames_v1": True, "exact_cfg_parallel": True})
        if request.method == "POST":
            body = json.loads(request.content)
            assert body["clip_id"] == "robot::abc" and body["prompt"] == "grasp the cup"
            assert body["hand_shift"] == 36 and "num_frames" not in body
            return httpx.Response(200, json={"id": "job1"})
        if request.url.path == "/v1/videos/job1":
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, content=b"artifact")
    client = H3Client("http://h3", "chosen", client=httpx.Client(transport=httpx.MockTransport(handle)))
    client.preflight()
    client.generate("robot::abc", "grasp the cup", 7, tmp_path)
    assert (tmp_path / "joints_full.npz").read_bytes() == b"artifact"
    assert (tmp_path / "h3_submission.json").exists()
    assert ("GET", "/v1/videos/job1/joints") in calls


@pytest.mark.parametrize("health", [
    {"ready": False},
    {"ready": True, "checkpoint_profile": "wrong"},
    {"ready": True, "checkpoint_profile": "chosen"},
    {"ready": True, "checkpoint_profile": "chosen", "robot_custom_frames_v1": True,
     "exact_cfg_parallel": True, "sol_cache_threshold": .1},
])
def test_h3_preflight_rejects_wrong_or_incompatible_engine(health):
    client = H3Client("http://h3", "chosen", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=health))))
    with pytest.raises(RuntimeError):
        client.preflight()


def test_trim_video_keeps_matching_96_frames(tmp_path):
    import imageio_ffmpeg
    source, target = tmp_path / "source.mp4", tmp_path / "video.mp4"
    writer = imageio_ffmpeg.write_frames(str(source), (32, 32), fps=24)
    writer.send(None)
    for _ in range(107):
        writer.send(np.zeros((32, 32, 3), np.uint8))
    writer.close()
    trim_video(source, target, 96, 24)
    reader = imageio_ffmpeg.read_frames(str(target))
    metadata = next(reader)
    assert metadata["fps"] == 24
    assert sum(1 for _ in reader) == 96


@pytest.fixture
def extension():
    path = ROOT / "integrations/h3/robot_custom_frames.py"
    spec = importlib.util.spec_from_file_location("robot_custom_frames_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def bundle_bytes(joints, K):
    buffer = io.BytesIO()
    np.savez(buffer, frame=np.zeros((480, 832, 3), np.uint8),
             joints3d=joints[0], wrist_R=np.tile(np.eye(3), (2, 1, 1)),
             valid=np.array([False, True]), K=K)
    return buffer.getvalue()


def test_custom_frame_registration_checkpoint_guard_and_staging(extension, prediction, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    monkeypatch.setenv("H3_SERVE_PROFILE", extension.PROFILE)
    manifest = tmp_path / "demo-checkpoint-manifest.json"
    manifest.write_text(json.dumps({"source_checkpoint_sha256": extension.CHECKPOINT_SHA256}))
    engine = SimpleNamespace(ready=True, ckpt=str(tmp_path / "checkpoint.safetensors"),
                             has_clip=lambda c, d=None: False, datasets=lambda: [],
                             base_clip_id=lambda c: c, first_frame=lambda c: None,
                             clip_caption=lambda c: "")
    app = FastAPI()
    extension.install_routes(app, engine)
    engine.ckpt = "/dev/shm/staged/checkpoint.safetensors"
    client = TestClient(app)
    payload = bundle_bytes(prediction[1], prediction[2])
    response = client.post("/v1/robot-frames", json={"bundle_b64": base64.b64encode(payload).decode()})
    assert response.status_code == 200, response.text
    cid = response.json()["clip_id"]
    assert engine.has_clip(cid, "robot_uploads")
    assert not engine.has_clip(cid, "oakink2")
    assert engine.first_frame(cid).shape == (480, 832, 3)
    manifest.write_text('{"source_checkpoint_sha256":"wrong"}')
    assert client.post("/v1/robot-frames", json={"bundle_b64": base64.b64encode(payload).decode()}).status_code == 409


def test_anchor_roundtrip_with_real_h3_decoder(extension, prediction, monkeypatch):
    runtime = os.environ.get("H3_TEST_RUNTIME")
    if not runtime:
        pytest.skip("set H3_TEST_RUNTIME to test the inspected H3 runtime")
    monkeypatch.syspath_prepend(str(Path(runtime) / "training"))
    from h3_joint.representations import decode_v2_abspose
    bundle = extension.load_bundle(bundle_bytes(prediction[1], prediction[2]))
    anchor, mask = extension.anchor_and_mask(bundle)
    decoded, _ = decode_v2_abspose(anchor)
    np.testing.assert_allclose(decoded[0, 1].numpy(), prediction[1][0, 1], atol=1e-6)
    assert mask.sum() == 69


def test_custom_conditioning_uses_uploaded_rgb_for_both_cfg_branches(extension, prediction, monkeypatch):
    runtime = os.environ.get("H3_TEST_RUNTIME")
    if not runtime:
        pytest.skip("set H3_TEST_RUNTIME to test the inspected H3 runtime")
    monkeypatch.syspath_prepend(str(Path(runtime) / "training"))
    import threading
    import torch
    import h3_joint.packing as packing
    # Token packing has its own upstream tests; test the real conditioning builder here.
    monkeypatch.setattr(packing, "extend_packed_with_hand_latents", lambda packed, **kw: packed)
    bundle = extension.load_bundle(bundle_bytes(prediction[1], prediction[2]))
    cond_embed, uncond_embed = torch.ones(2, 3), torch.zeros(2, 3)
    def encode(frame, prompt, n_frames):
        np.testing.assert_array_equal(frame, bundle["frame"])
        assert prompt == "grasp the cup" and n_frames == 107
        return ({"audio_input_latents": torch.zeros(1, 8, 10),
                 "input_latents": torch.zeros(1, 16, 32, 4, 4)},
                cond_embed, {"branch": "cond"}, uncond_embed, {"branch": "uncond"})
    engine = SimpleNamespace(_robot_frames={"robot::test": bundle},
                             _robot_frame_lock=threading.Lock(),
                             dataset=SimpleNamespace(stats={}, hand_tag=2), encode_continuation=encode)
    cond, uncond = extension.conditioning_items(engine, "robot::test", "grasp the cup")
    assert cond["prompt_embeds"] is cond_embed
    assert uncond["prompt_embeds"] is uncond_embed
    assert cond["motion_embodiment"] == cond["shared"]["motion_embodiment"] == 0
    np.testing.assert_array_equal(cond["shared"]["cam_K"].numpy(), bundle["K"])
    assert cond["skel_raw"].shape == (107, 138)
    assert torch.count_nonzero(cond["skel_raw"][1:, :6]) == 0
    assert torch.all(cond["chan_valid"].sum(1) == 69)
    assert cond["shared"] is uncond["shared"]


def test_install_patch_applies_exactly_once(tmp_path):
    runtime = os.environ.get("H3_TEST_RUNTIME")
    if not runtime:
        pytest.skip("set H3_TEST_RUNTIME to test patch against the inspected H3 runtime")
    spec = importlib.util.spec_from_file_location("install_h3_test", ROOT / "integrations/h3/install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = (Path(runtime) / "training/h3_joint/serve_hands.py").read_text()
    patched = module.patched_source(source)
    assert 'conditioning_items(self, cid, conditions[0].prompt)' in patched
    with pytest.raises(ValueError, match="mismatch/already patched"):
        module.patched_source(patched)


@pytest.mark.parametrize("failed_job", [False, True])
def test_gpu_smoke_client_records_success_or_failure(prediction, tmp_path, monkeypatch, failed_job):
    spec = importlib.util.spec_from_file_location("gpu_smoke_test", ROOT / "integrations/h3/smoke_client.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bundle = tmp_path / "frame.npz"
    bundle.write_bytes(bundle_bytes(prediction[1], prediction[2]))
    out = tmp_path / "gpu-result"
    monkeypatch.setattr(sys, "argv", ["smoke_client.py", "--bundle", str(bundle), "--out", str(out)])
    def request(url, data=None):
        if url.endswith("/health"):
            return {"ready": True, "checkpoint_profile": module.PROFILE,
                    "robot_custom_frames_v1": True, "exact_cfg_parallel": True}
        if url.endswith("/robot-frames"):
            assert data["bundle_b64"]
            return {"clip_id": "robot::test", "checkpoint_sha256": module.CHECKPOINT_SHA256}
        if url.endswith("/videos"):
            assert data["dataset"] == "robot_uploads" and data["hand_shift"] == 36
            return {"id": "real-queue-contract"}
        return {"status": "failed", "error": "GPU fault"} if failed_job else {"status": "completed"}
    monkeypatch.setattr(module, "request", request)
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda url, **kw: io.BytesIO(prediction[0].read_bytes()
                                                    if url.endswith("/joints") else b"video"))
    if failed_job:
        with pytest.raises(RuntimeError, match="GPU fault"):
            module.main()
    else:
        module.main()
    report = json.loads((out / "smoke_report.json").read_text())
    assert report["status"] == ("failed" if failed_job else "completed")
    assert report["new_gpu_generation"] is not failed_job
    assert (out / "submission.json").exists()
