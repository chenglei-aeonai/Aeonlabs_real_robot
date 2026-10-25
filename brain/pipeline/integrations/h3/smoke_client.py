"""Real GPU smoke request, run on the same node as the experimental H3 server."""
import argparse
import base64
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

import numpy as np

PROFILE = "fast-align-hf15-mirrored-12983"
CHECKPOINT_SHA256 = "5553ab5bbe4c1429e9d911b3e19a689e2eecbaa4e981b895849b2617693a161e"


def request(url, data=None):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30121")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1200)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "starting", "bundle": str(args.bundle),
              "input_source": "Existing real human photo + WiLoR bundle; image editing bypassed",
              "profile": PROFILE, "new_gpu_generation": False}
    try:
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            try:
                health = request(args.url + "/health")
            except (urllib.error.URLError, TimeoutError):
                time.sleep(5)
                continue
            if health.get("error"):
                raise RuntimeError(health["error"])
            if health.get("ready"):
                break
            time.sleep(5)
        else:
            raise TimeoutError("H3 failed to become ready")
        if (health.get("checkpoint_profile") != PROFILE
                or not health.get("robot_custom_frames_v1")
                or not health.get("exact_cfg_parallel")
                or health.get("sol_cache_threshold", 0)):
            raise RuntimeError(f"unexpected inference configuration: {health}")
        report["health"] = health
        registered = request(args.url + "/v1/robot-frames",
                             {"bundle_b64": base64.b64encode(args.bundle.read_bytes()).decode()})
        if registered["checkpoint_sha256"] != CHECKPOINT_SHA256:
            raise RuntimeError("checkpoint identity differs from selected deployment")
        report["registered"] = {k: v for k, v in registered.items() if k != "frame_png_b64"}
        spec = {"clip_id": registered["clip_id"], "dataset": "robot_uploads",
                "prompt": "The right hand pinches the edge of the white wrapping paper "
                          "and folds it toward the center. The left hand holds the paper steady. "
                          "The camera stays still.",
                "seed": 1234, "steps": 50, "cfg": 5.,
                "video_shift": 12., "hand_shift": 36., "match_checkpoint_recipe": True}
        job = request(args.url + "/v1/videos", spec)
        report.update(request=spec, job_id=job["id"], status="generating")
        (args.out / "submission.json").write_text(json.dumps(report, indent=2))
        print(f"Submitted real GPU generation {job['id']}", flush=True)
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            status = request(f"{args.url}/v1/videos/{job['id']}")
            if status["status"] == "failed":
                raise RuntimeError(status.get("error") or "GPU generation failed")
            if status["status"] == "completed":
                break
            time.sleep(5)
        else:
            raise TimeoutError("GPU generation did not complete")
        for endpoint, name in (("content", "generated_full.mp4"), ("joints", "joints_full.npz")):
            with urllib.request.urlopen(f"{args.url}/v1/videos/{job['id']}/{endpoint}", timeout=120) as response:
                with (args.out / name).open("wb") as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
        with np.load(args.out / "joints_full.npz", allow_pickle=False) as data, np.load(args.bundle, allow_pickle=False) as anchor:
            joints, K, fps = data["joints"], data["K"], float(data["fps"])
            if joints.shape != (107, 2, 21, 3) or fps != 24 or not np.isfinite(joints).all():
                raise ValueError("unexpected or non-finite generated actions")
            np.testing.assert_allclose(K, anchor["K"], atol=1e-5)
            np.testing.assert_allclose(joints[0, 1], anchor["joints3d"][1], atol=1e-4)
            anchor_error = float(np.abs(joints[0, 1] - anchor["joints3d"][1]).max())
        report.update(status="completed", new_gpu_generation=True, generation=status,
                      right_anchor_max_error_m=anchor_error)
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (args.out / "smoke_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
