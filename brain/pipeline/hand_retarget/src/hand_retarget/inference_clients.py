"""Concrete image-edit and H3 HTTP clients. Credentials are read from the environment."""
from __future__ import annotations

import base64
import io
import os
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageOps

EDIT_PROMPT = """Edit the first-person photograph. Replace only the robotic hand
end effectors with anatomically plausible human hands of the same handedness,
pose, size, perspective and contact locations. Keep the robot arms, wrist mounting
positions, objects, background, camera, lighting and framing unchanged.
The second image is an edit mask: white is the allowed hand replacement area,
black must remain unchanged. Return a single edited photograph without the mask."""


def png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def read_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def replace_hands(image: Path, mask: Path, out: Path, *, model: str,
                  edited_image: Path | None = None, client=None, provider: str = 'codex',
                  capture_json: Path | None = None) -> dict:
    """Hard-composite the edit inside the mask so outside pixels are identical."""
    original = read_rgb(image)
    with Image.open(mask) as source:
        allowed = ImageOps.exif_transpose(source).convert("L")
    values = np.asarray(allowed)
    if allowed.size != original.size or not np.isin(values, [0, 255]).all():
        raise ValueError("mask must match the oriented image size and contain only 0/255")
    if not (values == 255).any() or not (values == 0).any():
        raise ValueError("mask must contain both editable hand pixels and protected pixels")
    if edited_image is not None:
        edited = read_rgb(edited_image)
        provider = "provided_edited_image"
        details = {}
    elif provider == 'codex':
        from .codex_edit import edit_image
        candidate = out.with_name(out.stem + '_codex_candidate.png')
        if candidate.exists():
            raise ValueError(f'Refusing to overwrite existing edit candidate: {candidate}')
        details = edit_image(image, mask, candidate, capture_json=capture_json)
        edited = read_rgb(candidate)
        provider = 'codex_headless'
    else:
        if provider != 'gemini':
            raise ValueError(f'Unknown image provider: {provider}')
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("set GEMINI_API_KEY or supply --edited-image")
        if not model or "/" in model:
            raise ValueError("provide a Gemini image model ID")
        body = {"contents": [{"parts": [
            {"text": EDIT_PROMPT},
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(png_bytes(original)).decode()}},
            {"inlineData": {"mimeType": "image/png",
                            "data": base64.b64encode(png_bytes(allowed)).decode()}},
        ]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}}
        owned = client is None
        client = client or httpx.Client(timeout=180)
        try:
            response = client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": key}, json=body)
            response.raise_for_status()
            parts = [p for c in response.json().get("candidates", [])
                     for p in c.get("content", {}).get("parts", [])]
            images = [p["inlineData"] for p in parts if
                      p.get("inlineData", {}).get("mimeType", "").startswith("image/")]
            if len(images) != 1:
                raise ValueError("image editor did not return exactly one image")
            edited = Image.open(io.BytesIO(base64.b64decode(images[0]["data"], validate=True))).convert("RGB")
        finally:
            if owned:
                client.close()
        provider = "gemini"
        details = {}
    if edited.size != original.size:
        raise ValueError("edited image changed dimensions; cannot preserve scene geometry")
    composite = Image.composite(edited, original, allowed)
    composite.save(out, format="PNG")
    if capture_json:
        import json
        from .capture_guidance import load_guidance, overlay_on_edited
        guidance = load_guidance(capture_json, image)
        guidance_path = out.parent / 'capture_guidance.json'
        guidance_path.write_text(json.dumps(guidance, indent=2) + '\n')
        overlay_path = out.parent / 'edited_mano_overlay.png'
        overlay_on_edited(out, guidance, overlay_path)
        details['ground_truth_pose_overlay'] = str(overlay_path)
        details['capture_guidance'] = str(guidance_path)
        details['pose_source'] = 'capture_ground_truth'
    return {"provider": provider, "model": model if provider == "gemini" else None,
            **details,
            "outside_mask_pixels_identical": True, "editable_pixels": int((values == 255).sum()),
            "image": str(out)}


class H3Client:
    def __init__(self, url: str, expected_profile: str, *, client=None,
                 timeout: float = 1800, poll_interval: float = 2):
        self.url = url.rstrip("/")
        self.profile = expected_profile
        self.client = client or httpx.Client(timeout=60)
        self.timeout, self.poll_interval = timeout, poll_interval

    def close(self):
        self.client.close()

    def json(self, method: str, route: str, **kwargs) -> dict:
        response = self.client.request(method, self.url + route, **kwargs)
        response.raise_for_status()
        return response.json()

    def preflight(self) -> dict:
        health = self.json("GET", "/health")
        if not health.get("ready"):
            raise RuntimeError("H3 inference engine is not ready")
        if health.get("checkpoint_profile") != self.profile:
            raise RuntimeError(f"expected checkpoint profile {self.profile!r}, "
                               f"received {health.get('checkpoint_profile')!r}")
        if not health.get("robot_custom_frames_v1"):
            raise RuntimeError("H3 runtime lacks robot_custom_frames_v1; apply integrations/h3 custom-frame patch")
        if not health.get("exact_cfg_parallel") or health.get("sol_cache_threshold", 0):
            raise RuntimeError("expected exact Sol inference with approximate cache disabled")
        return health

    def register(self, bundle: Path) -> dict:
        return self.json("POST", "/v1/robot-frames", json={
            "bundle_b64": base64.b64encode(bundle.read_bytes()).decode()})

    def generate(self, clip_id: str, prompt: str, seed: int, out: Path) -> dict:
        # The selected mirrored checkpoint has a fixed 107-frame recipe.
        spec = {"clip_id": clip_id, "dataset": "robot_uploads", "prompt": prompt,
                "seed": seed, "steps": 50, "cfg": 5.0,
                "video_shift": 12.0, "hand_shift": 36.0,
                "match_checkpoint_recipe": True}
        job = self.json("POST", "/v1/videos", json=spec)
        # Persist the ID immediately: interrupted clients can recover the remote job.
        import json
        (out / "h3_submission.json").write_text(json.dumps({"request": spec, **job}, indent=2))
        job_id = job["id"]
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            result = self.json("GET", f"/v1/videos/{job_id}")
            if result["status"] == "failed":
                raise RuntimeError(f"H3 job {job_id} failed: {result.get('error')}")
            if result["status"] == "completed":
                for route, filename in (("content", "generated_full.mp4"), ("joints", "joints_full.npz")):
                    with self.client.stream("GET", f"{self.url}/v1/videos/{job_id}/{route}") as response:
                        response.raise_for_status()
                        part = out / (filename + ".part")
                        with part.open("wb") as stream:
                            for chunk in response.iter_bytes():
                                stream.write(chunk)
                        part.replace(out / filename)
                return {"request": spec, "result": result}
            time.sleep(self.poll_interval)
        raise TimeoutError(f"H3 job {job_id} did not finish within {self.timeout}s; see h3_submission.json")
