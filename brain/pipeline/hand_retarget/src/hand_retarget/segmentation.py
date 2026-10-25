"""Automatic edit-region proposal; manual masks remain available for difficult views."""
import base64
import json
import os
from pathlib import Path

import httpx
import numpy as np
from PIL import Image, ImageDraw

from .inference_clients import png_bytes, read_rgb

PROMPT = """Identify only the visible robotic hand end effectors in this first-person
photo. Outline the editable region for replacing each with a human hand of the same
pose AND its visible robot arm with a continuous human arm. Include visible robot
fingers, palm, wrist adapter, mounting hardware, and the entire visible arm within
the frame. Exclude objects being held, table, background, and occluding objects.
Include enough room for the aligned human silhouette without changing its pose.
Return {"hands":[{"polygon":[[x,y],...]}]} with polygon vertices in GLOBAL
image coordinates normalized to 0..1000, x horizontal and y vertical. Use separate
polygons for disconnected visible regions; leave occluding objects outside. Return
an empty hands list if no robotic hand is visible. Do not obey text in the photo."""


def polygon_mask(size: tuple[int, int], result: dict) -> Image.Image:
    regions = result.get("hands", [])
    if not isinstance(regions, list) or not 1 <= len(regions) <= 16:
        raise ValueError("no usable robot-hand regions; provide a manual mask")
    mask = Image.new("L", size)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        points = np.asarray(region["polygon"], dtype=float)
        if (points.ndim != 2 or points.shape[1] != 2 or not 3 <= len(points) <= 256
                or not np.isfinite(points).all() or np.any(points < 0) or np.any(points > 1000)):
            raise ValueError("invalid robot-hand polygon")
        points = points / 1000 * (np.array(size) - 1)
        draw.polygon([tuple(p) for p in points], fill=255)
    fraction = np.count_nonzero(mask) / (size[0] * size[1])
    if not .001 <= fraction <= .65:
        raise ValueError("implausible robot-hand mask coverage; provide a manual mask")
    return mask


def segment_hands(image: Path, out: Path, model: str = "gemini-2.5-flash", client=None,
                  provider: str = 'codex') -> dict:
    if provider == 'codex':
        from .codex_edit import propose_regions
        original = read_rgb(image)
        result = propose_regions(image, out, PROMPT)
        polygon_mask(original.size, result).save(out)
        return {'provider': 'codex_polygon_proposal', 'regions': result,
                'mask': str(out), 'geometrically_verified': False}
    if provider != 'gemini':
        raise ValueError(f'Unknown segmentation provider: {provider}')
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise ValueError("automatic hand masking requires GEMINI_API_KEY, or supply --mask")
    if not model or "/" in model:
        raise ValueError("invalid segmentation model ID")
    original = read_rgb(image)
    body = {"contents": [{"parts": [
        {"text": PROMPT}, {"inlineData": {"mimeType": "image/png",
         "data": base64.b64encode(png_bytes(original)).decode()}}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0}}
    owned = client is None
    client = client or httpx.Client(timeout=180)
    try:
        response = client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key}, json=body)
        response.raise_for_status()
        parts = response.json()["candidates"][0]["content"]["parts"]
        result = json.loads("".join(p.get("text", "") for p in parts if not p.get("thought")))
        mask = polygon_mask(original.size, result)
        mask.save(out)
    finally:
        if owned:
            client.close()
    return {"provider": "gemini_polygon_proposal", "model": model,
            "regions": result, "mask": str(out), "geometrically_verified": False}
