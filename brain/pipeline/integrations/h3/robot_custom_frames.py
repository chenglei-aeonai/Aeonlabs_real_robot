"""Copy beside h3_joint/serve_hands.py with install.py; no changes to model weights.

Single-frame WiLoR bundles: uint8 RGB, dpl/MANO joints in optical-camera metres,
wrist_R in the codec's MANO convention, valid[left,right], calibrated/crop-adjusted K.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

PROFILE = "fast-align-hf15-mirrored-12983"
CHECKPOINT_SHA256 = "5553ab5bbe4c1429e9d911b3e19a689e2eecbaa4e981b895849b2617693a161e"


class FrameRequest(BaseModel):
    bundle_b64: str = Field(max_length=36_000_000)


def load_bundle(payload: bytes) -> dict:
    # Check uncompressed size before NumPy materializes any arrays.
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        if sum(f.file_size for f in archive.infolist()) > 32_000_000:
            raise ValueError("first-frame bundle exceeds 32 MB unpacked")
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in ("frame", "joints3d", "wrist_R", "valid", "K")}
    frame, joints, rotation, valid, K = (bundle[k] for k in ("frame", "joints3d", "wrist_R", "valid", "K"))
    if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3
            or min(frame.shape[:2]) < 32 or max(frame.shape[:2]) > 2048
            or any(n % 32 for n in frame.shape[:2])):
        raise ValueError("frame must be uint8 RGB, dimensions 32–2048 and divisible by 32")
    if joints.shape != (2, 21, 3) or rotation.shape != (2, 3, 3):
        raise ValueError("expected dpl joints3d (2,21,3) and MANO wrist_R (2,3,3)")
    if valid.shape != (2,) or valid.dtype != np.bool_ or not valid[1]:
        raise ValueError("bundle must contain a valid right hand")
    if K.shape != (3, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("invalid camera intrinsics")
    if not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid homogeneous camera intrinsics")
    for h in range(2):
        if valid[h]:
            if not np.isfinite(joints[h]).all() or np.any(joints[h, :, 2] <= 0):
                raise ValueError("valid hand must have finite positive optical depth")
            R = rotation[h]
            if (not np.isfinite(R).all() or not np.allclose(R @ R.T, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(R), 1, atol=1e-5)):
                raise ValueError("wrist_R must be a proper rotation")
        else:
            # Absent left hand has an explicit masked neutral anchor.
            joints[h] = 0
            rotation[h] = np.eye(3)
    return bundle


def anchor_and_mask(bundle: dict):
    """Physical 21-keypoint anchor → v2_abspose; codec owns chirality conversion."""
    import torch
    from h3_joint.skeleton_math import rotmat_to_6d
    anchor = torch.zeros(1, 138, dtype=torch.float32)
    mask = torch.zeros(1, 138, dtype=torch.bool)
    for h, (ts, rs, fs) in enumerate(((slice(0, 3), slice(6, 12), slice(18, 78)),
                                     (slice(3, 6), slice(12, 18), slice(78, 138)))):
        R = torch.as_tensor(bundle["wrist_R"][h], dtype=torch.float32)
        J = torch.as_tensor(bundle["joints3d"][h], dtype=torch.float32)
        anchor[0, ts] = J[0]
        anchor[0, rs] = rotmat_to_6d(R)
        anchor[0, fs] = ((J[1:] - J[:1]) @ R).reshape(-1)
        if bundle["valid"][h]:
            mask[0, ts] = mask[0, rs] = mask[0, fs] = True
    return anchor, mask


def conditioning_items(engine, candidate_id: str, prompt: str):
    """Use the uploaded RGB for BOTH CFG branches and the supplied human anchor.

    No dataset clip supplies RGB, intrinsics, prompt embeddings, hand positions,
    or a gripper embodiment. Reuse only the checkpoint's stats and hand token tag.
    """
    import torch
    from h3_joint.serve_hands import native_sequence_items

    with engine._robot_frame_lock:
        bundle = engine._robot_frames[candidate_id]
    anchor, valid = anchor_and_mask(bundle)
    base = {"skel_raw": anchor, "chan_valid": valid, "stats": engine.dataset.stats,
            "clip_id": candidate_id, "motion_embodiment": 0,
            "shared": {"cam_K": torch.as_tensor(bundle["K"], dtype=torch.float32),
                       "motion_embodiment": 0}}
    encoded = engine.encode_continuation(bundle["frame"], prompt, 107)
    return native_sequence_items(engine.dataset, base, encoded, 107)


def install_routes(app, engine):
    """Extend the existing queue/API; never start a second GPU worker."""
    import os
    from fastapi import HTTPException
    from PIL import Image

    engine._robot_frames = {}
    engine._robot_frame_lock = threading.Lock()
    # create_app runs before the background loader stages weights into /dev/shm.
    canonical_checkpoint = Path(engine.ckpt)
    original_has_clip = engine.has_clip
    original_datasets = engine.datasets
    original_base = engine.base_clip_id
    original_frame = engine.first_frame
    original_caption = engine.clip_caption

    def has_clip(cid, dataset=None):
        with engine._robot_frame_lock:
            custom = cid in engine._robot_frames
        return (dataset in (None, "robot_uploads")) if custom else original_has_clip(cid, dataset)

    def datasets():
        with engine._robot_frame_lock:
            count = len(engine._robot_frames)
        return [*original_datasets(), {"id": "robot_uploads", "count": count}]

    def first_frame(cid):
        with engine._robot_frame_lock:
            bundle = engine._robot_frames.get(cid)
        return bundle["frame"] if bundle is not None else original_frame(cid)

    engine.has_clip = has_clip
    engine.datasets = datasets
    engine.first_frame = first_frame
    engine.base_clip_id = lambda cid: cid if cid.startswith("robot::") else original_base(cid)
    engine.clip_caption = lambda cid: "" if cid.startswith("robot::") else original_caption(cid)

    @app.post("/v1/robot-frames")
    def register(req: FrameRequest):
        if not engine.ready:
            raise HTTPException(503, "model not ready")
        if os.environ.get("H3_SERVE_PROFILE") != PROFILE:
            raise HTTPException(409, f"custom robot frames are validated only for {PROFILE}")
        manifest = canonical_checkpoint.parent / "demo-checkpoint-manifest.json"
        if (not manifest.is_file() or
                json.loads(manifest.read_text()).get("source_checkpoint_sha256") != CHECKPOINT_SHA256):
            raise HTTPException(409, "checkpoint manifest is not the deployed mirrored step12983 export")
        try:
            payload = base64.b64decode(req.bundle_b64, validate=True)
            bundle = load_bundle(payload)
        except (ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:
            raise HTTPException(422, str(exc)) from exc
        digest = hashlib.sha256(payload).hexdigest()
        cid = f"robot::{digest}"
        with engine._robot_frame_lock:
            if cid not in engine._robot_frames and len(engine._robot_frames) >= 64:
                raise HTTPException(429, "custom frame registry full; restart the experimental service")
            engine._robot_frames[cid] = bundle
        buf = io.BytesIO()
        Image.fromarray(bundle["frame"]).save(buf, format="PNG")
        return {"clip_id": cid, "dataset": "robot_uploads", "bundle_sha256": digest,
                "frame_png_b64": base64.b64encode(buf.getvalue()).decode(),
                "checkpoint_sha256": CHECKPOINT_SHA256, "motion_embodiment": 0}
