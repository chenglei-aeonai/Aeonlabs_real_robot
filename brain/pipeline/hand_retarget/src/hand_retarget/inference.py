"""Inference-time image → H3 video/actions → UR5e + DG5F robot demonstration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .h3_actions import (fixed_camera_transform, load_h3_actions, retarget_actions,
                         write_world_csv)

DEFAULT_PROFILE = "fast-align-hf15-mirrored-12983"


def trim_video(source: Path, destination: Path, frames: int, fps: float,
               source_frames: int | None = None) -> None:
    """Decode/count to verify timing, then encode the matching frame prefix."""
    import imageio_ffmpeg as ffmpeg
    reader = ffmpeg.read_frames(str(source))
    try:
        metadata = next(reader)
        if not np.isclose(metadata["fps"], fps, atol=1e-3):
            raise ValueError(f"video fps {metadata['fps']} differs from actions fps {fps}")
        count = sum(1 for _ in reader)
    finally:
        reader.close()
    if count < frames:
        raise ValueError(f"video has only {count} frames; need {frames}")
    if source_frames is not None and count != source_frames:
        raise ValueError(f"video has {count} frames but actions have {source_frames}")
    subprocess.run([ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(source),
                    "-map", "0:v:0", "-frames:v", str(frames), "-an", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)],
                   check=True, timeout=180)


def postprocess(joints: Path, video: Path | None, out: Path, seconds: float,
                camera_to_demo: np.ndarray, method: str,
                robot_from_demo: np.ndarray | None = None) -> dict:
    demo, metadata = load_h3_actions(joints, seconds, camera_to_demo,
                                     center_origin=robot_from_demo is None)
    if video is not None:
        trim_video(video, out / "video.mp4", demo.num_frames, demo.fps, metadata["source_frames"])
    with np.load(joints, allow_pickle=False) as data:
        np.savez(out / "human_actions.npz", joints=data["joints"][:demo.num_frames],
                 K=data["K"], fps=demo.fps)
    write_world_csv(demo, out / "human_world_joints.csv")
    robot = retarget_actions(demo, out, method, robot_from_demo)
    from .trajectory import export_trajectory
    robot["exports"] = export_trajectory(out, robot)
    return {"human": metadata, "robot": robot, "video": str(out / "video.mp4") if video else None}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="edit → uplift → H3 → retarget")
    run.add_argument("--image", type=Path, required=True)
    run.add_argument('--capture-json', type=Path)
    run.add_argument('--image-provider', choices=('codex', 'gemini'), default=os.environ.get('IMAGE_PROVIDER', 'codex'))
    run.add_argument("--mask", type=Path, help="white=end effectors; omitted: propose automatically")
    run.add_argument("--segmentation-model", default="gemini-2.5-flash")
    run.add_argument("--edited-image", type=Path, help="use an existing edited image instead of Gemini")
    run.add_argument("--image-model", default="gemini-2.5-flash-image")
    run.add_argument("--instruction", required=True)
    run.add_argument("--h3-url", required=True)
    run.add_argument("--profile", default=DEFAULT_PROFILE)
    run.add_argument("--uplift-python", type=Path, required=True)
    run.add_argument("--uplift-script", type=Path, required=True)
    run.add_argument("--hfov", type=float, required=True, help="camera horizontal field of view, degrees")
    run.add_argument("--seed", type=int, default=0)
    local = sub.add_parser("retarget", help="retarget an existing H3 joints.npz")
    local.add_argument("--joints", type=Path, required=True)
    local.add_argument("--video", type=Path)
    for command in (run, local):
        command.add_argument("--out", type=Path, required=True, help="new output directory")
        command.add_argument("--seconds", type=float, default=4.0)
        command.add_argument("--method", choices=("vector", "fingertip_ik", "joint_mapping"), default="vector")
        camera = command.add_mutually_exclusive_group(required=True)
        camera.add_argument("--fixed-camera-pitch", type=float, help="explicit stationary-camera assumption")
        camera.add_argument("--camera-transforms", type=Path, help="NPY (4,4) or (T,4,4), camera→demo")
        command.add_argument("--robot-from-demo", type=Path, help="calibrated rigid NPY (4,4); else upstream home placement")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not 4 <= args.seconds <= 5:
        parser.error("--seconds must be between 4 and 5")
    if args.command == "run" and args.seconds > 107 / 24:
        parser.error("the selected checkpoint supports at most 107/24 seconds; use 4 seconds")
    transform = (np.load(args.camera_transforms, allow_pickle=False) if args.camera_transforms
                 else fixed_camera_transform(args.fixed_camera_pitch))
    placement = np.load(args.robot_from_demo, allow_pickle=False) if args.robot_from_demo else None
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "command": args.command,
              "camera_assumption": "fixed" if args.fixed_camera_pitch is not None else "provided_transforms"}
    manifest = args.out / "manifest.json"
    def stage(name):
        temp = args.out / "progress.json.tmp"
        temp.write_text(json.dumps({"stage": name, "updated_at": time.time()}))
        temp.replace(args.out / "progress.json")
    try:
        if args.command == "run":
            if args.capture_json:
                from .capture_guidance import load_guidance
                import shutil
                load_guidance(args.capture_json, args.image)
                shutil.copyfile(args.capture_json, args.out / 'capture.json')
            from .inference_clients import H3Client, replace_hands
            if not args.instruction.strip() or not 20 <= args.hfov <= 160:
                raise ValueError("non-empty instruction and HFOV 20–160 are required")
            client = H3Client(args.h3_url, args.profile)
            try:
                stage("preflight")
                report["engine_health"] = client.preflight()
                report["instruction"] = args.instruction
                report["input_sha256"] = sha256(args.image)
                if args.mask is None:
                    from .segmentation import segment_hands
                    stage("mask")
                    args.mask = args.out / "edit_mask.png"
                    report["segmentation"] = segment_hands(
                        args.image, args.mask, model=args.segmentation_model, provider=args.image_provider)
                else:
                    import shutil
                    shutil.copyfile(args.mask, args.out / "edit_mask.png")
                report["edit_mask_sha256"] = sha256(args.mask)
                edited = args.out / "edited.png"
                stage("edit")
                report["edit"] = replace_hands(args.image, args.mask, edited,
                                               model=args.image_model, edited_image=args.edited_image,
                                               provider=args.image_provider, capture_json=args.capture_json)
                if report['edit'].get('requires_review'):
                    raise ValueError('The edited candidate requires pose review before downstream generation; inspect edited.png and edited_mano_overlay.png.')
                bundle = args.out / "first_frame.npz"
                stage("uplift")
                with (args.out / "uplift.log").open("w") as log:
                    subprocess.run([str(args.uplift_python.resolve()), str(args.uplift_script.resolve()),
                                    "--image", str(edited), "--out", str(bundle),
                                    "--hfov", str(args.hfov)], stdout=log, stderr=subprocess.STDOUT,
                                   check=True, timeout=900)
                report["first_frame"] = client.register(bundle)
                stage("generate")
                report["generation"] = client.generate(report["first_frame"]["clip_id"],
                                                        args.instruction, args.seed, args.out)
            finally:
                client.close()
            joints, video = args.out / "joints_full.npz", args.out / "generated_full.mp4"
        else:
            joints, video = args.joints, args.video
        report["source_joints_sha256"] = sha256(joints)
        stage("retarget")
        report.update(postprocess(joints, video, args.out, args.seconds, transform, args.method, placement))
        report["status"] = "completed"
        stage("completed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        stage("failed")
        raise
    finally:
        manifest.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"manifest": str(manifest), "robot_actions": report["robot"]["robot_actions"]}))


if __name__ == "__main__":
    main()
