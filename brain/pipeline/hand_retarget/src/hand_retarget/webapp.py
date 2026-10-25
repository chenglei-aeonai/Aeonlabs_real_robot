"""Loopback-only inference UI and persistent job API with headless Codex image editing."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from typing import Literal

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageOps

from .calibration_status import status as calibration_status
from .inference import DEFAULT_PROFILE
from .h3_actions import rigid_transforms
from .inference_clients import H3Client

STATIC = Path(__file__).with_name("web")
ARTIFACTS = {"edited.png", "edit_mask.png", "video.mp4", "robot_actions.npz",
             "robot_actions.csv", "trajectory.json", "controller_commands.json",
             "human_actions.npz", "manifest.json", "bundle.zip", 'capture.json',
             'capture_guidance.json', 'mano_overlay.png', 'robot_overlay.png', 'edited_mano_overlay.png'}
ARTIFACTS |= {'first_frame_gt.npz','first_frame_gt.json','future_hands.npz',
              'predicted_hand_overlay.mp4','diffusion_submission.json'}
ARTIFACTS |= {'retargeted_hand.npz', 'robot_demonstration.npz', 'retarget_preview.json', 'scene_geometry.json',
              'mujoco_camera.mp4','mujoco_overview.mp4','mujoco_motion.npz','mujoco_scene.xml','mujoco_scene.zip','cube_config.json'}
MAX_BODY = 48 * 1024 * 1024

ARTIFACTS |= {'robot_rollout.npz'}


@dataclass(frozen=True)
class Config:
    runs: Path
    h3_url: str = "http://127.0.0.1:30121"
    uplift_python: str = ""
    uplift_script: str = ""
    image_model: str = "gemini-2.5-flash-image"
    segmentation_model: str = "gemini-2.5-flash"
    profile: str = DEFAULT_PROFILE
    image_provider: str = 'codex'
    capture_runner: Path = Path(__file__).resolve().parents[4] / 'capture/run_capture.sh'


class JobInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    image_b64: str | None = Field(default=None, min_length=1, max_length=24 * 1024 * 1024)
    capture_id: str | None = Field(default=None, pattern=r'^[0-9a-f]{32}$')
    stop_after: Literal['edit', 'retarget'] = 'retarget'
    instruction: str = Field(min_length=1, max_length=4000)
    mask_b64: str | None = Field(default=None, max_length=24 * 1024 * 1024)
    edited_image_b64: str | None = Field(default=None, max_length=24 * 1024 * 1024)
    hfov: float | None = Field(default=None, gt=0, lt=180)
    fixed_camera_pitch: float | None = Field(default=None, gt=-89, lt=89)
    camera_to_demo: list | None = None
    robot_from_demo: list | None = None
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class CropDefaults(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    version: Literal[1] = 1
    width: int = Field(ge=32, le=4096)
    height: int = Field(ge=32, le=4096)
    zoom: float = Field(ge=1, le=4)
    x: float = Field(ge=0, le=100)
    y: float = Field(ge=0, le=100)


class CropInput(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    x: float = Field(ge=0)
    y: float = Field(ge=0)
    width: float = Field(ge=2)
    height: float = Field(ge=2)
    output_width: int = Field(default=832, ge=32, le=4096)
    output_height: int = Field(default=640, ge=32, le=4096)
    defaults: CropDefaults | None = None


class DiffusionConfigInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    url: str = Field(max_length=500)


class CheckpointInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    profile: str = Field(min_length=1,max_length=120)
    partition: Literal['normal','debug'] = 'normal'
    sol_enabled: bool = True


class SimulationInput(BaseModel):
    model_config = ConfigDict(extra='forbid',allow_inf_nan=False)
    source_job_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    pixel: tuple[float,float]
    edge_m: float = Field(default=.04,ge=.005,le=.2)
    mass_kg: float = Field(default=.05,ge=.001,le=2.)
    yaw_deg: float = Field(default=0.,ge=-180,le=180)
    friction: float = Field(default=.8,ge=.05,le=3.)


class RetargetInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_job_id: str = Field(pattern=r'^[0-9a-f]{32}$')


class RolloutInput(RetargetInput):
    speed_scale: float = Field(default=1.,ge=.25,le=1.,allow_inf_nan=False)


class HardwareRunInput(RetargetInput):
    confirmation: str
    workspace_clear: bool = False
    arm_only: bool = False


class HardwareHeartbeatInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str = Field(pattern=r'^[0-9a-f]{32}$')


class FutureInput(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_job_id: str = Field(pattern=r'^[0-9a-f]{32}$')
    instruction: str = Field(min_length=1,max_length=4000)
    profile: str = Field(min_length=1,max_length=120)
    seed: int = Field(default=0,ge=0,le=2**31-1)
    assume_zero_contact: bool = False


def write_json(path: Path, value: dict):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def package_result(result: Path):
    with zipfile.ZipFile(result / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in sorted(ARTIFACTS - {"bundle.zip"}):
            path = result / name
            if path.is_file():
                bundle.write(path, name)


def import_result(config: Config, source: Path, label: str):
    """Expose an existing completed run, explicitly marked as an import."""
    report = read_json(source / "manifest.json")
    if report.get("status") != "completed" or not (source / "robot_actions.npz").is_file():
        raise ValueError("Only completed pipeline results can be imported")
    job = config.runs / uuid.uuid4().hex
    (job / "result").mkdir(parents=True)
    for name in ARTIFACTS - {"bundle.zip"}:
        if (source / name).is_file():
            shutil.copyfile(source / name, job / "result" / name)
    from .trajectory import export_trajectory
    report["robot"]["exports"] = export_trajectory(job / "result", report["robot"])
    write_json(job / "result/manifest.json", report)
    package_result(job / "result")
    write_json(job / "status.json", {
        "id": job.name, "status": "completed", "created_at": time.time(),
        "instruction": label, "mode": "imported_result", "hardware_executed": False,
        "source_result": str(source.resolve()),
    })
    return job.name


def save_image(encoded: str, path: Path, *, mask=False) -> tuple[int, int]:
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 16_000_000 or min(source.size) < 32:
                raise ValueError("image must have at least 32 pixels per side and at most 16 megapixels")
            image = ImageOps.exif_transpose(source).convert("L" if mask else "RGB")
            if mask and not np.isin(np.asarray(image), [0, 255]).all():
                raise ValueError("mask must be binary: 0 or 255")
            if mask and len(np.unique(image)) != 2:
                raise ValueError("mask must contain both editable and protected regions")
            image.save(path, format="PNG")
            return image.size
    except Exception as exc:
        raise ValueError(f"invalid {'mask' if mask else 'image'}: {exc}") from exc


def build_command(config: Config, job: Path, request: JobInput) -> list[str]:
    if request.stop_after == 'edit':
        args = [sys.executable, '-m', 'hand_retarget.image_edit', '--image', str(job / 'input.png'),
                '--capture-json', str(job / 'capture.json'), '--out', str(job / 'result')]
        if (job / 'mask.png').exists():
            args.extend(['--mask', str(job / 'mask.png')])
        return args
    args = [sys.executable, "-m", "hand_retarget.inference", "run",
            "--image", str(job / "input.png"), "--instruction", request.instruction.strip(),
            "--h3-url", config.h3_url, "--profile", config.profile,
            "--uplift-python", config.uplift_python, "--uplift-script", config.uplift_script,
            "--image-model", config.image_model, "--segmentation-model", config.segmentation_model,
            '--image-provider', config.image_provider,
            "--hfov", str(request.hfov), "--seed", str(request.seed),
            "--seconds", "4", "--out", str(job / "result")]
    for filename, flag in (("mask.png", "--mask"), ("edited_input.png", "--edited-image"),
                           ('capture.json', '--capture-json'),
                           ("camera.npy", "--camera-transforms"),
                           ("placement.npy", "--robot-from-demo")):
        if (job / filename).exists():
            args.extend([flag, str(job / filename)])
    if request.fixed_camera_pitch is not None:
        args.extend(["--fixed-camera-pitch", str(request.fixed_camera_pitch)])
    return args


class Jobs:
    def __init__(self, config: Config):
        self.config = config
        config.runs.mkdir(parents=True, exist_ok=True)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="retarget")
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.process = None
        self.pending = 0
        self.scene = None
        self.scene_lock = threading.Lock()
        from .capture_bridge import Captures
        from .hardware_session import HardwareSession
        self.hardware = HardwareSession()
        self.captures = Captures(config.runs.parent / 'captures', config.capture_runner)
        from .diffusion_service import DiffusionService
        self.diffusion = DiffusionService(config.runs.parent)
        for status in config.runs.glob("*/status.json"):
            value = read_json(status)
            if value.get("status") in ("queued", "running"):
                value.update(status="interrupted", error="Server restarted; job was not resubmitted.")
                write_json(status, value)

    def locate(self, job_id):
        if not re.fullmatch(r"[0-9a-f]{32}", job_id):
            raise HTTPException(404, "Unknown job")
        path = self.config.runs / job_id
        if not (path / "status.json").is_file():
            raise HTTPException(404, "Unknown job")
        return path

    def snapshot(self, job: Path):
        value = read_json(job / "status.json")
        value["progress"] = read_json(job / "result/progress.json")
        capture_path = job / 'result/capture.json'
        if not capture_path.is_file() and value.get('source_job_id'):
            source = self.locate(value['source_job_id'])
            capture_path = source / 'result/capture.json'
        if capture_path.is_file():
            value['calibration_status'] = calibration_status(read_json(capture_path))
        # Only publish final downloadable commands after the complete pipeline succeeds.
        names = {"edited.png", "edit_mask.png", 'edited_mano_overlay.png'}
        if value["status"] == "completed":
            names = ARTIFACTS
            report = read_json(job / "result/manifest.json")
            robot = report.get("robot", {})
            value["metrics"] = {key: robot.get(key) for key in (
                "mean_fingertip_error_m", "placement_mode", "exports", "duration_seconds")}
            if value.get('mode')=='diffusion_video':value['metrics']=report.get('output',{})
            if value.get('mode')=='mujoco_replay':
                value['simulation']={key:report.get(key) for key in ('lift_observed','max_cube_lift_m','hand_contact_frames','joint_tracking_rms_rad','lift_contact_duration_s')}
        value["artifacts"] = {name: f"/api/jobs/{job.name}/files/{name}"
                              for name in sorted(names) if (job / "result" / name).is_file()}
        if value.get('mode')=='robot_rollout' and value.get('status')=='completed':
            report=read_json(job/'result/manifest.json')
            runner=Path(__file__).resolve().parents[3]/'run_robot_rollout.sh'
            command=shlex.join([str(runner),'replay','--folder',str((job/'result').resolve())])
            value['rollout']={key:report.get(key) for key in ('duration_s','speed_scale','source_duration_s','effective_speed_scale','native_timing','timing_policy','start_policy','live_relative_duration_s','interpolation','finger_smoothing_sigma_frames','maximum_finger_smoothing_change_rad','initial_transition_duration_s','arm_speed_limit_rad_s','hand_speed_limit_rad_s','minimum_table_clearance_m','maximum_ik_position_error_m','execution_ready','limitations')}
            value['rollout']['check_command']=command
            value['rollout']['send_command']=command+' --send'
        value["input_url"] = f"/api/jobs/{job.name}/input" if (job / "input.png").exists() else None
        return value

    def execute(self, job: Path, command: list[str]):
        state = read_json(job / "status.json")
        try:
            state.update(status="running", started_at=time.time())
            write_json(job / "status.json", state)
            with (job / "pipeline.log").open("w") as log:
                with self.lock:
                    if self.stopping.is_set():
                        raise RuntimeError("Server stopped; job was not resubmitted.")
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    self.process = process
                try:
                    code = process.wait(timeout=3600)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise TimeoutError("Pipeline exceeded one hour; remote H3 job may still exist.")
            if self.stopping.is_set():
                raise RuntimeError("Server stopped; remote H3 job may still exist.")
            report = read_json(job / "result/manifest.json")
            if code != 0 or report.get("status") != "completed":
                raise RuntimeError(report.get("error", f"Pipeline exited with status {code}"))
            package_result(job / "result")
            state.update(status="completed", finished_at=time.time())
            if report.get('edit', {}).get('requires_review'):
                state['review_note'] = ('Image generated; review the GT overlay. Alignment was not independently verified. ' + (report['edit'].get('review_reason') or ''))
                if report['edit'].get('diffusion_eligible') is False:
                    state['review_note'] += ' Diffusion is blocked; regenerate Stage 2 to obtain a matching candidate.'
        except Exception as exc:
            state.update(status="interrupted" if self.stopping.is_set() else "failed",
                         error=str(exc), finished_at=time.time())
        finally:
            write_json(job / "status.json", state)
            with self.lock:
                self.process = None
                self.pending -= 1

    def open_retarget_scene(self,identifier):
        job=self.locate(identifier)
        state=read_json(job/'status.json')
        if state.get('mode')!='generated_hand_retarget' or state.get('status')!='completed':
            raise ValueError('Choose a completed retargeting result.')
        with self.scene_lock:
            if self.scene is None or self.scene.result!=job/'result':
                if self.scene is not None:self.scene.close();self.scene=None
                from .retarget_scene import RetargetScene
                self.scene=RetargetScene(job/'result')
            return {'url':f'http://127.0.0.1:{self.scene.server.get_port()}',
                    'job_id':identifier,'geometry':self.scene.geometry,'hardware_executed':False}

    def close(self):
        """Stop local process groups and drain queued jobs without launching them."""
        self.hardware.close()
        with self.lock:
            self.stopping.set()
            if self.process is not None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.executor.shutdown(wait=True)
        with self.scene_lock:
            if self.scene is not None:self.scene.close();self.scene=None

    def source_edit(self, identifier):
        job=self.locate(identifier)
        if read_json(job/'status.json').get('status')!='completed':
            raise ValueError('Choose a completed human-hand edit first.')
        result=job/'result'
        if not (result/'edited.png').is_file() or not (result/'capture.json').is_file():
            raise ValueError('Source result must contain the converted image and its paired captured GT.')
        edit = read_json(result/'manifest.json').get('edit', {})
        if edit.get('diffusion_eligible') is False or edit.get('orientation_self_check') in ('uncertain', 'mismatched'):
            raise ValueError('This image is a review candidate: hand orientation/alignment was not confirmed. Regenerate Stage 2 before diffusion. ' + (edit.get('review_reason') or ''))
        return result

    def prepare_conditioning(self, identifier):
        from .diffusion_bundle import build_bundle
        result=self.source_edit(identifier)
        report=build_bundle(result/'edited.png',result/'capture.json',result/'first_frame_gt.npz')
        package_result(result)
        return {'report':report,'bundle':f'/api/jobs/{identifier}/files/first_frame_gt.npz',
                'metadata':f'/api/jobs/{identifier}/files/first_frame_gt.json'}

    def submit_rollout(self,request):
        source_job=self.locate(request.source_job_id)
        state=read_json(source_job/'status.json')
        if state.get('mode')!='generated_hand_retarget' or state.get('status')!='completed':
            raise ValueError('Choose a completed retargeted trajectory.')
        with self.lock:
            if self.hardware.active:raise HTTPException(409,'Stop physical replay before starting another task.')
            if self.stopping.is_set():raise HTTPException(503,'Server is stopping')
            if self.pending>=8:raise HTTPException(429,'Queue full')
            self.pending+=1
        job=self.config.runs/uuid.uuid4().hex
        try:
            inputs=job/'inputs';inputs.mkdir(parents=True)
            for name in ('capture.json','retargeted_hand.npz'):
                shutil.copyfile(source_job/'result'/name,inputs/name)
            state={'id':job.name,'status':'queued','mode':'robot_rollout','created_at':time.time(),
                   'source_job_id':request.source_job_id,'instruction':'Prepare supervised physical robot rollout','hardware_executed':False}
            write_json(job/'status.json',state);write_json(job/'request.json',request.model_dump())
            command=[sys.executable,'-m','hand_retarget.robot_rollout','prepare','--source',str(inputs),'--out',str(job/'result'),'--speed-scale',str(request.speed_scale)]
            self.executor.submit(self.execute,job,command)
            return state
        except Exception:
            with self.lock:self.pending-=1
            shutil.rmtree(job,ignore_errors=True);raise

    def submit_simulation(self,request):
        source_job=self.locate(request.source_job_id);state=read_json(source_job/'status.json')
        if state.get('mode')!='generated_hand_retarget' or state.get('status')!='completed':
            raise ValueError('Select a completed calibrated hand retargeting result.')
        source=source_job/'result'
        from .mujoco_replay import cube_position
        from .retarget_scene import scene_geometry
        cube_position(read_json(source/'capture.json'),scene_geometry(source),request.pixel,request.edge_m)
        with self.lock:
            if self.hardware.active:raise HTTPException(409,'Stop physical replay before starting another task.')
            if self.stopping.is_set():raise HTTPException(503,'Server is stopping')
            if self.pending>=8:raise HTTPException(429,'Queue full')
            self.pending+=1
        job=self.config.runs/uuid.uuid4().hex
        try:
            inputs=job/'inputs';inputs.mkdir(parents=True)
            for name in ('capture.json','retargeted_hand.npz'):shutil.copyfile(source/name,inputs/name)
            config=request.model_dump(exclude={'source_job_id'});write_json(job/'cube_config.json',config)
            state={'id':job.name,'status':'queued','created_at':time.time(),'mode':'mujoco_replay',
                   'source_job_id':request.source_job_id,'instruction':'MuJoCo calibrated cube contact replay','hardware_executed':False}
            command=[sys.executable,'-m','hand_retarget.mujoco_replay','--source',str(inputs),'--out',str(job/'result'),'--config',str(job/'cube_config.json')]
            write_json(job/'status.json',state);write_json(job/'request.json',request.model_dump())
            self.executor.submit(self.execute,job,command);return state
        except Exception:
            with self.lock:self.pending-=1
            shutil.rmtree(job,ignore_errors=True);raise

    def submit_generated_retarget(self, request):
        source_job=self.locate(request.source_job_id)
        source_status=read_json(source_job/'status.json')
        if source_status.get('mode')!='diffusion_video' or source_status.get('status')!='completed':
            raise ValueError('Select a completed future-video generation first.')
        source=source_job/'result'
        capture=self.source_edit(source_status['source_job_id'])/'capture.json'
        if not (source/'future_hands.npz').is_file() or not capture.is_file():
            raise ValueError('Generated hand trajectory or its original calibrated capture is missing.')
        with self.lock:
            if self.hardware.active:raise HTTPException(409,'Stop physical replay before starting another task.')
            if self.stopping.is_set():raise HTTPException(503,'Server is stopping')
            if self.pending>=8:raise HTTPException(429,'Queue full')
            self.pending+=1
        job=self.config.runs/uuid.uuid4().hex
        try:
            inputs=job/'inputs';inputs.mkdir(parents=True)
            shutil.copyfile(capture,inputs/'capture.json')
            shutil.copyfile(source/'future_hands.npz',inputs/'future_hands.npz')
            value={'id':job.name,'status':'queued','created_at':time.time(),
                   'instruction':'Retarget generated right hand into calibrated robot base',
                   'mode':'generated_hand_retarget','source_job_id':request.source_job_id,'hardware_executed':False}
            command=[sys.executable,'-m','hand_retarget.generated_retarget','--input-dir',str(inputs),'--out',str(job/'result')]
            write_json(job/'request.json',request.model_dump());write_json(job/'status.json',value)
            self.executor.submit(self.execute,job,command)
            return value
        except Exception:
            with self.lock:self.pending-=1
            shutil.rmtree(job,ignore_errors=True);raise

    def submit_future(self, request):
        if not request.instruction.strip():raise ValueError('Describe what the hand should do next.')
        source=self.source_edit(request.source_job_id)
        connection=self.diffusion.config()
        if not connection['url']:raise ValueError('Configure the aeon-webui inference service first.')
        state=self.diffusion.status()
        if not state.get('connected') or state.get('state')!='ready' or state.get('selected_profile')!=request.profile:
            raise ValueError('Load the selected checkpoint and wait until it is ready.')
        capability=state.get('capture_gt_bridge')
        if not capability or request.profile not in capability.get('profiles',[]):
            raise ValueError('The service needs the capture-GT bridge and a compatible checkpoint.')
        if request.profile in capability.get('zero_contact_profiles',[]) and not request.assume_zero_contact:
            raise ValueError('This checkpoint needs contact data; select another or explicitly assume a no-contact start.')
        with self.lock:
            if self.hardware.active:raise HTTPException(409,'Stop physical replay before starting another task.')
            if self.stopping.is_set():raise HTTPException(503,'Server is stopping')
            if self.pending>=8:raise HTTPException(429,'Queue full')
            self.pending+=1
        job=self.config.runs/uuid.uuid4().hex
        try:
            job.mkdir()
            shutil.copyfile(source/'edited.png',job/'input.png')
            value={'id':job.name,'status':'queued','created_at':time.time(),
                   'instruction':request.instruction.strip(),'mode':'diffusion_video',
                   'checkpoint':request.profile,'source_job_id':request.source_job_id,'hardware_executed':False}
            command=[sys.executable,'-m','hand_retarget.diffusion','--source-result',str(source),
                     '--out',str(job/'result'),'--service-url',connection['url'],'--token',connection['token'],
                     '--profile',request.profile,'--instruction',request.instruction.strip(),'--seed',str(request.seed)]
            if request.assume_zero_contact:command.append('--assume-zero-contact')
            write_json(job/'request.json',request.model_dump());write_json(job/'status.json',value)
            self.executor.submit(self.execute,job,command)
            return value
        except Exception:
            with self.lock:self.pending-=1
            shutil.rmtree(job,ignore_errors=True);raise

    def submit(self, request: JobInput) -> dict:
        if not request.instruction.strip():
            raise ValueError("instruction cannot be blank")
        if self.hardware.active:raise HTTPException(409,'Stop physical replay before starting another task.')
        capture_folder = self.captures.locate(request.capture_id) if request.capture_id else None
        if capture_folder:
            capture = read_json(capture_folder / 'hand_joints.json')
            camera = capture['camera']
            k = np.asarray(camera['undistorted_intrinsics']['matrix'], dtype=float)
            # Compatibility scalar for downstream tools; the editor receives full K.
            hfov = float(np.degrees(np.arctan(k[0, 2]/k[0, 0]) +
                                   np.arctan((camera['width']-k[0, 2])/k[0, 0])))
            updates = {'hfov': hfov, 'fixed_camera_pitch': None}
            if request.stop_after != 'edit' and request.camera_to_demo is None:
                from .robot import ROBOT_BASE_POSITION
                camera_from_base = np.asarray(capture['camera_from_robot_base'], dtype=float)
                rigid_transforms(camera_from_base, 1)
                updates['camera_to_demo'] = np.linalg.inv(camera_from_base).tolist()
                if request.robot_from_demo is None:
                    placement = np.eye(4); placement[:3, 3] = ROBOT_BASE_POSITION
                    updates['robot_from_demo'] = placement.tolist()
            request = request.model_copy(update=updates)
        elif request.hfov is None:
            raise ValueError('Use a calibrated robot capture; uploaded images need explicit camera calibration through the API.')
        if request.stop_after != 'edit' and (request.fixed_camera_pitch is None) == (request.camera_to_demo is None):
            raise ValueError("choose fixed-camera pitch or camera-to-demo transforms")
        if request.camera_to_demo is not None:
            a = np.asarray(request.camera_to_demo, dtype=float)
            rigid_transforms(a, 1 if a.ndim == 2 else len(a))
        if request.robot_from_demo is not None:
            rigid_transforms(np.asarray(request.robot_from_demo), 1)
        if self.config.image_provider == 'codex' and not request.edited_image_b64:
            from .codex_edit import readiness
            ready, detail = readiness()
            if not ready:
                raise ValueError(detail)
        if self.config.image_provider == 'gemini' and not request.edited_image_b64 and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Image editing is not configured: set GEMINI_API_KEY on the server")
        if self.config.image_provider == 'gemini' and not request.mask_b64 and not os.environ.get("GEMINI_API_KEY"):
            raise ValueError("Automatic masking is not configured; upload a mask")
        if request.stop_after == 'edit' and (not request.capture_id or self.config.image_provider != 'codex'):
            raise ValueError('Capture-to-human editing requires a robot capture and the Codex provider.')
        if request.stop_after != 'edit' and not all(Path(p).is_file() for p in (self.config.uplift_python, self.config.uplift_script)):
            raise ValueError("WiLoR interpreter and script must be configured on the server")
        if request.capture_id and request.image_b64 is not None:
            raise ValueError('Use either capture_id or an uploaded image, not both.')
        if capture_folder is None and request.image_b64 is None:
            raise ValueError('Upload an image or select a robot capture.')
        with self.lock:
            if self.hardware.active:raise HTTPException(409,"Stop physical replay before starting another task.")
            if self.stopping.is_set():
                raise HTTPException(503, "Server is stopping")
            if self.pending >= 8:
                raise HTTPException(429, "Queue full; wait for an existing job")
            self.pending += 1
        job = self.config.runs / uuid.uuid4().hex
        try:
            job.mkdir()
            if capture_folder:
                shutil.copyfile(capture_folder / 'undistorted.png', job / 'input.png')
                shutil.copyfile(capture_folder / 'hand_joints.json', job / 'capture.json')
                shutil.copyfile(capture_folder / 'mano_overlay.png', job / 'mano_overlay.png')
                shutil.copyfile(capture_folder / 'overlay.png', job / 'robot_overlay.png')
                with Image.open(job / 'input.png') as source:
                    size = source.size
            else:
                size = save_image(request.image_b64, job / "input.png")
            for value, filename, is_mask in ((request.mask_b64, "mask.png", True),
                                           (request.edited_image_b64, "edited_input.png", False)):
                if value and save_image(value, job / filename, mask=is_mask) != size:
                    raise ValueError(f"{filename} dimensions must match the oriented input")
            for value, name in ((request.camera_to_demo, "camera.npy"),
                                (request.robot_from_demo, "placement.npy")):
                if value is not None:
                    np.save(job / name, np.asarray(value, dtype=float))
            state = {"id": job.name, "status": "queued", "created_at": time.time(),
                     "instruction": request.instruction.strip(),
                     "mode": 'capture_to_human' if request.stop_after == 'edit' else "image_to_robot",
                     'capture_id': request.capture_id,
                     "checkpoint": self.config.profile, "duration_seconds": 4,
                     "hardware_executed": False}
            write_json(job / "request.json", request.model_dump(
                exclude={"image_b64", "mask_b64", "edited_image_b64"}))
            write_json(job / "status.json", state)
            self.executor.submit(self.execute, job, build_command(self.config, job, request))
            return state
        except Exception:
            with self.lock:
                self.pending -= 1
            shutil.rmtree(job, ignore_errors=True)
            raise


def create_app(config: Config) -> FastAPI:
    jobs = Jobs(config)

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.close()

    app = FastAPI(title="Robot Demonstration Studio", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.jobs = jobs

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        # Same-origin custom header prevents cross-site job submissions to localhost.
        if request.method == "POST":
            if request.headers.get("x-retarget-client") != "1":
                return JSONResponse({"detail": "Missing X-Retarget-Client: 1"}, status_code=403)
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > MAX_BODY:
                    return JSONResponse({"detail": "Upload too large (48 MiB maximum)"}, status_code=413)
            request._body = bytes(body)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        if request.url.path in ('/', '/app.js', '/i18n.js', '/style.css', '/retarget.js', '/simulation.js', '/rollout.js') or request.url.path.startswith(('/api/diffusion/', '/api/calibration/', '/api/robot/')):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get('/api/calibration/status')
    def current_calibration(capture_id: str | None = None):
        capture = read_json(jobs.captures.locate(capture_id)/'hand_joints.json') if capture_id else None
        return calibration_status(capture)

    @app.get('/api/diffusion/status')
    def diffusion_status():return jobs.diffusion.status()

    @app.post('/api/diffusion/config')
    def diffusion_config(request: DiffusionConfigInput):
        try:return jobs.diffusion.configure(request.url)
        except ValueError as error:raise HTTPException(400,str(error))

    @app.post('/api/diffusion/start')
    def diffusion_start(request: CheckpointInput):
        try:return jobs.diffusion.start(request.profile,request.partition,request.sol_enabled)
        except ValueError as error:raise HTTPException(400,str(error))
        except Exception as error:raise HTTPException(503,str(error))

    @app.post('/api/diffusion/stop')
    def diffusion_stop():
        try:return jobs.diffusion.stop()
        except Exception as error:raise HTTPException(503,str(error))

    @app.post('/api/jobs/{job_id}/conditioning')
    def conditioning(job_id: str):
        try:return jobs.prepare_conditioning(job_id)
        except (ValueError,KeyError) as error:raise HTTPException(400,str(error))

    @app.post('/api/diffusion/generate',status_code=202)
    def future_video(request: FutureInput):
        try:return jobs.submit_future(request)
        except (ValueError,KeyError) as error:raise HTTPException(400,str(error))

    @app.post('/api/captures', status_code=201)
    def capture_robot():
        with jobs.lock:
            if jobs.hardware.active:raise HTTPException(409,'Stop physical replay before capturing robot state.')
            return jobs.captures.capture()

    @app.get('/api/crop-defaults')
    def crop_defaults():
        return {'defaults': jobs.captures.saved_defaults()}

    @app.get('/api/captures/latest')
    def latest_capture():
        return jobs.captures.latest()

    @app.get('/api/captures/{capture_id}')
    def get_capture(capture_id: str):
        return jobs.captures.snapshot(capture_id)

    @app.post('/api/captures/{capture_id}/crop', status_code=201)
    def crop_capture(capture_id: str, parameters: CropInput):
        try:
            return jobs.captures.crop(capture_id, parameters.model_dump())
        except ValueError as error:
            raise HTTPException(400, str(error))

    @app.get('/api/captures/{capture_id}/files/{filename}')
    def capture_file(capture_id: str, filename: str):
        from .capture_bridge import FILES
        if filename not in FILES:
            raise HTTPException(404, 'Unknown capture file')
        return FileResponse(jobs.captures.locate(capture_id) / filename)

    @app.get("/app.js")
    def javascript():
        return FileResponse(STATIC / "app.js", media_type="text/javascript")

    @app.get("/style.css")
    def css():
        return FileResponse(STATIC / "style.css", media_type="text/css")

    @app.get("/i18n.js")
    def translations():
        return FileResponse(STATIC / "i18n.js", media_type="text/javascript")

    @app.get("/aeonlabs-wordmark.png")
    def wordmark():
        return FileResponse(STATIC / "aeonlabs-wordmark.png", media_type="image/png")

    @app.get("/api/status")
    def status():
        errors = []
        if config.image_provider == 'codex':
            from .codex_edit import readiness
            image_ready, detail = readiness()
            if not image_ready:
                errors.append(detail)
        else:
            image_ready = bool(os.environ.get("GEMINI_API_KEY"))
            if not image_ready:
                errors.append('Image editing requires GEMINI_API_KEY for the explicitly selected Gemini provider.')
        uplift_ready = True
        try:
            if not Path(config.uplift_script).is_file():
                raise ValueError("WiLoR script is missing")
            subprocess.run([config.uplift_python, "--version"], capture_output=True,
                           timeout=5, check=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            uplift_ready = False
            errors.append("WiLoR 环境不可执行或尚未配置")
        client = H3Client(config.h3_url, config.profile, client=httpx.Client(timeout=3))
        try:
            client.preflight()
            h3_ready = True
        except Exception:
            h3_ready = False
            errors.append("H3 未就绪：需要 mirrored-12983 及自定义首帧接口")
        finally:
            client.close()
        return {"ready": image_ready and uplift_ready and h3_ready,
                'image_provider': config.image_provider,
                "image_ready": image_ready, "uplift_ready": uplift_ready, "h3_ready": h3_ready,
                "checkpoint": config.profile, "errors": errors,
                "codex_required": config.image_provider == 'codex'}

    @app.get('/api/robot/status')
    def robot_status():return jobs.hardware.snapshot()

    @app.post('/api/robot/run',status_code=202)
    def run_robot(request: HardwareRunInput):
        with jobs.lock:
            if jobs.pending or jobs.stopping.is_set() or jobs.captures.lock.locked():
                raise HTTPException(409,'Wait for capture and pipeline tasks to finish before hardware execution.')
            job=jobs.locate(request.source_job_id)
            state=read_json(job/'status.json');report=read_json(job/'result/manifest.json')
            if state.get('mode')!='robot_rollout' or state.get('status')!='completed' or not report.get('execution_ready'):
                raise HTTPException(400,'Prepare and validate this rollout before physical execution.')
            try:return jobs.hardware.start(job,request.confirmation,request.workspace_clear,arm_only=request.arm_only)
            except (ValueError,OSError) as exc:raise HTTPException(400,str(exc)) from exc

    @app.post('/api/robot/stop')
    def stop_robot():return jobs.hardware.stop()

    @app.post('/api/robot/heartbeat')
    def robot_heartbeat(request: HardwareHeartbeatInput):
        try:return jobs.hardware.heartbeat(request.run_id)
        except ValueError as exc:raise HTTPException(409,str(exc)) from exc

    @app.post('/api/rollout/prepare',status_code=202)
    def prepare_rollout(request: RolloutInput):
        try:return jobs.submit_rollout(request)
        except (ValueError,KeyError,OSError) as exc:raise HTTPException(400,str(exc)) from exc

    @app.get('/rollout.js')
    def rollout_script():return FileResponse(STATIC/'rollout.js',media_type='text/javascript')

    @app.post('/api/simulation/generate',status_code=202)
    def simulate_generated(request: SimulationInput):
        try:
            return jobs.submit_simulation(request)
        except (ValueError,KeyError,OSError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.get('/simulation.js')
    def simulation_script():
        return FileResponse(STATIC/'simulation.js',media_type='text/javascript')

    @app.post('/api/retarget/generate',status_code=202)
    def retarget_generated(request: RetargetInput):
        try:
            return jobs.submit_generated_retarget(request)
        except (ValueError,TypeError,KeyError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.post('/api/retarget/{job_id}/scene')
    def retarget_scene(job_id: str):
        try:
            return jobs.open_retarget_scene(job_id)
        except (ValueError,KeyError,OSError) as exc:
            raise HTTPException(400,str(exc)) from exc

    @app.get('/retarget.js')
    def retarget_script():
        return FileResponse(STATIC/'retarget.js',media_type='text/javascript')

    @app.post("/api/jobs", status_code=202)
    def submit(request: JobInput):
        try:
            return jobs.submit(request)
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/jobs")
    def history():
        paths = sorted(config.runs.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [jobs.snapshot(path.parent) for path in paths[:50]]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return jobs.snapshot(jobs.locate(job_id))

    @app.get("/api/jobs/{job_id}/input")
    def get_input(job_id: str):
        path = jobs.locate(job_id) / "input.png"
        if not path.is_file():
            raise HTTPException(404, "No input image for imported run")
        return FileResponse(path)

    @app.get("/api/jobs/{job_id}/files/{filename}")
    def file(job_id: str, filename: str):
        job = jobs.locate(job_id)
        if filename not in jobs.snapshot(job)["artifacts"]:
            raise HTTPException(404, "Artifact not available")
        inline = filename.endswith((".png", ".mp4"))
        return FileResponse(job / "result" / filename,
                            filename=None if inline else filename)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument('--image-provider', choices=('codex', 'gemini'), default=os.environ.get('IMAGE_PROVIDER', 'codex'))
    parser.add_argument('--edit-prompt-file', type=Path, default=os.environ.get('CODEX_EDIT_PROMPT_FILE'))
    parser.add_argument("--runs", type=Path, default=Path("out/web-runs"))
    parser.add_argument("--h3-url", default=os.environ.get("H3_URL", "http://127.0.0.1:30121"))
    parser.add_argument("--uplift-python", default=os.environ.get("UPLIFT_PYTHON", ""))
    parser.add_argument("--uplift-script", default=os.environ.get("UPLIFT_SCRIPT", ""))
    parser.add_argument("--image-model", default=os.environ.get("IMAGE_MODEL", "gemini-2.5-flash-image"))
    parser.add_argument("--segmentation-model", default=os.environ.get("SEGMENTATION_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--import-result", type=Path, help="Import a completed run, then exit")
    parser.add_argument("--import-label", default="已保存的 H3 输出 · Retarget 回归结果")
    args = parser.parse_args()
    if args.edit_prompt_file:
        os.environ['CODEX_EDIT_PROMPT_FILE'] = str(args.edit_prompt_file.resolve())
    import uvicorn
    config = Config(args.runs.resolve(), args.h3_url, args.uplift_python, args.uplift_script,
                    args.image_model, args.segmentation_model, image_provider=args.image_provider)
    if args.import_result:
        print(import_result(config, args.import_result, args.import_label))
        return
    uvicorn.run(create_app(config), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
