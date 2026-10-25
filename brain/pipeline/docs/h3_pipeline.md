# H3 → UR5e + Tesollo DG5F inference experiment

The repository **does contain the required right-hand retargeting**. This experiment
uses its `retarget_demonstration` / `HandStage` / `ArmStage` and
`export_robot_demonstration` functions. It does not replace the solver with a
placeholder or export human coordinates under robot joint names.

```mermaid
flowchart LR
    A[Robot first-person RGB] --> B[Masked image-to-image edit]
    B --> C[WiLoR first-frame human skeleton and camera K]
    C --> D[H3 mirrored step12983 + instruction]
    D --> E[Video and decoded human actions]
    E --> F[Camera-to-demonstration transform]
    F --> G[Repository UR5e + DG5F retargeter]
    G --> H[60 Hz robot_actions.npz]
```

## Checkpoint selection and evidence

The latest deployment found is **HF15 Human + T-Rex + AgiBot mirrored,
step12983**, profile `fast-align-hf15-mirrored-12983`. The launcher at
`/capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983/code/webui/runtime/start_mirrored_12983_store.sh`
sets it as the page default, with exact Sol enabled. The recorded page is on
`daint-ln002`, loopback port `8038`; this is not a public URL.

`DEPLOYMENT.json`, `recipe.json`, `inference-bundle/demo-checkpoint-manifest.json`
and `validation/health-final.json` in that deployment agree on the checkpoint.
The original export's SHA256 is
`5553ab5bbe4c1429e9d911b3e19a689e2eecbaa4e981b895849b2617693a161e`.
The deployment merges the trained DiT weights into a base model and keeps a
compact hand sidecar. Using the sidecar with the pristine base would be wrong.
Use the deployment's **model_base and sidecar together**, mirrored Motion VAE
step59000, matching statistics, and chirality policy
`all-left-mirrored-native-gripper-left-v3`.

The recorded successful generation on October 19 used 50 steps, CFG5,
video shift12 / hand shift36, exact CFG parallelism, and no approximate cache.
It produced 107 frames at 24fps (4.458 seconds), with exact initial hand anchors
and unchanged camera intrinsics. See `configs/h3-mirrored-12983.json`.

This establishes **latest deployed and generation-validated**, not “best quality
among all checkpoints.” No matched evaluation ranking including step12983 was
found. The older HF15 report contains tracking MPJPE 14.93mm at step29000 and
14.84mm at step30000; these are a different run/evaluation and cannot rank this
mirrored checkpoint. The training review site required authentication when
accessed from this session.

On October 20 this session could not connect to the recorded inference host or
page; SSH to the older documented login host also lacked usable credentials.
The October 19 saved status must not be read as proof that the service is live now.

## Important interface gap and included fix

The deployed `serve_hands.py` accepts `clip_id`, not arbitrary RGB. Its frame
catalog contains dataset anchors. The WebUI has a WiLoR upload implementation,
but explicitly excludes the mirrored step12983 profile from `UPLIFT_PROFILES`.
Simply posting an image, or choosing another dataset clip, does not condition
this checkpoint on the requested image.

`integrations/h3/` provides a narrow extension for an **experimental H3 runtime
copy**. It adds `/v1/robot-frames`, receiving an NPZ first-frame bundle from the
existing WiLoR uploader. The extension:

- checks the deployed checkpoint manifest and requires a valid right hand;
- converts physical dpl/MANO joints and MANO wrist rotations to `v2_abspose`;
- explicitly sets human embodiment `0` (not AgiBot gripper embodiment);
- encodes the uploaded RGB with the instruction **and with the empty CFG prompt**;
- uses the runtime's existing native conditioning builder, codec, sampler,
  queue and `/v1/videos/{id}/joints` export;
- advertises `robot_custom_frames_v1`, which the pipeline checks before image editing.

Install into a separate runtime copied from the deployed code snapshot:

```bash
python integrations/h3/install.py /absolute/path/to/experimental-minimax-h3
```

Review the resulting diff and launch that runtime with the same deployment
weights, statistics and Sol settings. The current shared launcher hardcodes the
mirrored runtime path, so merely setting `H3_RUNTIME_ROOT` before that launcher
does not select a patched runtime. Use an experimental launcher copy and change
its mirrored-profile `H3_RUNTIME_ROOT` and `H3_SOL_ROOT` assignments to the
patched copy; assign a separate service port/job name. Do not edit the live
deployment in place. No service was restarted or deployed in this experiment.
The installer checks all source edit anchors before writing and refuses a
version mismatch or a second application.

## Run

For the independent upload/camera web UI, script client and controller exports,
see [Robot Demonstration Studio](web_pipeline.md). Neither requires Codex.

Install from the repository checkout (assets/URDFs are needed at runtime):

```bash
cd hand_retarget
uv sync --group dev
```

Use the existing WiLoR interpreter and script on the machine where they are
installed. The script's `--hfov` is used when EXIF has no focal length; EXIF takes
precedence. It creates the crop-adjusted K and an 832×480 RGB first frame. This
crop/resize is part of H3 preprocessing, after the background-preserving edit.

```bash
uv run h3-retarget run \
  --image /data/robot.png --mask /data/end_effectors_mask.png \
  --edited-image /data/human_edited.png \
  --instruction "Grasp the cup with the right hand and lift it." \
  --h3-url http://COMPUTE_HOST:EXPERIMENT_PORT \
  --uplift-python /capstor/store/cscs/2go/go091/aeon-webui/envs/wilor/bin/python \
  --uplift-script /capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983/code/webui/runtime/uplift/uplift_photo.py \
  --hfov 70 --fixed-camera-pitch 30 \
  --seconds 4 --out /data/runs/example
```

The camera values above are **examples**, not a measured calibration. Supply
your actual values. To invoke image-to-image, omit `--edited-image` and set
`GEMINI_API_KEY`; select an image-capable model with `--image-model`.
The adapter implements Google's documented
[Gemini image editing REST contract](https://ai.google.dev/gemini-api/docs/generate-content/image-generation).
The provider is replaceable; it is not the MiniMax video/action model.

An explicit edit mask is a same-resolution binary image: white permits changes
around the end effectors, black preserves decoded RGB pixels exactly. Include
the intended human-hand silhouette in the mask. When omitted, the pipeline now
requests a Gemini polygon proposal and rasterizes it; this automatic mask still
needs quality validation on robot images. A prompt alone cannot guarantee an unchanged
background; the compositor enforces it. The editor must preserve dimensions
and spatial alignment. Image-edit API behavior and edit quality still need
validation on a real robot photo.

Already generated H3 artifacts can be retargeted independently:

```bash
uv run h3-retarget retarget \
  --joints /data/joints.npz --video /data/clean.mp4 \
  --fixed-camera-pitch 30 --seconds 4 \
  --out /data/runs/retarget-only
```

The output directory must be new. Failures write `manifest.json` with a failed
status; an interrupted remote submission retains its ID in `h3_submission.json`.
The CLI never sends commands to the physical robot.

## Geometry and time contracts

H3 outputs `joints: (T,2,21,3)`, `K: (3,3)`, and `fps`. Coordinates are metres
in the optical camera frame (right, down, forward), hand index1 is right.
Joint order is identical to this repository: wrist; index/middle/pinky/ring/thumb
three-joint chains; then the five tips. There is no left/right swap, finger
reordering, guessed scaling, or per-frame wrist centering.

Choose either an explicit fixed-camera pitch or `--camera-transforms` with an
NPY `(4,4)` / `(T,4,4)` mapping each optical frame into a common z-up frame.
In home-pose mode all frames share the same initial-wrist origin. Calibrated
mode preserves the absolute demonstration-frame coordinates. A moving head camera requires
its extrinsics; one image cannot determine its future motion. Fixed-camera
mode is an explicit experimental assumption.

By default the repository places the first human palm at the robot home hand,
then follows displacement. This is useful for kinematic experiments but does
not register a real object/table. `--robot-from-demo` accepts a calibrated
rigid `(4,4)` transform from the uncentered demonstration frame into the
repository's robot world. Thus a point is mapped as
`robot_from_demo @ camera_to_demo[t] @ point_camera`, without subtracting the
initial wrist. Note the robot base
in this world is at `(0,0.6,0.55)`, not the origin.

The selected checkpoint supports a single 107-frame window. Generate that
native window, keep the first96 RGB frames and human-action frames, and produce
exactly4 seconds at24fps. No video speed change is used. Robot actions are
resampled by the existing exporter to60Hz. The last pose is held for1/24 second
to close the final video-frame interval: 241 samples with timestamps0…4
inclusive. Up to107/24 seconds is supported for this checkpoint;5 seconds is
rejected instead of duplicating a video or silently enabling unsupported rollouts.

## Outputs and limits

`robot_actions.npz` is the repository's native output: `arm_q (N,6)`,
`hand_q_measured (N,20)`, velocities and timestamps. Despite the legacy key name
`hand_q_measured`, these are predicted commands, not hardware measurements.
Joint order, units, placement, fingertip error, palm errors, and finger
penetration are recorded in `manifest.json`.

Also retained: edited RGB, first-frame bundle, full native H3 video/joints,
the cropped4-second video, `human_actions.npz`, and viewer-compatible
`human_world_joints.csv`. Use the existing `physics_sim` workflow to assess
robot motion quality. Joint limits and speed limits do not by themselves prove
object contact, collision-free arm motion, reachability, or successful grasp.

The included tests exercise the real retargeter/exporter, 4-second timing,
joint and speed bounds, coordinate transforms, missing hands, mask compositing,
HTTP queue/artifact contracts, checkpoint admission, and the real H3 decoder
and conditioning builder when `H3_TEST_RUNTIME` is set. They do not constitute
a real GPU image→video inference run.

## October 20 execution attempt

The experimental runtime is now actually staged and patched at
`/capstor/store/cscs/2go/go091/aeon-webui/h3-retarget-pipeline-20261020/minimax-h3`.
The source deployment is unchanged. `prepare_experiment.py` prepares this
layout reproducibly, including a same-node GPU client and an automatically
terminated service:

```bash
# From a Daint login environment with the go091-0 GPU allocation:
sbatch /capstor/store/cscs/2go/go091/aeon-webui/h3-retarget-pipeline-20261020/smoke.sbatch
```

This requests one four-GPU node for at most45 minutes, starts the patched
step12983 service on loopback port30121, submits one full50-step generation,
downloads its video and joints, and stops the service when the client finishes
or fails. It uses the existing `mcd16` human-photo/WiLoR bundle, so image editing
and new WiLoR estimation are deliberately outside this GPU smoke test.
The client checks checkpoint identity,107frames/24fps, finite coordinates,
camera intrinsics, and the exact initial right-hand anchor. Its HTTP control
logic has CPU tests; a successful GPU result has not yet been obtained.

The current session is on `eiger-ln002` (x86_64). Its Slurm controller exposes
only `alps-eiger` with no GPU GRES. `sbatch --test-only` rejects the prepared job
with `Invalid generic resource (gres) specification`; `sinfo -M daint` reports
no such registered cluster. SSH to `wuche@daint-ln002` fails with
`Permission denied (publickey)`. The saved runtime/WiLoR interpreters are
aarch64 and fail locally with `Exec format error`. The previously recorded
GPU inference endpoint also could not be reached. **GPU-environment access is
the remaining execution blocker.** No GPU job was submitted.

A separate actual-data regression completed locally: the saved human H3 output
`20261014T201501Z_11ff4303` (profile `fast-align-hf15-human-trex-13500`, OakInk2)
was decoded, cropped to4 seconds and retargeted into241 samples of6+20 joints.
The mean solver fingertip error was4.94mm; maximum measured finger penetration
was0. This is an existing step13500 output, not a new step12983 inference, and
does not change the selected checkpoint. Fixed-camera pitch0 was an explicit
test assumption, not real-scene calibration.
Files are under `out/real-h3-human-13500-regression/`.
