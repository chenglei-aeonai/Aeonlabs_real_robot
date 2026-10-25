# Image-to-robot pipeline

**Future video and hand 3D:** the UI now includes an aeon-webui checkpoint panel
and a direct converted-RGB + captured-GT inference bridge. See
[captured GT diffusion setup](docs/captured_gt_diffusion.md) for the remote
service extension, checkpoint loading, conditioning export, and current GPU
verification limits. Local conditioning export needs no inference service.

Extracted from merged [RobotRetargeting PR #1](https://github.com/nadine-cmd/RobotRetargeting/pull/1),
merge `b0ce2e292371897e57203e5a997e7ccbf84cd9f8`.
`UPSTREAM.json` records the copied files and their original checksums. The
checkout in `../RobotRetargeting/` is kept separate; runtime code and assets here
do not depend on that checkout or the external `AeonLabs/hand_retarget` directory.

The flow is robot image → masked human-hand image edit → WiLoR first-frame
pose → H3 video/hand actions → UR5e/DG5F retargeting → trajectory exports.
Generation and preparation are offline. The separate replay command and the
web Run on robot action require explicit SEND confirmation before physical motion.

## Local setup and startup

From the real-robot repository root:

```bash
./brain/pipeline/setup.sh
./brain/pipeline/run_webui.sh --port 8040
```

Open **http://127.0.0.1:8040**. Jobs and outputs persist in
`brain/pipeline/out/web-runs/`. This has its own Python 3.12 environment at
`hand_retarget/.venv/`, separate from camera capture's environment.
Wrappers limit numerical-library threads to one to avoid CPU oversubscription
on this PC. Override with `RETARGET_NUM_THREADS` if needed.

## Supervised physical replay

Choose a completed retargeting result, **Prepare and check robot rollout**, then
**Run on robot**. **Stop robot** interrupts the owned execution. The browser must
remain connected. Preparation uses captured calibration, plans from the live
start again before execution, and allocates time locally under an eight-second
budget. Historical prepared files retain their previous timing policy.

Read [physical replay and fixed-pose reset](docs/physical_replay.md) for installation
requirements, commands, safeguards, and test instructions. The
[hardware report](docs/eight_second_replay_20261025.md) records a complete 6.31-second
arm-and-hand path, 7.77 seconds with settling, and its remaining finger error.

## Generation services

Local modification: the default image provider is now **headless Codex**, not
the upstream Gemini adapter. It uses the operator's `codex login`; it does not
read or require `GEMINI_API_KEY`. The explicitly selectable legacy Gemini adapter
remains available for upstream compatibility. The default saved prompt is
`hand_retarget/src/hand_retarget/prompts/robot_to_human_hand_arm.txt`, implementing
the operator's hand-and-arm conversion instructions with captured MANO alignment.
Use `CODEX_EDIT_PROMPT_FILE` for an explicit override, including Nadine prompt D
if supplied later. The saved default is not labeled as that missing CSCS prompt.

Starting the UI does not install the inference models. For new generation,
configure these on the machine running the server:

```bash
export H3_URL=http://127.0.0.1:30121
export UPLIFT_PYTHON=/absolute/path/to/wilor/bin/python
export UPLIFT_SCRIPT=/absolute/path/to/uplift_photo.py
# Optional: export CODEX_EDIT_PROMPT_FILE=/absolute/path/to/operator_prompt.txt
# Authenticate once with codex login; GEMINI_API_KEY is not needed.
./brain/pipeline/run_webui.sh --port 8040
```

H3 must have PR #1's custom-frame patch and matching mirrored-12983 deployment
weights. `integrations/h3/` contains that integration, not the pretrained model
or a local GPU installation. The UI reports missing services and credentials.
See [upstream pipeline details](docs/h3_pipeline.md) and
[web/API instructions](docs/web_pipeline.md); their CSCS paths are upstream
deployment examples, not paths on this robot PC.

The CLI is also available:

```bash
./brain/pipeline/run_pipeline.sh --help
./brain/pipeline/run_pipeline.sh retarget --help
```

`scripts/submit_pipeline.py` submits jobs to the web API using standard Python.
Existing H3 joints/video can be retargeted locally without running H3 generation.
The UI's **Capture robot camera** button invokes `../capture/run_capture.sh`
in the separate calibration environment. It connects the paired capture
image/intrinsics/MANO bundle directly to the human-hand editing stage. H3
conditioning and trajectory placement still require their own measured
coordinate transforms; capture intrinsics must not be replaced by guessed FOV.

## Included components and verification

- `hand_retarget/`: inference clients, web UI, retargeter, URDF/mesh assets and tests.
- `integrations/h3/`, `configs/`, `scripts/`, `docs/`: PR integration and clients.
- `real_robot/`: upstream replay-format compatibility code needed by tests;
  it is not launched by these wrappers.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  brain/pipeline/hand_retarget/.venv/bin/python -m pytest \
  brain/pipeline/hand_retarget/tests/test_h3_pipeline.py \
  brain/pipeline/hand_retarget/tests/test_webapp.py
```

The upstream tests mock generation services. Passing them validates local code
and export compatibility, not a fresh GPU-generated demonstration or robot execution.

## Run only image-to-image editing

This command does not require WiLoR, H3, camera access, or robot connections:

```bash
./brain/pipeline/run_image_edit.sh \
  --image brain/capture/results/<capture>/undistorted.png \
  --capture-json brain/capture/results/<capture>/hand_joints.json \
  --out brain/pipeline/out/prompt-d-001
```

Supply `--mask /path/to/binary_mask.png` to use a checked edit region. Otherwise,
headless Codex proposes polygons covering the visible robot hand, wrist hardware,
and arm before editing. A manual mask must also cover the arm to permit its
replacement. Outside-mask decoded RGB pixels
are hard-composited from the original image. Higher-resolution generations with
exactly the same aspect ratio are explicitly normalized to the input dimensions
with Lanczos resampling and recorded in the manifest. Changed aspect ratios are
rejected rather than cropped or stretched. Text-only results are not images.
The Codex runtime must expose its real image-generation/editing tool and save a
generated image artifact; otherwise the job fails with logs.

The command preserves literal prompts and per-job JSONL/stderr logs, records
the prompt checksum, and saves `edit_mask.png`, `edited.png`, and `manifest.json`.
At the operator's request, Codex jobs default to YOLO mode:
`--dangerously-bypass-approvals-and-sandbox`. This disables approval prompts and
the Codex child process's sandbox. Set `CODEX_YOLO=0` to use `workspace-write`
instead. This setting does not change the host chat/tool permission controls.
`CODEX_BIN`, `CODEX_MODEL`, and `CODEX_EDIT_TIMEOUT` optionally
configure the installed CLI, agent model, and per-command timeout. An image-tool
capability probe and a real headless image-edit smoke test succeeded locally.
The test uses a clearly labeled technical prompt, not Nadine prompt D. Its
1280×720 output and validation are under `out/codex-smoke-002/`: 92,384 changed
pixels inside the supplied mask and zero outside it. This edit-only command is
not labeled as the CSCS Stage 2
workflow until its actual instructions are available.

References: [official non-interactive Codex documentation](https://learn.chatgpt.com/docs/non-interactive-mode)
and [official image-generation documentation](https://learn.chatgpt.com/docs/image-generation).

## Robot capture → human-hand editing

1. Keep `./arm/run_controller.sh`, `./hand/run_driver.sh`, and
   `./hand/run_bridge.sh` running, with the arm and fingers stationary. Close
   other camera consumers and hand-overlay listeners.
2. Open the UI at `http://127.0.0.1:8040` and click **Capture robot camera**.
   This is the robot PC's RealSense, independent of the browser's **Use camera**
   option. Capture itself needs no image-editing credentials or generation services.
3. A default **832×640** crop/resize is applied, biased toward the hand and table.
   Adjust output width/height (which set the aspect ratio), crop zoom, and
   horizontal/vertical position in the live preview. Click **Apply crop** to
   save your framing. Width, height, zoom, and horizontal/vertical position persist
   on the server and are automatically restored and applied to every recapture,
   including after page reloads, server restarts, or switching browsers;
   **Reset hand/table framing** recenters from the captured hand landmarks.
4. Review the cropped undistorted frame, robot overlay, and mapped MANO overlay.
   Download the paired JSON or undistorted image if needed.
5. Click **Convert robot hand and arm to human**. This queues only the
   Codex editing stage; it does not require WiLoR or H3 or a camera pitch value.
   The saved hand-and-arm prompt works by default; no Gemini key or prompt-file
   setup is needed. An explicit `CODEX_EDIT_PROMPT_FILE` overrides that default.

Captures persist under `out/captures/<id>/<timestamp>/`. Jobs copy the exact
server-side undistorted image bytes and `hand_joints.json` as `capture.json`;
browser previews/re-encoding cannot replace or alter the paired input. The
headless editor receives the full undistorted K, camera-space MANO points,
their projections, capture provenance, and MANO overlay as a visual reference.
The pose is tagged **capture_ground_truth** and authoritative; edit-only jobs
never run a human pose estimator or modify those reference landmarks. The
recorded wrist conversion and synthetic pinky DIP metadata are retained.

After editing, `edited_mano_overlay.png` draws the same captured GT points onto
the converted image, using only `K @ XYZ` and perspective division. No camera
extrinsics, image crop, or resize is applied. This overlay and
`capture_guidance.json` appear in the UI and result bundle alongside the edited
photo and original capture data. The unchanged pose reference is not a separate
measurement of how accurately image generation reproduced that pose.

API: `POST /api/captures` (requires `X-Retarget-Client: 1`),
`GET /api/captures/<id>`, then `POST /api/jobs` with `capture_id`,
`stop_after: "edit"`, and `instruction`. No manual FOV or pitch is required.
Overlapping captures are rejected; missing/stale feedback or capture failures
produce an error and do not submit an edit. All capture operations are read-only
with respect to robot/hand controllers.

### Crop geometry

`POST /api/captures/<id>/crop` accepts source-pixel `x`, `y`, `width`, `height`,
and integer `output_width`/`output_height` (defaults 832/640). Cropping happens on
undistorted pixels. The crop must lie inside the source and have the same aspect
ratio as the requested output; stretching is rejected. Controls always reference
the original full frame, avoiding repeated resampling as framing changes.

Each applied crop creates a new immutable capture bundle and leaves its source
untouched. Camera-space 3D positions and camera extrinsics remain unchanged.
The saved pixel transform updates K and all 2D projections, including the
half-pixel correction required by resize sampling. Crop origin, source camera,
output dimensions, and pixel transform are recorded in `hand_joints.json` and
passed into editing guidance. `raw.png` remains the original full-frame raw
image; `undistorted.png` and overlays use the cropped output dimensions.

Editing uses the most recently applied crop. Changing a crop control disables
conversion until **Apply crop** saves a matching image/intrinsics/GT bundle.
Only successfully applied settings replace the saved defaults; uncommitted slider
changes do not. The server persists them in `out/crop_defaults.json` and the UI
retrieves them from `GET /api/crop-defaults` on each capture before automatically
applying the crop. Position is stored as a percentage of available crop travel,
so framing adapts to source dimensions. Resetting changes the preview only until
applied. Browser storage remains a fallback; saved defaults remain in effect until
you apply new values. Earlier installations recover defaults from the latest
successfully applied crop bundle if the defaults file does not yet exist.
On page load or refresh, the UI also loads the server defaults and restores the
latest saved capture through `GET /api/captures/latest`, including the crop
preview, controls, overlays, and calibration JSON. This does not acquire the
camera or move the robot; mismatched saved image framing is reapplied from the
original source frame using the saved settings.

### Headless hand-and-arm conversion

The editor receives the full crop-adjusted intrinsic matrix (including principal
point and zero distortion), immutable camera-space MANO GT, a MANO overlay, and
`alignment_reference.png` combining MANO points with captured robot arm geometry.
Nominal arm DH geometry is reconstructed from the recorded arm joints and
camera/base calibration; its camera-space positions are projected with K only.
The human forearm guide connects the captured robot elbow to the mapped MANO
wrist. These robot arm origins guide alignment; they are not human joint detections.

Human overlays draw the elbow-to-MANO-joint-0 segment and use that same joint 0
as the hand root. Tool0/TCP and the UR5e wrist axes remain in diagnostic data but
are not drawn as extra human wrists. This display convention is also recorded in
`human_forearm_reference` and the saved image-edit prompt. Wrist rotation for
diffusion is rooted at the same mapped MANO joint 0.

The saved hand-and-arm prompt requires matching projected wrist, finger joints,
fingertips, perspective, and hand scale. Its text and all conditioning metadata
are logged per job. Image generation is not a hard geometric constraint solver:
`alignment_verified` remains false. Capture-conditioned candidates are shown with
a review notice until an independent alignment check is available, rather than
being claimed verified. Such drafts do not automatically continue into H3 generation. The GT overlay
supports visual inspection; no human pose is estimated to replace the capture GT.

The calibrated capture UI no longer asks for horizontal FOV or fixed pitch.
Legacy API consumers can still supply those fields for older upload workflows.
Captured jobs derive the compatibility FOV scalar from K on the server and use
saved camera/base extrinsics for downstream spatial placement, not an assumed
pitch. Editing itself uses only camera-space points and intrinsics.

The **Future video + hand 3D** panel includes native video playback for RGB and
predicted-skeleton overlays, plus a selector for previous generation jobs.
Right-handed capture identity and MANO wrist rotation are shared by Step 02
and diffusion; see [the hand 3D contract](docs/hand_3d_contract.md).

## Generated trajectory → DG5F preview (Step 05)

After Future video + hand 3D completes, choose that generation in **Retarget to
robot hand** and click **Retarget generated hand**. This runs locally without a
GPU allocation or robot connection. The saved calibration paired with that
generation supplies `robot_base_from_camera = inverse(camera_from_robot_base)`;
no guessed camera orientation or simulation-world offset is applied.

The vector retargeter produces 20 DG5F angles at the generated frame rate,
respecting the existing joint and inter-frame speed limits. A fixed anatomical
offset derived from the capture aligns the human palm frame with the measured
robot palm; subsequent generated palm motion is expressed in robot-base metres.
The interactive URDF-joint skeleton viewer supports orbit, zoom, playback and
frame scrubbing, alongside the generated MANO skeleton and angle readout.

Download `retargeted_hand.npz` (angles, calibrated MANO/robot positions, palm
poses, timing and transform), `retarget_preview.json`, and `manifest.json`.
The report records calibration provenance, transform round-trip error, scaled
fingertip fitting error and finger penetration estimates. These are offline
hand results: arm IK, workspace/table clearance and execution are not validated.
There is no command to move hardware in this step.

For mesh rendering, click **Open URDF hand + table + board** in Step 05. This
opens a loopback Viser scene inside the page (normally port 8041), using the
DG5F visual meshes and the same retargeted joint angles/palm poses. Its own
playback slider animates the hand; the generated MANO skeleton can be toggled.

The board uses the source legacy calibration's world frame and the configured
174 × 290 mm ChArUco pattern. Its saved pose is not updated if the loose board
moves after calibration. The table surface follows that calibration's printed
board plane. The shown 1.0 × 0.7 m table footprint and 15 mm visual thickness
are illustrative, not measured geometry; viewer controls adjust its extent,
position on the plane and board-to-table height offset. These controls never
modify the calibration or robot safety guard. `scene_geometry.json` records
the initial scene transforms, plane and assumptions. The original calibration
file's hash is checked before loading, so a replaced transform cannot silently
change the scene.

## MuJoCo scene and cube replay (Step 06)

Select a completed retargeting result, click the **center of the cube's top
face** in its paired first frame, and enter its estimated edge length, table
heading, mass and friction. Click **Run MuJoCo contact replay**. The camera ray
is intersected with the calibrated plane at one cube-edge height; half the edge
is subtracted along the normal to obtain the cube's center. This assumes a
cube resting flat on the table. It is an approximate reconstruction, not an
object detector, tracker, or measured estimate of mass/friction.

MuJoCo 3.14.0 uses the DG5F URDF collision meshes, 20 position drives and an
interpolated prescribed palm trajectory. The cube has a free joint and is moved
only by gravity and contacts; no weld or artificial attachment is used. The
calibration board is visual context at its saved pose. Table contact uses an
infinite calibrated plane; other scene objects and UR5e arm dynamics are not
reconstructed. Contact results depend on convex collision approximations,
actuator/contact assumptions and object estimates, and do not validate hardware.

Outputs: calibrated-camera and overview MP4s, `mujoco_motion.npz`, a portable
`mujoco_scene.zip` containing MJCF and meshes, and `manifest.json` with contacts,
tracking error, closest hand/cube surface gap and lift outcome. A reported lift
requires the cube bottom to stay over 20 mm above the table with hand contact
and no table contact for at least 0.25 seconds. A missed grasp is reported as
such; the recorded hand trajectory is not moved toward the cube to manufacture
success.

The calibrated camera uses full crop-adjusted intrinsics and the optical-to-
OpenGL axis conversion. The principal-point pixel-center convention is checked
with an actual rendered-point regression test; see MuJoCo's
[camera parameter reference](https://mujoco.readthedocs.io/en/3.3.7/XMLreference.html#body-camera).

A first test of generation `f6e5c35…` and retargeting `f1addd0b…`, using the larger
pink block (approximately 70.45 mm edge inferred from manually identified face
corners, assumed 50 g mass and 0.8 friction), produced no contact or lift. Its
minimum simulated hand–cube surface gap was about 203 mm. That trajectory needs
a reach/grasp change before it can manipulate this estimated object.

## Latest calibration in the webpage

New captures automatically use the newest complete timestamped robot/camera pair
in `calibration/results/original_legacy/` for UR5e and camera `138322253845`.
The summary text alone is insufficient: both `.npy` transforms must exist and
pass rigid-transform validation. An incomplete pair is ignored; an invalid newest
complete pair is reported as unavailable instead of silently using an older one.

The page checks calibration every five seconds and displays its timestamp.
After recalibrating, click **Capture** again; saved crop defaults still apply.
New conversion, conditioning export, generation, retargeting, and simulation
requests reject captures from older calibrations. Regenerate those stages from
the new capture. Existing results and jobs already running retain their original
calibration for reproducibility; history labels identify older results. Refresh
the page once after installing this UI update. No controller restart is needed.

Stage 2 captured-frame edits use a deterministic mask from projected MANO,
robot finger, and arm joints; they do not call an independent hand detector.
A supplied manual mask overrides this envelope. The envelope does not estimate
object occlusion, so inspect the result around table objects. The editor must
preserve all 21 GT landmarks and render a right hand with its dorsal surface
(back) facing the camera. GT coordinates and wrist rotation remain unchanged.

The headless editor requests `gpt-image-2.5-sunburst`, `quality=max`, when its
image tool exposes those controls. Codex's image-tool backend is managed and
this integration cannot pin or independently verify its model; manifests record
that distinction (`image_model_verified: false`). `CODEX_MODEL` selects the
reasoning model, not the image model. Exact skeletal alignment remains subject
to visual review; an editor self-check is not a measured alignment guarantee.
