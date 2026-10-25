# Robot Demonstration Studio

The pipeline runs independently of Codex. Chrome and a Python script are clients
of the same persistent job API:

```text
Chrome / submit_pipeline.py → POST /api/jobs
  → automatic hand-region proposal, or supplied mask
  → Gemini image editing, or supplied edited image
  → WiLoR first-frame pose
  → patched H3 mirrored-12983 engine
  → repository UR5e + Tesollo DG5F right-hand retargeter
  → NPZ / CSV / trajectory JSON / controller messages / video
```

The UI reuses the existing Aeon inference page's original
`aeonlabs-wordmark.png`, Cormorant Garamond / DM Sans fonts, warm background,
gold buttons, and loopback/SSH access model. It does not modify the shared
inference page or its services. English is the default; the top-right
中文 / English button switches interface language and remembers the selection.
Switching preserves uploads, masks, form values, the selected result and video
playback position. User instructions, filenames and raw technical diagnostics
retain their original content.

## Start the independent page

Install on the architecture where it will run:

```bash
cd /users/wuche/dev/RobotRetargeting-h3-pipeline
uv sync --project hand_retarget --group dev
export H3_URL=http://127.0.0.1:30121
export UPLIFT_PYTHON=/capstor/store/cscs/2go/go091/aeon-webui/envs/wilor/bin/python
export UPLIFT_SCRIPT=/capstor/store/cscs/2go/go091/aeon-webui/hf15-mirrored-12983/code/webui/runtime/uplift/uplift_photo.py
# Configure GEMINI_API_KEY in the server environment through your credential setup.
bash scripts/run_webui.sh --port 8040 --runs out/web-runs
```

`RETARGET_PYTHON` selects an already-installed interpreter. `IMAGE_MODEL` and
`SEGMENTATION_MODEL` select the Gemini models; integration defaults are
`gemini-2.5-flash-image` and `gemini-2.5-flash`. These are not a measured ranking
or a claim of account availability. Credentials stay on the server.

The page is at `http://127.0.0.1:8040` **on that server**. From a laptop:

```bash
# The destination must be the exact host running the UI.
bash scripts/connect_webui.sh wuche eiger-ln002
```

Then visit **http://localhost:8040** in local Chrome. Equivalent tunnel:

```bash
ssh -J wuche@ela.cscs.ch -o ExitOnForwardFailure=yes \
  -N -L 8040:127.0.0.1:8040 wuche@eiger-ln002
```

The localhost tunnel also enables browser camera capture. A static HTML opened
with `file://` cannot perform the server-side inference.

On October 20 the UI was started on `eiger-ln002:8040`, with logs at
`out/webui.log` and PID at `out/webui.pid`. The x86 test environment is
`/tmp/robot-retargeting-h3-venv`; it may be cleaned by the host. Use the installation
command above for persistent deployment. The page serves imported real H3
regression results and reports missing image credentials, WiLoR and H3 as
unavailable. New end-to-end generation has **not** been validated there.

## Complete stack on Daint

The prepared experimental runtime is described in [h3_pipeline.md](h3_pipeline.md).
To keep H3, WiLoR and the web worker on one allocated GPU node:

```bash
# Run on Daint; create an aarch64 environment here, not on Eiger.
export RETARGET_REPO=/users/wuche/dev/RobotRetargeting-h3-pipeline
uv sync --project "$RETARGET_REPO/hand_retarget" --group dev
export RETARGET_PYTHON="$RETARGET_REPO/hand_retarget/.venv/bin/python"
# GEMINI_API_KEY must already be configured in this environment.
sbatch "$RETARGET_REPO/integrations/h3/serve_pipeline.sbatch"
```

The job requests four GPUs for up to two hours. Its log prints the allocated
hostname; tunnel to **that exact node** with `connect_webui.sh`. H3 loads in the
background and the page reports readiness. Both services stop when either exits.
The paired checkpoint/base and exact Sol settings come from the staged launcher.

This job was syntax-checked, not successfully submitted from the Eiger session.
Eiger has no GPU GRES and Daint SSH authentication was unavailable. Daint
dependency installation and real GPU execution remain unverified.

## Browser workflow

Upload an RGB image or capture a frame, enter an instruction and camera HFOV/pitch,
then submit once. Camera values persist in the browser. A camera-to-demo JSON
transform can replace fixed pitch. Fixed-camera mode assumes no camera movement.

By default Gemini proposes robotic-hand polygons and the worker rasterizes them.
Manual painting and mask upload are available. This automatic proposal is not a
verified geometric segmentation; evaluate it on each new camera/robot setup.
Pixels outside the mask are copied from the original decoded RGB. The image
adapter rejects changed image dimensions rather than silently cropping/stretching.
Hand pose, wrist alignment and object contact are not proven by compositing.
Editing/masking quality on actual Tesollo photos remains untested.

The page shows job stages, the edited image, four-second human video, robot joint
curves and downloads. Refreshing restores the last selected job. H3 preflight
checks checkpoint identity and the custom-frame capability before paid editing.
Jobs execute serially, with at most eight queued/running jobs. Inputs, parameters,
progress, logs and results persist under the configured runs directory. Restart
marks unfinished jobs interrupted; it does not automatically resubmit GPU work.

## Script workflow

The HTTP client uses only Python's standard library:

```bash
python scripts/submit_pipeline.py \
  --url http://localhost:8040 --image robot.png \
  --instruction "Grasp the cup with the right hand and lift it." \
  --hfov 70 --fixed-camera-pitch 30 --out results/run-001
```

Replace the example camera parameters with measured values. Optional `--mask`,
`--edited-image`, `--camera-transforms` and `--robot-from-demo` match the browser
overrides. This HTTP client takes transforms as JSON; the underlying CLI takes NPY.
The CLI also works without a web server:

```bash
uv run --project hand_retarget h3-retarget run \
  --image robot.png --instruction "Grasp the cup." \
  --h3-url "$H3_URL" --uplift-python "$UPLIFT_PYTHON" \
  --uplift-script "$UPLIFT_SCRIPT" \
  --hfov 70 --fixed-camera-pitch 30 --out results/run-002
```

## Exports and controller compatibility

| File | Contract |
| --- | --- |
| `robot_actions.npz` | Existing replay format: timestamp, 6 arm and 20 hand positions/velocities; radians, 60 Hz, 241 samples spanning 0–4 s |
| `robot_actions.csv` | Named time, position and velocity columns |
| `trajectory.json` | Joint names, per-point time/positions/velocities, units and spatial placement |
| `controller_commands.json` | Existing arm ZMQ `{time,path}` payload and timed DG5F UDP `hand_target` messages |
| `video.mp4` | Human demonstration, 96 frames at 24 fps |
| `human_actions.npz` | Corresponding human skeleton sequence |
| `manifest.json` | Checkpoint, edit provenance, transforms, solver diagnostics and export checks |
| `bundle.zip` | Available results above plus edited image and mask |

The NPZ is directly readable by `real_robot/replay_real.py`, which handles the
start pose, synchronized dispatch and state/stop checks. Controller JSON is a
payload bundle, not a live connection. The web worker does not execute hardware.

Without `robot_from_demo`, trajectories align to the repository's home hand pose;
this does not register the real table/object. With calibration, the uncentered
demonstration coordinates are preserved: `robot_from_demo @ camera_to_demo[t]`
maps an optical-camera point into robot world. Supplying that transform does
not prove reachability, collision avoidance or task success.
Solver fingertip error measures retargeting fit, not real-world task success.

On graceful server shutdown, the worker kills its active local subprocess group
and marks queued jobs interrupted without starting them. An already submitted
remote H3 job may continue; its ID is saved in the result's `h3_submission.json`.
Restarting the UI never automatically resubmits interrupted jobs.

## Verification and imported results

`tests/test_webapp.py` covers upload/mask validation, local request guards,
literal prompt arguments, interrupted jobs, credential failures, automatic-mask
parsing, queue failures and a real subprocess through retargeting/export.
The resulting NPZ is checked using the real-robot `Demonstration` loader without
opening hardware connections. Upstream GPU generation is replaced in this test.

```bash
cd hand_retarget
uv run pytest tests/test_webapp.py tests/test_h3_pipeline.py tests/test_retargeting.py
```

The actual saved H3 step13500 sample was reprocessed into
`out/web-real-h3-regression` and imported into the UI, explicitly marked as a
historical result. New generation still uses mirrored step12983. Import with:

```bash
uv run --project hand_retarget h3-retarget-web \
  --runs out/web-runs --import-result out/web-real-h3-regression \
  --import-label "Historical H3 step13500 · retarget regression"
```

Optional browser checks require Playwright, Chromium, image credentials unset,
and an imported completed result:

```bash
python scripts/verify_webui.py --url http://localhost:8040 --out out/browser-checks
```

They exercise upload, mask painting, credential errors, a browser test camera,
video metadata, plotting, NPZ download, refresh persistence and mobile overflow.
They do not invoke real image or video generation.

## Headless Codex

[`codex exec`](https://learn.chatgpt.com/docs/non-interactive-mode) supports
non-interactive runs, JSON events and saved authentication. A wrapper could
request an edit and collect the file **if that runtime exposes image generation**.
Each invocation still runs an agent, with authentication, limits, tool
availability and output parsing to manage. The official
[image guide](https://learn.chatgpt.com/docs/image-generation) points programmatic
image generation to the Images API.

This project's normal runtime does not invoke Codex or reuse its credentials.
Headless image generation was not verified: the installed CLI on this Eiger
host is an ARM binary and fails with `Exec format error`.
