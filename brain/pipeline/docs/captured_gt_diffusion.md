# Converted capture → H3 future video and 3D hands

This integration was checked against
[`chenglei-aeonai/aeon-webui`](https://github.com/chenglei-aeonai/aeon-webui),
master `a8b2ada7a697a834a323ef77bf00ce5debc2f459`.
The local reference checkout is `brain/pipeline/vendor/aeon-webui/` (ignored by
Git). `configs/aeon-checkpoints.json` records the inspected checkpoint catalog.

## What runs where

The robot PC runs capture, cropping, image editing, UI, and conditioning export.
The connected aeon-webui installation owns checkpoint loading and GPU inference.
Upstream launchers request a four-GPU CSCS node and load matching model bases,
hand codecs and statistics from Store. This PC's 12 GB RTX 3080 Ti does not
provide that deployment. No checkpoint weights are included or downloaded by
this integration, and selecting a catalog entry does not mean it is loaded.

## Configure the inference service

Use a separate aeon-webui checkout/service with the existing custom-frame H3
runtime described in upstream `docs/photo-uplift.md`. Install our small GT import
extension in that UI checkout, **on the inference host**:

```bash
python /path/to/brain/pipeline/integrations/aeon_webui/install.py \
  /absolute/path/to/isolated/aeon-webui
```

The installer copies `robot_gt_bridge.py` and adds two routes before static
mounts. It checks its edit anchor, compiles the result, and refuses to overwrite
an existing helper or apply twice. It does not modify model weights, submit GPU
jobs, or start/restart services. Restart the isolated aeon-webui service after
installation. The H3 runtime must already support `uploads` custom frames and
`POST /v1/custom-frames/reload`; the extension does not install H3 itself.

Expose that service through your CSCS SSH tunnel. For example, replacing the
user/node/port placeholders with your actual installation:

```bash
ssh -J <CSCS-user>@ela.cscs.ch -N \
  -L 8017:127.0.0.1:<remote-webui-port> <CSCS-user>@<login-node>
```

Then enter `http://127.0.0.1:8017` in our **Future video + hand 3D** panel and
click **Connect / refresh**. Alternatively set `AEON_WEBUI_URL` before starting
our UI. Connection settings persist in `out/diffusion_config.json`.
Use the root service URL: multi-tab deployments are detected automatically,
with one persistent leased tab token and heartbeat during local UI polling and
generation. Single-app deployments use their existing endpoints directly.

## UI workflow

The upper-right GPU status button opens **Server startup & checkpoint**. It uses
the connected aeon-webui's queue/boot/ready state, rather than the older local
WiLoR/H3 readiness badge. The console shows the job ID, node, queue, remaining
walltime, and live boot-log tail. Updates arrive every five seconds (upstream
status may be cached for ten). Click the same button to reopen the console;
closing it keeps the allocation and polling running.

The robot-GT branch's `startup_status_v1` endpoint also retains an owned job's
terminal state and boot log after it leaves Slurm's active queue. Failed jobs
therefore remain visible instead of silently becoming “stopped.”

1. Capture the stationary robot, apply the saved crop, and convert to a human
   hand/arm as usual. Inspect the GT overlay; image editing is approximate.
2. In **Future video + hand 3D**, choose the completed converted frame.
3. **Export GT conditioning bundle** works locally without a GPU service.
4. Connect the service and choose a supported checkpoint. **Load checkpoint**
   uses upstream `/api/hands/start` with the chosen profile, Normal/Debug queue,
   Sol selection and one server. It starts a real remote GPU allocation.
   **Stop server** uses upstream `/api/hands/stop`.
5. Wait for the selected profile to report `ready`, describe the future action,
   and click **Generate future video + hand 3D**. This explicitly uses the
   selected converted frame as conditioning; it does not repeat image editing.

The dropdown shows the actual remote profiles when connected and a pinned
catalog otherwise. Profiles excluded by upstream's custom-frame support are
disabled. Contact-conditioned profiles require an explicit **no initial
contact or force** assumption because the capture supplies no contact sensors;
otherwise select a non-contact profile. No contact observation is fabricated
without that selection. Match-checkpoint recipe stays enabled, using the
upstream profile's shifts, 50 steps and CFG 5.

## Conditioning contract

See [hand 3D definition and metric audit](hand_3d_contract.md) for the exact
joint order, coordinate/rotation equations, model channel layout, metric
definitions, and comparison with the upstream demo on a captured bundle.

`first_frame_gt.npz` contains:

| Key | Shape / meaning |
| --- | --- |
| `frame` | `(H,W,3)` uint8, exact converted RGB pixels |
| `K` | `(3,3)`, exact crop-adjusted undistorted intrinsics |
| `joints3d` | `(2,21,3)` metres in optical camera coordinates; right hand index 1 |
| `valid` | `[false,true]`; no invented left hand |
| `wrist_R` | `(2,3,3)`, proper MANO-convention rotations |

All 21 captured positions remain unchanged. There is no WiLoR pass, pose fitting,
wrist centering, depth rescaling, or camera extrinsic applied to them. Wrist
orientation follows upstream `wrist_from_geometry`: wrist→middle-MCP and
index→pinky axes, using DPL's right-hand signature from
`nadine-cmd/data-preprocessing-loop` at
`48575bd9dbec73c7180cc430fa6cee31c62ec187`. Its rounded signature is projected
to the nearest proper rotation; this changes no camera-space joint positions.
`wrist_R` maps MANO-right wrist coordinates into the optical camera frame:
for row vectors, `J_camera = J_wrist @ wrist_R.T + wrist_position_camera`.
The exporter checks chirality-aware axes against the MANO-right signature,
proper rotation, and camera-space reconstruction. A 180° convention flip is
covered by a rejection test. The right-hand matrix, convention error in degrees,
and reconstruction error in metres are written to `first_frame_gt.json`.

Input dimensions must be multiples of 32 between 32 and 2048. The default
832×640 crop satisfies this. The bridge does not silently recrop to upstream's
phone-upload 832×480 preset. Actual inference at the chosen size still needs
validation on the deployed H3 runtime/GPU.

The remote GT endpoint validates the bundle, saves its exact bytes into the
existing custom-frame catalog, registers the candidate, and reloads the
running H3 server. It returns the SHA-256 and explicitly confirms no pose
re-estimation. Submission fails if the checkpoint, custom-frame reload,
bundle identity, or contact assumptions do not match.

## Outputs and verification

Generation jobs appear in normal local job history and persist under
`out/web-runs/<job>/result/`:

- `video.mp4`: native future video, not artificially limited to four seconds.
- `predicted_hand_overlay.mp4`: upstream's predicted skeleton video, when supplied.
- `future_hands.npz`: generated `(T,2,21,3)` camera-space hands, `K`, and `fps`.
- `first_frame_gt.npz` / `.json`: exact conditioning and provenance.
- `diffusion_submission.json`: remote job/clip IDs, persisted immediately.
- `manifest.json` and `bundle.zip`: validation and downloadable results.

Before marking success, the worker checks finite right-hand positions, camera
intrinsics, first-frame GT error ≤0.1 mm, and matching video resolution, frame
count and FPS. It does not send arm/hand commands or replay generated motion.
Timeout or local shutdown may leave remote work running; use the saved job ID
and remote UI to inspect it. Restart never automatically resubmits a GPU job.

CPU tests use mock inference responses and real NPZ/video artifacts. They verify
registration, checkpoint/start APIs, multi-tab routing, GT preservation, contact
gating, downloads, and export validation. A real GPU generation has not been
run from this robot PC: no inference URL/tunnel is currently configured and the
documented local ports 8017, 8027 and 8038 are unreachable.

### Live generation progress

The Future video + hand 3D results panel polls its selected active job every
approximately two seconds, independently of the Step 02 result selection.
It displays the server's phase, sampling step/total, stage ETA when available,
elapsed time, and the age of the last progress update. Encoding, decoding,
preview rendering, downloading, and validation use an indeterminate indicator
rather than an invented overall percentage. Temporary status-fetch errors are
shown and retried. Refreshing the page does not stop a running generation.
