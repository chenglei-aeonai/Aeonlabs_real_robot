# Hand 3D input definition and compatibility audit

Checked on 2026-10-24 against aeon-webui master
[`a8b2ada`](https://github.com/chenglei-aeonai/aeon-webui/blob/a8b2ada7a697a834a323ef77bf00ce5debc2f459/runtime/uplift/custom_frame.py)
and the H3 demo runtime's
[`v2_abspose` decoder](https://github.com/chenglei-aeonai/minimax-h3/blob/145c4918b487c529178e6a3a81beceaac35d8ec0/training/h3_joint/representations.py).
The GitHub aeon-webui master revision matched our local reference checkout.

## Definition of hand 3D

The external bundle contains **21 Cartesian landmarks per hand**, including the
wrist, in DPL/MANO order. These are positions, not MANO joint angles, shape
coefficients, a mesh, robot actuator angles, or normalized image coordinates.

| Property | Definition |
| --- | --- |
| `joints3d` | `(2,21,3)`; axes are hand, landmark, XYZ |
| Hand slots | `0 = left`, `1 = right` |
| Units | Metres; 1 m = 1000 mm |
| Coordinate origin | Camera optical centre |
| Positive axes | X right in image, Y down in image, Z forward from camera |
| Positions | Absolute camera coordinates, including the wrist |
| `valid` | Boolean `[false,true]`; missing left points are zero |
| `wrist_R` | `(2,3,3)`; MANO wrist-local → camera rotation; absent left is identity |
| `K` | `(3,3)` crop-adjusted undistorted intrinsics, focal lengths/principal point in pixels |
| `frame` | `(H,W,3)` RGB uint8, matched to K; current crop is 832×640 |

The exact joint indices are:

| Indices | Landmarks, in order |
| --- | --- |
| 0 | Wrist |
| 1, 2, 3 | Index MCP, PIP, DIP |
| 4, 5, 6 | Middle MCP, PIP, DIP |
| 7, 8, 9 | Pinky MCP, PIP, DIP |
| 10, 11, 12 | Ring MCP, PIP, DIP |
| 13, 14, 15 | Thumb CMC, MCP, IP |
| 16, 17, 18, 19, 20 | Index, middle, pinky, ring, thumb tips |

This matches the demo's OpenPose-to-DPL permutation:
`[0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3,8,12,20,16,4]`.
Do not apply that permutation again to our already-ordered capture.

Our positions come from measured robot state, forward kinematics, calibration,
and the saved robot-to-human landmark mapping. The shifted wrist and interpolated
pinky DIP are synthetic. They are the authoritative conditioning pose, but they
are not independently measured human anatomy or an official fitted MANO mesh.
The export JSON records landmark names and synthetic indices.

## Rotation, projection, and model representation

For the right hand, derive forward from wrist to middle MCP and normal from
`cross(pinky_MCP - index_MCP, forward)`. Normalize both; the side axis is
`cross(normal, forward)`. Stack these axes as rows in G. Then
`R = G.T @ S`, where S is the nearest proper rotation to DPL's measured right-hand
signature. For row-vector positions:

```text
J_wrist = (J_camera - wrist_camera) @ R
J_camera = J_wrist @ R.T + wrist_camera
q = J_camera @ K.T
uv = q[:, :2] / q[:, 2:3]
```

Projection uses zero distortion and no further extrinsic transform. Cropping
changes K and pixels, not the camera-frame positions.

The custom-frame NPZ is a physical input format. The H3 `v2_abspose` model uses
138 channels internally at frame zero:

| Channel slice | Meaning |
| --- | --- |
| 0:3, 3:6 | Left/right absolute camera-space wrist position |
| 6:12, 12:18 | Left/right rotation in 6D: first column of R followed by second column |
| 18:78, 78:138 | Left/right 20 non-wrist joints in wrist-local coordinates, flattened XYZ |

Later `v2_abspose` wrist positions are relative to the frame-zero wrist;
rotations remain absolute, and finger coordinates remain wrist-local. The
runtime handles representation encoding, checkpoint statistics and masks.
Do not normalize the external NPZ or manually mirror the right hand.
Contact checkpoints add their own conditioning requirements; our capture has no
contact measurements and requires an explicit zero-contact assumption for those profiles.

## Metrics and their meanings

Let J be the supplied joints, R the wrist rotation, and S the proper reference
signature. All positional Euclidean norms below use metres unless stated otherwise.

| Metric | Definition / current limit | Interpretation |
| --- | --- | --- |
| Wrist convention angle | `degrees(acos(clip((trace(G_local @ S.T)-1)/2,-1,1)))`; ≤0.001° | Self-consistency with the reference axes used to construct R |
| Camera roundtrip | `max_j ||((J_j-J_0)@R)@R.T+J_0-J_j||₂`, non-wrist joints; ≤1e-8 m | Numerical reversibility |
| Generated initial anchor | `max_j ||J_generated[0,right,j]-J_input[right,j]||₂`; ≤1e-4 m (0.1 mm) | Preservation of all 21 supplied starting landmarks |
| Intrinsics | Per-element `abs(K_generated-K_input) ≤ 1e-4`, zero relative tolerance | Camera preservation |
| Joints inside image | Count with `0 ≤ u < W`, `0 ≤ v < H` | Framing only; does not establish visibility through occlusion |
| MPJPE | `mean_j ||J_pred,j-J_reference,j||₂`; ×1000 gives mm | Requires an independent 3D reference to measure physical accuracy |
| Mean 2D reprojection error | `mean_j ||project(K,J_j)-uv_reference,j||₂`, pixels | Requires independently detected/annotated image landmarks |

We do **not** currently measure physical MPJPE or independent human-image
reprojection accuracy. Saved-projection checks and overlays use the supplied
joints themselves. The upstream photo demo reports reprojection against WiLoR's
own 2D detections, along with depth, wrist-to-joint span, in-frame count, and
WiLoR-vs-geometry orientation. Our capture workflow does not run WiLoR, so those
WiLoR comparison metrics cannot be reported as measured values.

DPL's upstream convention gate uses a 45° median threshold across samples to
catch major convention flips. Our 0.001° construction check has a different
purpose and does not imply greater calibration accuracy.

## Audit results and intentional differences

Evaluated saved export from job `9c3c0a68e4154c21ac8f5e81b0ad2f2f`:

- All 21 joint labels/order and the right-hand signature match upstream.
- Executing upstream geometry gives our R after nearest-SO(3) projection.
- Encoding the saved positions into the 138-channel anchor and executing the
  actual upstream float32 decoder reconstructs the right hand with maximum
  error **3.55e-8 m (0.0000355 mm)** and mean error **1.92e-8 m**.
- Wrist depth is **0.514795 m**; maximum wrist-to-landmark distance is
  **0.213750 m**; **21/21** landmarks project inside the crop. These are
  descriptive measurements, not anatomical or alignment acceptance criteria.
- Our NPZ uses float64 geometry/K to preserve captured values; the photo demo
  writes float32. The runtime decode test above exercises float32 conversion.
- Upstream uses a rounded reference matrix directly: its resulting rotation has
  determinant **1.0069** and maximum absolute orthogonality residual **0.0065**.
  We use the nearest proper rotation. The convention matches, but matrix values
  are deliberately not bit-identical.
- The demo includes an `overlay` preview array and normally uses 832×480.
  Our transport provides the five conditioning arrays and separate overlay files,
  with the user's 832×640 image/K. The documentation lists overlay as a preview;
  live loader acceptance and inference at this size remain to be verified.

Local evidence is saved in `out/format-audit/report.json`, with the audit script
and downloaded runtime source beside it. The source functions were executed
without loading weights. The deployed CSCS custom-loader implementation is a
separate Store checkout described upstream; this audit does not establish that
our chosen endpoint has that patch, or that GPU inference succeeds there.

## Right-only capture and server validity

This robot is always right-handed. Slots remain `[left, right]`, with
`valid=[false,true]`; a missing left hand is not filled with the right hand.
Capture guidance rejects explicit left-hand metadata. Older robot captures
without a handedness field retain the fixed right-hand interpretation.
Step 02 and diffusion share `mano_convention.py`: wrist rotation maps MANO-right
local coordinates to the optical camera, with joint 0 as the origin. Step 02
receives this rotation and a no-mirroring instruction; it still needs visual
review because image editing does not guarantee exact geometric alignment.

The aeon robot branch launches `robot_gt_runtime.py`, which applies a checked,
in-memory patch to the upstream custom-frame loader. The input bundle supplies
channel validity for both CFG branches and the latent codec; demo-template
validity must not override it. Missing-hand template channels are zeroed.
Shared CSCS runtime files are untouched. New outputs include `valid` and
`wrist_R`; the client checks right-only validity and initial wrist rotation
agreement within 0.1 degrees. Existing results without these fields remain
readable and are identified as lacking those checks in the manifest.
A GPU server launched before this update must be restarted to load the wrapper.

Step 02 also receives named camera-space orientation anchors and a reference
with thumb, index, middle, ring, and pinky labels. Its edit instructions prohibit
mirroring, thumb/pinky exchange, or turning the palm over, and require checking
the candidate against the original capture. Captured-image edits only succeed
when the editor returns `orientation_check=matched`; missing, uncertain, or
mismatched checks retain the candidate for diagnosis but fail the job, preventing
its selection as diffusion input. This is a model self-check, not an independent
image-pose measurement (`orientation_independently_verified=false`). Existing
converted images are unchanged; rerun Step 02 to apply these constraints.
