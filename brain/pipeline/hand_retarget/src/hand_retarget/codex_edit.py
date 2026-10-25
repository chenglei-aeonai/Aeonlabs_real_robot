"""Headless Codex adapters, using the operator's existing Codex login."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

from PIL import Image, ImageOps


def prompt_file() -> Path:
    value = os.environ.get('CODEX_EDIT_PROMPT_FILE', '')
    path = Path(value).expanduser() if value else Path(__file__).with_name('prompts') / 'robot_to_human_hand_arm.txt'
    if not path.is_file() or not path.read_text().strip():
        raise ValueError('Image-edit prompt file is missing or empty; check CODEX_EDIT_PROMPT_FILE.')
    return path.resolve()


def readiness(require_prompt=True) -> tuple[bool, str]:
    binary = shutil.which(os.environ.get('CODEX_BIN', 'codex'))
    if not binary:
        return False, 'Codex CLI is missing; install it and run codex login.'
    try:
        result = subprocess.run([binary, 'login', 'status'], capture_output=True, text=True, timeout=10)
        if result.returncode:
            return False, 'Codex is not logged in; run codex login.'
        if require_prompt:
            prompt_file()
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, str(error)
    return True, 'Codex CLI login and prompt configured; image-tool availability is verified during execution.'


def run_codex(workspace: Path, images: list[Path], prompt: str, schema: dict) -> dict:
    """Keep literal prompts, JSONL logs and final JSON per isolated edit workspace."""
    binary = shutil.which(os.environ.get('CODEX_BIN', 'codex'))
    if not binary:
        raise ValueError('Codex CLI is unavailable.')
    workspace.mkdir(parents=True, exist_ok=False)
    for index, source in enumerate(images):
        shutil.copyfile(source, workspace / f'input_{index}.png')
    (workspace / 'request.txt').write_text(prompt)
    (workspace / 'response.schema.json').write_text(json.dumps(schema))
    yolo = os.environ.get('CODEX_YOLO', '1').lower()
    if yolo not in ('1', '0', 'true', 'false'):
        raise ValueError('CODEX_YOLO must be 1/true or 0/false.')
    permissions = (['--dangerously-bypass-approvals-and-sandbox'] if yolo in ('1', 'true')
                   else ['--sandbox', 'workspace-write'])
    command = [binary, 'exec', *permissions, '--skip-git-repo-check',
               '--cd', str(workspace.resolve()), '--json', '--color', 'never',
               '--output-schema', str((workspace / 'response.schema.json').resolve()),
               '--output-last-message', str((workspace / 'response.json').resolve())]
    if os.environ.get('CODEX_MODEL'):
        command.extend(['--model', os.environ['CODEX_MODEL']])
    for index in range(len(images)):
        command.extend(['--image', str((workspace / f'input_{index}.png').resolve())])
    command.append('-')
    with (workspace / 'events.jsonl').open('w') as events, (workspace / 'stderr.log').open('w') as errors:
        result = subprocess.run(command, input=prompt, text=True, stdout=events, stderr=errors,
                                timeout=float(os.environ.get('CODEX_EDIT_TIMEOUT', '900')))
    if result.returncode:
        raise RuntimeError(f'Headless Codex failed ({result.returncode}); see {workspace}/stderr.log.')
    path = workspace / 'response.json'
    if not path.is_file():
        raise RuntimeError(f'Codex returned no structured result; see {workspace}/events.jsonl.')
    return json.loads(path.read_text())


def edit_image(image: Path, mask: Path, output: Path, capture_json: Path | None = None) -> dict:
    source_prompt = prompt_file()
    exact_prompt = source_prompt.read_text()
    guidance = None
    if capture_json:
        from .capture_guidance import load_guidance
        guidance = load_guidance(capture_json, image)
        (output.parent / 'capture_guidance.json').write_text(json.dumps(guidance, indent=2) + '\n')
    workspace = output.parent / f'codex-edit-{uuid.uuid4().hex[:12]}'
    schema = {'type': 'object', 'additionalProperties': False,
              'properties': {'status': {'type': 'string', 'enum': ['completed', 'unavailable']},
                             'image_path': {'type': 'string'}, 'detail': {'type': 'string'}},
              'required': ['status', 'image_path', 'detail']}
    if guidance:
        schema['properties']['orientation_check'] = {'type': 'string', 'enum': ['matched', 'mismatched', 'uncertain']}
        schema['required'].append('orientation_check')
    requested_image_model = 'gpt-image-2.5-sunburst'
    prompt = (
        'Perform image-to-image editing using the built-in image generation/editing tool. '
        'input_0.png is the photograph; input_1.png is its allowed edit mask (white editable). '
        'Use the photograph as the actual image-edit reference, not just a textual description. '
        'Save exactly one generated edited photograph as edited.png in this workspace. '
        'Keep dimensions and framing unchanged. Do not draw an approximation using code, '
        'copy the original as a substitute, call Gemini, use browser services, or access robot hardware. '
        'If the image tool cannot produce a file, return status unavailable with a truthful detail. '
        'A completed status means a genuine edited candidate was saved, not that exact landmark '
        'alignment was independently verified. Report alignment limitations in detail. The caller '
        'will restore protected pixels and may normalize a same-aspect-resolution output back to '
        'the requested dimensions, without cropping or changing framing. '
        'Return the JSON schema, with image_path relative to this workspace.\n\n'
        'BEGIN EXACT OPERATOR PROMPT\n' + exact_prompt + '\nEND EXACT OPERATOR PROMPT\n')
    prompt += ('\nIMAGE MODEL PREFERENCE: Use gpt-image-2.5-sunburst with quality=max if the image tool exposes model and quality selection. '                'Otherwise use the available GPT image-editing tool and explicitly report that the backend model cannot be selected or verified. '                'Do not confuse the Codex reasoning model with the image model; never claim a model ID without tool evidence.\n')
    if guidance:
        intrinsic = guidance['camera']['undistorted_intrinsics']
        prompt += ('\nCAMERA INTRINSICS FOR THIS EXACT CROP: ' + json.dumps(intrinsic) +
                   '\nThese are the effective intrinsic parameters of input_0.png, not the uncropped camera model.\n')
        prompt += ('\nFIXED HAND IDENTITY: This is always the human RIGHT hand and right forearm. '
                   'Do not mirror the image, swap fingers, or substitute a left hand. Use MANO joint 0 '
                   'as the shared wrist. wrist_R_right maps MANO-right wrist-local coordinates to '
                   'camera coordinates; preserve its palm orientation and thumb side.\n')
        prompt += ('\nCAPTURE GEOMETRY: The input photograph is undistorted. Preserve its original '
                   'framing and camera. Match the human right hand to the supplied MANO landmark '
                   'layout, including wrist and contact locations. Positions are already in the '
                   'camera optical frame, in metres. Project with the undistorted intrinsic matrix '
                   'only; never apply extrinsics again. Synthetic landmarks and wrist adjustment '
                   'are identified in the metadata.\n' + json.dumps(guidance, allow_nan=False))
        prompt += ('\nThe capture-stage pose is the authoritative ground-truth conditioning pose. '
                   'Do not estimate a replacement pose, change landmark positions, rescale the '
                   'camera model, or reinterpret the wrist conversion. Preserve these supplied '
                   'landmarks when replacing appearance. Any inability to satisfy this constraint '
                   'must be reported rather than modifying the reference pose.\n')
    references = [image, mask]
    if capture_json and (capture_json.parent / 'mano_overlay.png').is_file():
        references.append(capture_json.parent / 'mano_overlay.png')
        prompt += '\ninput_2.png visualizes the ground-truth MANO landmarks for alignment only; do not draw its lines or labels into the edited photograph.\n'
    if guidance:
        from .capture_guidance import overlay_on_edited
        reference = output.parent / 'alignment_reference.png'
        overlay_on_edited(image, guidance, reference, title='Captured MANO GT and forearm guide — one shared wrist')
        index = len(references)
        references.append(reference)
        prompt += f'\ninput_{index}.png contains the combined hand GT and robot arm centerline guides. Align hand size, wrist, fingers, and visible arm to these guides, then output a clean human hand-and-arm photograph.\n'
    if guidance:
        prompt += ('\nMANDATORY ORIENTATION CONSTRAINTS: Include these constraints in the image-edit '
                   'tool prompt, not only in your final response. Use input_0.png as the sole scene '
                   'photograph; the other images are alignment guides, never alternative first frames. '
                   'The BACK (dorsal surface) of the RIGHT hand faces the camera; the palm faces away. '
                   'Show dorsal knuckles and appropriate fingernails, not palm creases. '
                   'Use supplied projected joints to locate the hand; do not run a hand detector or re-estimate the pose. '
                   'The human thumb MUST occupy the supplied THUMB BASE/TIP side of the captured '
                   'skeleton, and the little finger MUST occupy the PINKY side. Match all named '
                   'orientation_anchors in capture_guidance; do not interpret screen-left as a left '
                   'hand or screen-right as a right hand. Preserve the captured 3D palm orientation, '
                   'finger depth ordering, and wrist_R_right. Do not turn the palm over to expose '
                   'the opposite surface, mirror just the hand, exchange thumb and pinky, or rotate '
                   'the hand to a more familiar pose. Preserve the camera, crop, and background.\n'
                   'Inspect the generated photograph alongside input_0.png and the labeled reference '
                   'before returning. Check RIGHT-hand anatomy, thumb/pinky correspondence, and '
                   'BACK-facing orientation and every projected joint against the supplied pose. If flipped or misaligned, correct the '
                   'edit using the ORIGINAL photograph and the same guides. Return orientation_check '
                   'matched only if those checks pass; otherwise return mismatched or uncertain '
                   'and explain why. This is an editor self-check, not an independent metric.\n')
    result = run_codex(workspace, references, prompt, schema)
    if guidance and result.get('orientation_check') not in ('matched', 'mismatched', 'uncertain'):
        raise RuntimeError('Image editor omitted its required orientation self-check.')
    if guidance and result.get('status') != 'completed':
        raise RuntimeError(f'Image edit unavailable: {result.get("detail", "unknown error")}')
    # A valid generated image with an uncertain self-check is a reviewable
    # candidate, not an execution failure. Admission to diffusion is separate.
    artifact = (workspace / result['image_path']).resolve()
    reference_names = {f'input_{index}.png' for index in range(len(references))}
    if not artifact.is_relative_to(workspace.resolve()) or artifact.name in reference_names:
        raise ValueError('Codex returned an invalid generated-image path.')
    if not artifact.is_file():
        raise RuntimeError(f'Codex image edit unavailable: {result["detail"]}; logs: {workspace}')
    with Image.open(image) as original:
        target_size = ImageOps.exif_transpose(original).size
    with Image.open(artifact) as edited:
        edited = edited.convert('RGB')
        generated_size = edited.size
        if generated_size != target_size:
            if abs(generated_size[0]*target_size[1] - generated_size[1]*target_size[0]) > 1e-6:
                raise ValueError('Generated image changed aspect ratio; refusing crop or stretch.')
            edited = edited.resize(target_size, Image.Resampling.LANCZOS)
        edited.save(output, format='PNG')
    return {'provider': 'codex_headless', 'prompt_file': str(source_prompt),
            'pose_source': 'capture_ground_truth' if guidance else None,
            'capture_guidance': str(output.parent / 'capture_guidance.json') if guidance else None,
            'generated_size_px': list(generated_size), 'output_size_px': list(target_size),
            'resolution_normalized': generated_size != target_size,
            'orientation_self_check': result.get('orientation_check'),
            'diffusion_eligible': not guidance or result.get('orientation_check') == 'matched',
            'review_reason': result.get('detail') if guidance and result.get('orientation_check') != 'matched' else None,
            'orientation_independently_verified': False,
            'alignment_verified': False, 'requires_review': bool(guidance) or result['status'] != 'completed',
            'model_detail': result['detail'],
            'requested_image_model': requested_image_model, 'requested_image_quality': 'max',
            'image_model_verified': False,
            'required_visible_surface': 'dorsal' if guidance else None,
            'prompt_sha256': hashlib.sha256(exact_prompt.encode()).hexdigest(),
            'codex_model': os.environ.get('CODEX_MODEL', 'Codex configured default'),
            'workspace': str(workspace), 'artifact': str(artifact)}


def propose_regions(image: Path, output: Path, instruction: str) -> dict:
    schema = {'type': 'object', 'additionalProperties': False, 'properties': {
        'hands': {'type': 'array', 'items': {'type': 'object', 'additionalProperties': False,
                   'properties': {'polygon': {'type': 'array', 'items': {'type': 'array',
                                  'items': {'type': 'number'}}}}, 'required': ['polygon']}}},
        'required': ['hands']}
    return run_codex(output.parent / f'codex-mask-{uuid.uuid4().hex[:12]}', [image],
                     instruction + '\nInspect input_0.png and return only the structured JSON. '
                     'Do not edit images, execute commands, or access hardware.', schema)
