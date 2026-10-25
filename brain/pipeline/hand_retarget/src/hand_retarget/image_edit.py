"""Run only image masking/editing; no WiLoR, H3, or robot connections."""
import argparse
import json
import os
from pathlib import Path
import shutil

from .codex_edit import prompt_file
from .inference_clients import replace_hands
from .segmentation import segment_hands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--capture-json', type=Path, help='Paired camera intrinsics and camera-space MANO capture.')
    parser.add_argument('--prompt-file', type=Path, help='Optional prompt override; default: captured-skeleton hand-and-arm conversion.')
    parser.add_argument('--out', type=Path, required=True, help='New output directory.')
    args = parser.parse_args()
    if args.prompt_file:
        os.environ['CODEX_EDIT_PROMPT_FILE'] = str(args.prompt_file.resolve())
    prompt_file()  # Validate the saved default or explicit operator override.
    args.image = args.image.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'running', 'scope': 'image_edit_only', 'image_provider': 'codex',
                'input': str(args.image), 'hardware_executed': False}
    def progress(stage):
        (args.out / 'progress.json').write_text(json.dumps({'stage': stage}))
    try:
        if args.capture_json:
            from .capture_guidance import load_guidance
            load_guidance(args.capture_json, args.image)
            manifest['pose_source'] = 'capture_ground_truth'
            shutil.copyfile(args.capture_json, args.out / 'capture.json')
            for source, target in [('mano_overlay.png', 'mano_overlay.png'), ('overlay.png', 'robot_overlay.png'),
                                   ('robot_overlay.png', 'robot_overlay.png')]:
                path = args.capture_json.parent / source
                if path.is_file():
                    shutil.copyfile(path, args.out / target)
        mask = args.out / 'edit_mask.png'
        if args.mask:
            shutil.copyfile(args.mask, mask)
        elif args.capture_json:
            from .capture_mask import projected_capture_mask
            progress('mask')
            manifest['segmentation'] = projected_capture_mask(args.image, args.capture_json, mask)
        else:
            progress('mask')
            manifest['segmentation'] = segment_hands(args.image, mask, provider='codex')
        progress('edit')
        manifest['edit'] = replace_hands(args.image, mask, args.out / 'edited.png',
                                         model='', provider='codex', capture_json=args.capture_json)
        manifest['status'] = 'completed'
        progress('completed')
    except Exception as error:
        manifest.update(status='failed', error=f'{type(error).__name__}: {error}')
        progress('failed')
        raise
    finally:
        (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Saved {args.out}/edited.png; no downstream inference or robot commands.')


if __name__ == '__main__':
    main()
