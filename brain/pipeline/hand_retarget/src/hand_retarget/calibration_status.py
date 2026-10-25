"""Live calibration discovery; historical captures keep their original transforms."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
_spec = importlib.util.spec_from_file_location('capture_calibration_selection', ROOT / 'brain/capture/calibration_selection.py')
_selection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_selection)
DIRECTORY = ROOT / 'calibration/results/original_legacy'


def timestamp(provenance):
    name = Path(provenance.get('world_from_base', {}).get('path', '')).name
    return name.split('_base_pose_robot_')[0] if '_base_pose_robot_' in name else 'unknown'


def status(capture=None):
    try:
        _, provenance = _selection.load_calibration(DIRECTORY, 'ur5e', '138322253845')
        result = {'available': True, 'latest': timestamp(provenance), 'provenance': provenance}
    except (OSError, ValueError) as error:
        result = {'available': False, 'error': str(error)}
    if capture is not None:
        previous = capture.get('calibration', {})
        result['capture_calibration'] = timestamp(previous)
        result['current'] = result['available'] and all(
            previous.get(key, {}).get('sha256') == result['provenance'][key]['sha256']
            and Path(previous.get(key, {}).get('path', '')).name == Path(result['provenance'][key]['path']).name
            for key in ('world_from_base', 'world_from_camera'))
    return result


def require_current(capture):
    state = status(capture)
    if not state.get('current'):
        raise ValueError('Capture calibration is outdated or unavailable. Latest calibration: '
                         + state.get('latest', 'unavailable')
                         + '. Capture a new frame and regenerate its downstream results. Historical results retain their original calibration.')
