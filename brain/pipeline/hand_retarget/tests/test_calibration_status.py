import copy
import numpy as np
import pytest
from hand_retarget import calibration_status as registry


def pair(folder, stamp, shift=0):
    base = np.eye(4); base[0, 3] = shift
    np.save(folder/f'{stamp}_base_pose_robot_ur5e.npy', base)
    np.save(folder/f'{stamp}_camera_extrinsic_138322253845.npy', np.eye(4))


def test_latest_complete_pair_updates_without_restart(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'DIRECTORY', tmp_path)
    pair(tmp_path, '20261024_095538')
    old = {'calibration': registry.status()['provenance']}
    registry.require_current(old)
    # An interrupted write must not mix transforms from different runs.
    np.save(tmp_path/'20261024_190421_base_pose_robot_ur5e.npy', np.eye(4))
    assert registry.status()['latest'] == '20261024_095538'
    pair(tmp_path, '20261024_190421', .2)
    assert registry.status(old)['current'] is False
    with pytest.raises(ValueError, match='Capture a new frame'):
        registry.require_current(old)
    matrix, provenance = registry._selection.load_calibration(tmp_path, 'ur5e', '138322253845')
    assert matrix[0, 3] == .2
    registry.require_current({'calibration': provenance})
    forged = copy.deepcopy(provenance)
    forged['world_from_camera']['sha256'] = 'wrong'
    assert not registry.status({'calibration': forged})['current']


def test_invalid_new_pair_does_not_silently_use_old(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, 'DIRECTORY', tmp_path)
    pair(tmp_path, '20261024_095538')
    pair(tmp_path, '20261024_190421')
    np.save(tmp_path/'20261024_190421_camera_extrinsic_138322253845.npy', np.zeros((4, 4)))
    assert not registry.status()['available']
    with pytest.raises(ValueError):
        registry.require_current({})
