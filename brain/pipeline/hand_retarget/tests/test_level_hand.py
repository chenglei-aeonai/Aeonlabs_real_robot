"""Offline geometry regression for leveling without lowering the wrist."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    'level_hand', Path(__file__).resolve().parents[2] / 'scripts/level_hand.py')
level = importlib.util.module_from_spec(spec)
spec.loader.exec_module(level)


def start_angles():
    return np.array([-1.803715531, -1.085797237, 1.458274666,
                     -.706991271, 1.177627683, -2.006018464])


def test_level_preserves_wrist_xy_lifts_and_finishes_horizontal():
    model = level.RobotSurfaceModel()
    q = start_angles()
    h = np.zeros(20)
    initial = model.transforms(q, h)['tool0']
    qs, times, curve, _ = level.plan(model, q, h, np.array([0., 0., 1.]), 1.)
    np.testing.assert_allclose(qs[0], q, atol=1e-12)
    for at in np.linspace(0, times[-1], 301):
        pose = model.transforms(curve(at), h)['tool0']
        assert pose[2, 3] >= initial[2, 3] - 1e-5
        assert np.linalg.norm(pose[:2, 3] - initial[:2, 3]) < .002
    final = model.transforms(qs[-1], h)
    np.testing.assert_allclose(final['rl_dg_palm'][:3, 0], [0, 0, -1], atol=1e-5)
    assert final['tool0'][2, 3] - initial[2, 3] == pytest.approx(.01, abs=1e-5)


def test_level_rejects_large_hand_flip():
    q = start_angles()
    q[-1] += np.pi
    with pytest.raises(RuntimeError, match='exceeds 60 degrees'):
        level.plan(level.RobotSurfaceModel(), q, np.zeros(20), np.array([0., 0., 1.]), 1.)


def test_level_requires_unchanged_validated_plane(tmp_path):
    import json
    path = tmp_path / 'robot_rollout.npz'
    np.savez(path, plane_normal=[0., 0., 1.], plane_offset=.2)
    report = {'execution_ready':True, 'rollout_sha256':level.digest(path),
              'collision_model_sha256':'expected-model'}
    (tmp_path/'manifest.json').write_text(json.dumps(report))
    normal, offset = level.checked_plane(tmp_path, 'expected-model')
    np.testing.assert_array_equal(normal, [0., 0., 1.])
    assert offset == .2
    with pytest.raises(ValueError, match='validated'):
        level.checked_plane(tmp_path, 'other-model')
    np.savez(path, plane_normal=[0., 0., 1.], plane_offset=100.)
    with pytest.raises(ValueError, match='validated'):
        level.checked_plane(tmp_path, 'expected-model')
