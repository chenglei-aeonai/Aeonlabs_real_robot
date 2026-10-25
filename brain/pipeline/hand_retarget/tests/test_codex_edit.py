import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest

from hand_retarget import codex_edit
from hand_retarget.inference_clients import replace_hands


def test_default_hand_arm_prompt_and_invalid_override(monkeypatch):
    monkeypatch.delenv('CODEX_EDIT_PROMPT_FILE', raising=False)
    prompt = codex_edit.prompt_file().read_text()
    assert 'human right hand and arm' in prompt
    assert 'camera intrinsics' in prompt
    monkeypatch.setenv('CODEX_EDIT_PROMPT_FILE', '/missing/operator/prompt.txt')
    with pytest.raises(ValueError, match='missing or empty'):
        codex_edit.prompt_file()


def test_headless_edit_keeps_exact_prompt_and_protects_pixels(tmp_path, monkeypatch):
    prompt = tmp_path / 'prompt_d.txt'
    prompt.write_text('Exact operator instructions with literal $(do_not_execute).')
    monkeypatch.setenv('CODEX_EDIT_PROMPT_FILE', str(prompt))
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    original, mask, output = [tmp_path / name for name in ('input.png', 'mask.png', 'edited.png')]
    Image.new('RGB', (64, 48), 'blue').save(original)
    allowed = np.zeros((48, 64), np.uint8); allowed[10:30, 10:30] = 255
    Image.fromarray(allowed).save(mask)

    def generate(workspace, images, instruction, schema):
        assert prompt.read_text() in instruction
        assert images == [original, mask]
        workspace.mkdir()
        Image.new('RGB', (64, 48), 'red').save(workspace / 'edited.png')
        return {'status': 'completed', 'image_path': 'edited.png', 'detail': 'test generator'}

    monkeypatch.setattr(codex_edit, 'run_codex', generate)
    report = replace_hands(original, mask, output, model='', provider='codex')
    assert report['provider'] == 'codex_headless'
    assert report['prompt_file'] == str(prompt)
    pixels = np.asarray(Image.open(output))
    assert np.all(pixels[allowed == 0] == [0, 0, 255])
    assert np.all(pixels[allowed == 255] == [255, 0, 0])


@pytest.mark.parametrize('status,path', [('unavailable', ''), ('completed', '../external.png'),
                                         ('completed', 'input_0.png')])
def test_no_fake_success_or_untrusted_artifact(tmp_path, monkeypatch, status, path):
    prompt = tmp_path / 'prompt.txt'; prompt.write_text('Operator prompt')
    monkeypatch.setenv('CODEX_EDIT_PROMPT_FILE', str(prompt))
    monkeypatch.setattr(codex_edit, 'run_codex', lambda *args: {
        'status': status, 'image_path': path, 'detail': 'Tool unavailable'})
    with pytest.raises((RuntimeError, ValueError)):
        codex_edit.edit_image(tmp_path / 'input.png', tmp_path / 'mask.png', tmp_path / 'output.png')


@pytest.mark.parametrize('yolo', ['1', '0'])
def test_cli_uses_stdin_and_argument_array_without_shell(tmp_path, monkeypatch, yolo):
    source = tmp_path / 'source.png'; Image.new('RGB', (64, 48)).save(source)
    monkeypatch.setattr(codex_edit.shutil, 'which', lambda name: '/bin/codex-test')
    workspace = tmp_path / 'workspace'
    monkeypatch.setenv('CODEX_YOLO', yolo)

    def run(command, **kwargs):
        assert command[1] == 'exec' and command[-1] == '-'
        if yolo == '1':
            assert '--dangerously-bypass-approvals-and-sandbox' in command
            assert '--sandbox' not in command
        else:
            assert command[command.index('--sandbox') + 1] == 'workspace-write'
            assert '--dangerously-bypass-approvals-and-sandbox' not in command
        assert '--image' in command
        assert kwargs['input'] == 'literal $(not_a_command)'
        assert not kwargs.get('shell', False)
        (workspace / 'response.json').write_text('{"value":true}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(codex_edit.subprocess, 'run', run)
    assert codex_edit.run_codex(workspace, [source], 'literal $(not_a_command)', {}) == {'value': True}
    assert (workspace / 'request.txt').read_text() == 'literal $(not_a_command)'


def test_web_status_and_admission_use_codex_instead_of_gemini(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from hand_retarget.webapp import Config, create_app
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setattr(codex_edit, 'readiness', lambda: (False, 'Exact prompt D is missing'))
    with TestClient(create_app(Config(tmp_path / 'runs'))) as client:
        status = client.get('/api/status').json()
        assert status['image_provider'] == 'codex'
        assert not status['image_ready']
        assert 'Exact prompt D is missing' in status['errors']
        assert status['codex_required']


@pytest.mark.parametrize('generated_size,accept', [((128, 96), True), ((128, 100), False)])
def test_generated_resolution_is_normalized_only_when_framing_aspect_matches(tmp_path, monkeypatch, generated_size, accept):
    source = tmp_path / 'source.png'; Image.new('RGB', (64, 48), 'blue').save(source)
    mask = tmp_path / 'mask.png'; Image.new('L', (64, 48), 255).save(mask)
    monkeypatch.delenv('CODEX_EDIT_PROMPT_FILE', raising=False)
    def generate(workspace, images, prompt, schema):
        workspace.mkdir()
        Image.new('RGB', generated_size, 'red').save(workspace / 'edited.png')
        return {'status': 'unavailable', 'image_path': 'edited.png', 'detail': 'Candidate needs landmark review.'}
    monkeypatch.setattr(codex_edit, 'run_codex', generate)
    output = tmp_path / 'edited.png'
    if not accept:
        with pytest.raises(ValueError, match='aspect ratio'):
            codex_edit.edit_image(source, mask, output)
    else:
        result = codex_edit.edit_image(source, mask, output)
        assert Image.open(output).size == (64, 48)
        assert result['resolution_normalized']
        assert result['requires_review']
        assert result['alignment_verified'] is False
