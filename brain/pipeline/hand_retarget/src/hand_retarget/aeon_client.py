"""HTTP contracts read from aeon-webui master a8b2ada; no local GPU loading."""
import base64
from pathlib import Path
import time
from urllib.parse import urlsplit

import httpx


def service_url(value):
    parsed = urlsplit(value.strip())
    if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/')):
        raise ValueError('Use an http(s) aeon-webui root URL, e.g. http://127.0.0.1:8017.')
    return value.strip().rstrip('/')


class AeonClient:
    def __init__(self, url, token, client=None):
        self.url = service_url(url)
        self.token = token
        self.client = client or httpx.Client(timeout=120, follow_redirects=False)
        self.prefix = ''
        self.last_claim = 0.
        self.multitab = None

    def close(self):
        self.client.close()

    def session(self):
        if self.multitab is None:
            response = self.client.get(self.url + '/api/tabs/config')
            self.multitab = response.status_code == 200 and response.json().get('enabled', False)
        if self.multitab and (not self.prefix or time.monotonic() - self.last_claim > 20):
            response = self.client.post(self.url + '/api/tabs', json={
                'token': self.token, 'document_id': 'brain-pipeline-' + self.token})
            response.raise_for_status()
            self.prefix = response.json()['prefix']
            if self.prefix != '/tabs/'+self.token:
                raise ValueError('Unexpected inference tab-session prefix.')
            self.last_claim = time.monotonic()

    def request(self, method, route, **kwargs):
        self.session()
        response = self.client.request(method, self.url + self.prefix + route,
                                       headers={'X-Retarget-Client': '1'}, **kwargs)
        if response.is_error:
            if response.status_code == 404 and route == '/api/hands/status':
                raise RuntimeError(
                    f'{self.url} is reachable, but it does not provide the aeon-webui hands API. '
                    'Check the SSH tunnel destination port and start the robot-GT branch on that port; '
                    'the tunnel may be reaching a different video service.')
            try: detail = response.json().get('detail', response.text)
            except ValueError: detail = response.text
            raise RuntimeError(f'aeon-webui {response.status_code}: {str(detail)[:1500]}')
        return response.json()

    def status(self):
        status = self.request('GET', '/api/hands/status')
        try:
            status['capture_gt_bridge'] = self.request('GET', '/api/robot/gt-capabilities')
        except RuntimeError:
            status['capture_gt_bridge'] = None
        if (status.get('capture_gt_bridge') or {}).get('startup_status_v1'):
            status.update(self.request('GET', '/api/robot/gt-startup'))
        return status

    def start(self, profile, partition='normal', sol_enabled=True):
        status = self.status()
        if profile not in {p['id'] for p in status.get('profiles', [])}:
            raise ValueError('Checkpoint profile is not advertised by aeon-webui.')
        return self.request('POST', '/api/hands/start', json={
            'profile': profile, 'partition': partition, 'sol_enabled': sol_enabled, 'servers': 1})

    def register(self, bundle: Path, assume_zero_contact=False):
        return self.request('POST', '/api/robot/gt-frames', json={
            'bundle_b64': base64.b64encode(bundle.read_bytes()).decode(),
            'assume_zero_contact': assume_zero_contact})

    def generate(self, clip, instruction, seed, sol_enabled):
        return self.request('POST', '/api/hands/generate', json={
            'prompt': instruction, 'clip_id': clip['clip_id'], 'frame_dataset': clip['dataset'],
            'seed': seed, 'steps': 50, 'cfg': 5., 'match_checkpoint_recipe': True,
            'sol_enabled': sol_enabled, 'candidates': 1, 'continuous_rollouts': 0})

    def download(self, path, target: Path):
        # Gallery responses are same-origin paths; never fetch an arbitrary URL
        # returned in remote job metadata.
        if not isinstance(path, str) or not path.startswith('/gallery/') or '..' in path:
            raise ValueError('Invalid aeon-webui gallery artifact path.')
        with self.client.stream('GET', self.url + path) as response:
            response.raise_for_status()
            part = target.with_suffix(target.suffix + '.part')
            with part.open('wb') as stream:
                for chunk in response.iter_bytes():stream.write(chunk)
            part.replace(target)
