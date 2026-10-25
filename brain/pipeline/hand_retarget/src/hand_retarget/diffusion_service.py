"""Persist local connection settings; remote aeon-webui owns checkpoint loading."""
import json
import os
from pathlib import Path
import uuid

from .aeon_client import AeonClient, service_url


class DiffusionService:
    def __init__(self, output_root):
        self.path=output_root/'diffusion_config.json'
        self.snapshot_path=Path(__file__).resolve().parents[3]/'configs/aeon-checkpoints.json'
        self.token=str(uuid.uuid4())

    def config(self):
        if self.path.is_file():return json.loads(self.path.read_text())
        url=os.environ.get('AEON_WEBUI_URL','')
        value={'url':service_url(url) if url else '', 'token':self.token}
        if url:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(self.path)
        return value

    def configure(self,url):
        old=self.config();value={'url':service_url(url),'token':old['token'] if old['url']==url else str(uuid.uuid4())}
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(self.path)
        return {'url':value['url']}

    def client(self):
        config=self.config()
        if not config['url']:raise ValueError('Configure the aeon-webui URL or SSH tunnel in the Diffusion panel.')
        return AeonClient(config['url'],config['token'])

    def status(self):
        config=self.config()
        fallback=json.loads(self.snapshot_path.read_text())['profiles']
        if not config['url']:return {'connected':False,'state':'not_configured','url':'','profiles':fallback}
        client=None
        try:
            client=self.client();value=client.status();value.update(connected=True,url=config['url'])
            return value
        except Exception as error:
            return {'connected':False,'state':'unreachable','url':config['url'],'profiles':fallback,'error':str(error)}
        finally:
            if client:client.close()

    def start(self,profile,partition,sol_enabled):
        client=self.client()
        try:
            status=client.status();capability=status.get('capture_gt_bridge')
            if not capability or not capability.get('capture_gt_v1'):
                raise ValueError('Install the remote capture-GT bridge before loading a checkpoint for this workflow.')
            if profile not in capability.get('profiles',[]):raise ValueError('Profile does not support captured RGB/pose frames.')
            return client.start(profile,partition,sol_enabled)
        finally:client.close()

    def stop(self):
        client=self.client()
        try:return client.request('POST','/api/hands/stop')
        finally:client.close()
