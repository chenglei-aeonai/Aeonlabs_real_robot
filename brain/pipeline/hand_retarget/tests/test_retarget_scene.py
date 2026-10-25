import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from hand_retarget import retarget_scene
from hand_retarget.placement import apply
from hand_retarget.webapp import Config,create_app


def test_board_corners_and_table_share_calibrated_plane(tmp_path,monkeypatch):
    root=tmp_path/'root';boards=root/'calibration/boards';boards.mkdir(parents=True)
    (boards/'charuco_board.json').write_text(json.dumps({'squares_x':3,'squares_y':5,'square_length_m':.058}))
    monkeypatch.setattr(retarget_scene,'ROOT',root)
    transform=np.array([[0.,-1,0,.2],[1,0,0,-.1],[0,0,1,.04],[0,0,0,1]])
    path=tmp_path/'world_from_base.npy';np.save(path,transform)
    result=tmp_path/'result';result.mkdir()
    (result/'capture.json').write_text(json.dumps({'calibration':{'world_from_base':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}))
    geometry=retarget_scene.scene_geometry(result)
    corners=apply(np.asarray(geometry['robot_base_from_board']),np.array([[0,0,0],[.174,0,0],[0,.29,0],[.174,.29,0]]))
    plane=geometry['table_plane']
    np.testing.assert_allclose(corners@plane['normal']+plane['offset_m'],0,atol=1e-12)
    np.testing.assert_allclose(geometry['robot_base_from_world'],np.linalg.inv(transform))
    assert geometry['board_size_m']==pytest.approx([.174,.29])
    assert geometry['table_display']['footprint_measured'] is False
    # The displayed texture faces up, while the OpenCV board normal faces down.
    texture_rotation=np.asarray(geometry['robot_base_from_board'])[:3,:3]@np.diag([1,-1,-1])
    np.testing.assert_allclose(texture_rotation[:,2],plane['normal'],atol=1e-12)
    np.save(path,np.eye(4))
    with pytest.raises(ValueError,match='changed'):retarget_scene.scene_geometry(result)


def test_scene_api_reuses_job_scene_and_closes_on_shutdown(tmp_path,monkeypatch):
    app=create_app(Config(tmp_path/'runs'));identifier=uuid.uuid4().hex
    job=tmp_path/'runs'/identifier;(job/'result').mkdir(parents=True)
    (job/'status.json').write_text(json.dumps({'id':identifier,'mode':'generated_hand_retarget','status':'completed'}))
    opened=[];closed=[]
    class Scene:
        def __init__(self,result):
            self.result=result;self.server=SimpleNamespace(get_port=lambda:8041);self.geometry={'coordinate_frame':'robot_base'};opened.append(result)
        def close(self):closed.append(self.result)
    monkeypatch.setattr(retarget_scene,'RetargetScene',Scene)
    with TestClient(app) as client:
        route=f'/api/retarget/{identifier}/scene'
        assert client.post(route).status_code==403
        for _ in range(2):
            result=client.post(route,headers={'X-Retarget-Client':'1'})
            assert result.status_code==200 and result.json()['url']=='http://127.0.0.1:8041'
        assert len(opened)==1
    assert len(closed)==1
