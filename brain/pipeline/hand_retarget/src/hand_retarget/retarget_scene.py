"""Loopback-only URDF hand, calibrated board, and estimated table-plane viewer."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading

import numpy as np
from PIL import Image

from .h3_actions import rigid_transforms
from .placement import apply

ROOT=Path(__file__).resolve().parents[5]
WORLD_FROM_BOARD=np.array([[0.,1,0,0],[1,0,0,0],[0,0,-1,0],[0,0,0,1]])


def scene_geometry(result):
    capture=json.loads((result/'capture.json').read_text())
    provenance=capture['calibration']['world_from_base']
    path=Path(provenance['path'])
    if not path.is_absolute():path=ROOT/path
    if hashlib.sha256(path.read_bytes()).hexdigest()!=provenance['sha256']:
        raise ValueError('Source calibration transform changed; cannot place the board reliably.')
    world_from_base=rigid_transforms(np.load(path,allow_pickle=False),1)[0]
    base_from_world=np.linalg.inv(world_from_base)
    base_from_board=base_from_world@WORLD_FROM_BOARD
    board=json.loads((ROOT/'calibration/boards/charuco_board.json').read_text())
    width=board['squares_x']*board['square_length_m'];height=board['squares_y']*board['square_length_m']
    center=apply(WORLD_FROM_BOARD,np.array([width/2,height/2,0.]))
    return {'coordinate_frame':'robot_base','units':'metres','robot_base_from_world':base_from_world.tolist(),
            'robot_base_from_board':base_from_board.tolist(),'board_size_m':[width,height],
            'table_plane':{'normal':world_from_base[2,:3].tolist(),'offset_m':float(world_from_base[2,3]),
                           'equation':'normal dot point_robot_base + offset_m = 0',
                           'reference':'Printed calibration-board surface at capture calibration time; bare-table thickness offset unknown.'},
            'table_display':{'width_m':1.,'height_m':.7,'center_world_m':center.tolist(),
                             'footprint_measured':False,'note':'Illustrative extent, not measured table boundaries.'},
            'calibration':provenance,
            'board_definition_sha256':hashlib.sha256((ROOT/'calibration/boards/charuco_board.json').read_bytes()).hexdigest(),
            'board_pose_note':'Board position at calibration time; a board moved afterwards is not tracked.'}


class RetargetScene:
    def __init__(self,result,port=8041):
        import viser
        import yourdfpy
        from viser.extras import ViserUrdf
        from .robot import write_hand_urdf,HAND_JOINT_NAMES
        from .skeleton import BONES
        self.result=Path(result);self.geometry=scene_geometry(self.result)
        with np.load(self.result/'retargeted_hand.npz',allow_pickle=False) as data:
            self.q=data['hand_q_rad'].copy();self.poses=data['palm_pose_robot_base'].copy()
            self.human=data['mano_joints_robot_base_m'].copy();self.fps=float(data['fps'])
            names=data['joint_names'].tolist()
        if names!=list(HAND_JOINT_NAMES):raise ValueError('Unexpected DG5F joint order.')
        self.done=threading.Event();self.lock=threading.RLock()
        self.directory=tempfile.TemporaryDirectory(prefix='dg5f-preview-')
        urdf=yourdfpy.URDF.load(write_hand_urdf(Path(self.directory.name)))
        self.server=viser.ViserServer(host='127.0.0.1',port=port,label='Calibrated DG5F preview',verbose=False)
        try:
            scene=self.server.scene;scene.set_up_direction('+z')
            scene.add_frame('/robot_base',axes_length=.12,axes_radius=.002)
            self.root=scene.add_frame('/dg5f',show_axes=False)
            self.urdf=ViserUrdf(self.server,urdf,root_node_name='/dg5f')
            self.order=[names.index(name) for name in self.urdf.get_actuated_joint_names()]
            self.bones=BONES
            self.skeleton=scene.add_line_segments('/generated_mano',self.human[0][np.asarray(BONES)],(40,155,210),thickness=.002)
            base_world=np.asarray(self.geometry['robot_base_from_world'])
            self.world=scene.add_frame('/calibration_world',wxyz=self.quaternion(base_world[:3,:3]),position=base_world[:3,3],show_axes=False)
            table=self.geometry['table_display'];center=np.asarray(table['center_world_m'])
            self.table=scene.add_box('/calibration_world/table',dimensions=(1.,.7,.015),position=center-[0,0,.0075],color=(130,140,125),opacity=.55)
            self.grid=scene.add_grid('/calibration_world/plane_grid',width=1.,height=.7,cell_size=.05,section_size=.25,position=center+[0,0,.0002],plane='xy')
            base_board=np.asarray(self.geometry['robot_base_from_board']);w,h=self.geometry['board_size_m']
            # Image local y is upward; board image coordinates have y downward.
            image_rotation=base_board[:3,:3]@np.diag([1.,-1.,-1.])
            self.board=scene.add_image('/calibration_board',np.asarray(Image.open(ROOT/'calibration/boards/charuco_board.png').convert('RGB')),
                render_width=w,render_height=h,wxyz=self.quaternion(image_rotation),position=apply(base_board,np.array([w/2,h/2,-.001])),format='png')
            scene.add_label('/board_label','Calibration board (saved pose)',position=apply(base_board,np.array([w/2,h+.02,0.])))
            gui=self.server.gui
            gui.add_markdown('**Robot-base coordinates · metres**\n\nURDF hand preview only. Table plane comes from calibration; its displayed edges are illustrative. The board shows its saved calibration pose.')
            self.play=gui.add_checkbox('Play trajectory',False)
            self.frame=gui.add_slider('Frame',min=0,max=len(self.q)-1,step=1,initial_value=0)
            self.mano=gui.add_checkbox('Show generated MANO',True)
            self.show_table=gui.add_checkbox('Show table / plane',True)
            self.show_board=gui.add_checkbox('Show calibration board',True)
            self.info=gui.add_markdown('')
            with gui.add_folder('Table display (not measured bounds)'):
                self.width=gui.add_slider('Width [m]',min=.2,max=2.,step=.01,initial_value=1.)
                self.height=gui.add_slider('Depth [m]',min=.2,max=2.,step=.01,initial_value=.7)
                self.dx=gui.add_slider('World X offset [m]',min=-1.,max=1.,step=.01,initial_value=0.)
                self.dy=gui.add_slider('World Y offset [m]',min=-1.,max=1.,step=.01,initial_value=0.)
                self.thickness=gui.add_slider('Board above table [mm]',min=0.,max=30.,step=1.,initial_value=0.)
            self.frame.on_update(lambda _:self.render())
            self.mano.on_update(lambda _:setattr(self.skeleton,'visible',self.mano.value))
            self.show_board.on_update(lambda _:setattr(self.board,'visible',self.show_board.value))
            for control in (self.width,self.height,self.dx,self.dy,self.thickness,self.show_table):control.on_update(lambda _:self.update_table())
            def reset_camera(client):
                target=(self.poses[self.frame.value,:3,3]+apply(base_world,center))/2
                with client.atomic():
                    # Setting position also translates look_at in Viser, so set the
                    # desired target last, after moving the virtual camera.
                    client.camera.position=target+np.array([.65,-.8,.65])
                    client.camera.up_direction=np.array([0.,0.,1.])
                    client.camera.look_at=target
            self.server.on_client_connect(reset_camera)
            reset=gui.add_button('Reset view')
            reset.on_click(lambda event:reset_camera(event.client) if event.client is not None else None)
            (self.result/'scene_geometry.json').write_text(json.dumps(self.geometry,indent=2)+'\n')
            self.render()
            self.worker=threading.Thread(target=self.animate,daemon=True);self.worker.start()
        except Exception:
            self.server.stop();self.directory.cleanup();raise

    @staticmethod
    def quaternion(rotation):
        import pinocchio as pin
        x,y,z,w=pin.Quaternion(rotation).coeffs();return np.array([w,x,y,z])

    def render(self):
        with self.lock,self.server.atomic():
            index=self.frame.value;pose=self.poses[index]
            self.root.position=pose[:3,3];self.root.wxyz=self.quaternion(pose[:3,:3])
            self.urdf.update_cfg(self.q[index,self.order])
            self.skeleton.points=self.human[index][np.asarray(self.bones)]
            self.info.content=f'Frame **{index+1}/{len(self.q)}** · {index/self.fps:.2f} s'

    def update_table(self):
        with self.lock:
            center=np.asarray(self.geometry['table_display']['center_world_m'])+np.array([self.dx.value,self.dy.value,-self.thickness.value/1000])
            self.table.dimensions=(self.width.value,self.height.value,.015)
            self.table.position=center-[0,0,.0075];self.grid.position=center+[0,0,.0002]
            self.grid.width=self.width.value;self.grid.height=self.height.value
            self.table.visible=self.grid.visible=self.show_table.value

    def animate(self):
        while not self.done.wait(1/self.fps):
            if self.play.value:self.frame.value=(self.frame.value+1)%len(self.q)

    def close(self):
        self.done.set()
        if hasattr(self,'worker'):self.worker.join(timeout=2)
        self.server.stop();self.directory.cleanup()
