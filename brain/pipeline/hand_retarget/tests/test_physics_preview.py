import numpy as np
from scipy.spatial.transform import Rotation
from hand_retarget.physics_preview import table_pose, transform_points


def test_calibrated_table_remains_on_the_same_plane_in_simulator_world():
    world_from_base=np.eye(4)
    world_from_base[:3,:3]=Rotation.from_euler('xyz',[.2,-.1,2.]).as_matrix()
    world_from_base[:3,3]=[.1,.6,.55]
    base_from_table=np.eye(4)
    base_from_table[:3,:3]=Rotation.from_euler('xyz',[.05,-.08,.3]).as_matrix()
    base_from_table[:3,3]=[.2,-.4,-.15]
    geometry={'robot_base_from_world':base_from_table.tolist(),
              'table_display':{'center_world_m':[.12,.18,0]}}
    table,center=table_pose(world_from_base,geometry)
    np.testing.assert_allclose(center,transform_points(world_from_base,transform_points(base_from_table,[.12,.18,0])))
    for point in ([0,0,0],[.2,.3,0],[-.4,.1,0]):
        moved=transform_points(table,point)
        assert abs((moved-center)@table[:3,2]) < 1e-12
    # A point above the table retains its signed distance through both transforms.
    above=transform_points(table,[.12,.18,.07])
    assert abs((above-center)@table[:3,2]-.07) < 1e-12
