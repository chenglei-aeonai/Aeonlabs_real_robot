"""Command line export: retarget a human demonstration and write the robot demonstration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .pipeline import Settings, export_robot_demonstration, retarget_demonstration
from .placement import load_placement
from .robot import TABLE_TOP_Z
from .tube import TubeAdjustment, place_tube
from .retargeters import METHODS
from .robot import RobotKinematics
from .skeleton import load_world_joints


def export_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="a *_world_joints.csv from the hand pose estimator")
    parser.add_argument("--method", choices=METHODS, default="vector")
    parser.add_argument("--scaling", choices=("global", "per_finger"), default=None)
    parser.add_argument("--filter", action="store_true", help="One-Euro filter the skeletons first")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    saved = load_placement(args.csv)  # the placement tuned in the viewer, if any
    demo = load_world_joints(args.csv)
    settings = Settings(method=args.method, scaling_mode=args.scaling, filter=args.filter, adjustment=saved.adjustment, grasp=saved.grasp)
    replay = retarget_demonstration(demo, RobotKinematics(), settings).cut(saved.first_frame, saved.last_frame)
    out = args.out or Path("out") / f"{demo.name}_{args.method}.npz"
    table_top = TABLE_TOP_Z + saved.table_offset
    tube = place_tube(replay, table_top, TubeAdjustment(*saved.tube_offset))
    export_robot_demonstration(replay, out, table_top, tube)
    errors = replay.stack("fingertip_error") * 1000
    print(f"{out}: {replay.demo.num_frames} frames, fingertip error mean {errors.mean():.1f} mm, per finger {np.round(errors.mean(0), 1)}")
