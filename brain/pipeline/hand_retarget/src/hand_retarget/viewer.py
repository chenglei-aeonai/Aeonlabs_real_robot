"""Browser viewer: the robot replaying a human demonstration next to the skeleton it imitates.

The robot and its table are at the world origin, as in simtoolreal; the skeleton is
drawn on a second, identical table beside it, placed the same way relative to its table.
"""

from __future__ import annotations

import argparse
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
import pinocchio as pin
import trimesh
import viser
import yourdfpy
from viser.extras import ViserUrdf

from .collisions import FINGER_NAMES
from .grasp import GRASP_JOINTS, GraspStrengthening
from .pipeline import Replay, Settings, export_robot_demonstration, retarget_demonstration, retarget_hands
from .placement import PlacementAdjustment, SavedPlacement, load_placement, save_placement
from .tube import TubeAdjustment, place_tube
from .retargeters import METHODS, OPTIMIZERS
from .robot import (
    ARM_JOINT_NAMES,
    ASSETS_DIR,
    HAND_JOINT_NAMES,
    ROBOT_BASE_POSITION,
    ROBOT_URDF,
    TABLE_CENTER_XY,
    TABLE_SIZE,
    TABLE_TOP_Z,
    RobotKinematics,
    write_hand_urdf,
)
from .skeleton import BONES, FINGERTIPS, load_world_joints

SKELETON_TABLE_OFFSET = np.array([1.0, 0.0, 0.0])
FINGER_COLORS = np.array([[230, 90, 60], [240, 180, 40], [80, 190, 90], [60, 150, 230], [170, 90, 220]])
SKELETON_COLOR = (255, 120, 40)
TABLE_COLOR = (209, 143, 89)
TUBE_COLOR = (60, 190, 80, 255)
EXPORT_DIR = Path(__file__).resolve().parents[2] / "out"


def _rotation_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wxyz(rotation: np.ndarray) -> np.ndarray:
    x, y, z, w = pin.Quaternion(rotation).coeffs()
    return np.array([w, x, y, z])


class ReplayViewer:
    def __init__(self, csv_path: Path, port: int):
        self.csv_path = csv_path
        self.saved = load_placement(csv_path)
        self.demo = load_world_joints(csv_path)
        self.robot = RobotKinematics()
        self.server = viser.ViserServer(port=port, label="hand retarget")
        self.lock = threading.Lock()
        self.replay: Replay | None = None
        # The fingers take seconds, the placement and arm a tenth of that: finger results
        # are kept per hand setting, and a worker thread redoes only what changed, always
        # for the latest request, dropping the ones a dragged slider made in between.
        self._hands: dict[tuple, list] = {}
        self._wanted = threading.Event()
        self._build_scene()
        self._build_gui()
        threading.Thread(target=self._worker, daemon=True).start()
        self.recompute()

    # -- scene -----------------------------------------------------------------
    def _build_scene(self) -> None:
        scene = self.server.scene
        scene.set_up_direction("+z")
        self.tables = [
            (scene.add_box(name, color=TABLE_COLOR, dimensions=tuple(TABLE_SIZE)), offset)
            for name, offset in (("/robot_table", np.zeros(3)), ("/skeleton_table", SKELETON_TABLE_OFFSET))
        ]
        self.robot_root = scene.add_frame("/robot", show_axes=False, position=ROBOT_BASE_POSITION)
        self.robot_urdf = ViserUrdf(
            self.server,
            # Mesh paths in the URDF are relative to the assets directory.
            yourdfpy.URDF.load(ROBOT_URDF, filename_handler=partial(yourdfpy.filename_handler_relative, dir=ASSETS_DIR)),
            root_node_name="/robot",
        )
        self._urdf_order = [
            (ARM_JOINT_NAMES + HAND_JOINT_NAMES).index(name) for name in self.robot_urdf.get_actuated_joint_names()
        ]
        self.floating_root = scene.add_frame("/floating_hand", show_axes=False, visible=False)
        self.floating_urdf = ViserUrdf(self.server, yourdfpy.URDF.load(write_hand_urdf()), root_node_name="/floating_hand")
        self._floating_order = [HAND_JOINT_NAMES.index(n) for n in self.floating_urdf.get_actuated_joint_names()]

    def _draw_skeleton(self, name: str, skeleton: np.ndarray, visible: bool = True) -> None:
        bones = np.stack([skeleton[[a, b]] for a, b in BONES])
        self.server.scene.add_line_segments(
            f"{name}/bones", bones, SKELETON_COLOR, thickness=0.004, visible=visible
        )
        self.server.scene.add_point_cloud(
            f"{name}/joints", skeleton, SKELETON_COLOR, point_size=0.005, point_shape="circle", visible=visible
        )
        self.server.scene.add_point_cloud(
            f"{name}/tips", skeleton[list(FINGERTIPS)], FINGER_COLORS, point_size=0.008,
            point_shape="circle", visible=visible,
        )

    # -- gui -------------------------------------------------------------------
    def _build_gui(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Playback"):
            self.frame = gui.add_slider("Frame", 0, self.demo.num_frames - 1, 1, 0)
            self.playing = gui.add_checkbox("Play", True)
            self.speed = gui.add_slider("Speed", 0.1, 2.0, 0.1, 1.0)
            last = self.demo.num_frames - 1
            hint = "Playback and the exported robot demonstration keep only the frames from start to end"
            self.first_frame = gui.add_slider("Start frame", 0, last, 1, 0, hint=hint)
            self.last_frame = gui.add_slider("End frame", 0, last, 1, last, hint=hint)
        with gui.add_folder("Retargeting"):
            self.status = gui.add_markdown("")
            self.method = gui.add_dropdown("Method", METHODS, initial_value="vector")
            self.optimizer = gui.add_dropdown(
                "Optimizer", OPTIMIZERS, initial_value="own",
                hint="own: phalanx directions and finger collisions; dex: dex-retargeting, fingertips only",
            )
            self.scaling = gui.add_dropdown("Hand scaling", ("method default", "global", "per_finger"))
            self.filter = gui.add_checkbox("One-Euro filter", False)
            self.floating = gui.add_checkbox("Floating hand (no arm)", False)
        with gui.add_folder("Strengthen grasp"):
            self.grasp_sliders = [
                gui.add_slider(
                    f"{finger.capitalize()} (deg)", 0.0, 60.0, 1.0, 0.0,
                    hint=f"Extra closure of {joint}, the joint that closes the {finger} at its base",
                )
                for finger, joint in zip(FINGER_NAMES, GRASP_JOINTS)
            ]
            hint = "The extra closure is complete from the start frame to the end frame"
            self.grasp_first = gui.add_slider("Start frame", 0, last, 1, 0, hint=hint)
            self.grasp_last = gui.add_slider("End frame", 0, last, 1, last, hint=hint)
            self.grasp_ramp = gui.add_slider(
                "Ramp (frames)", 0, 48, 1, 12,
                hint="The extra closure grows over this many frames before the start frame, and goes away over as many after the end frame",
            )
        with gui.add_folder("View"):
            self.overlay = gui.add_checkbox("Skeleton over robot", False)
            self.show_tube = gui.add_checkbox("Show tube", True)
            self.table_height = gui.add_slider(
                "Table up/down (cm)", -20.0, 20.0, 0.5, 0.0,
                hint="From simtoolreal's table, 3.5 cm below the robot base",
            )
        with gui.add_folder("Placement"):
            self.placement_sliders = {
                "x": gui.add_slider("Right/left x (cm)", -30.0, 30.0, 0.5, 0.0),
                "y": gui.add_slider("Front/back y (cm)", -30.0, 30.0, 0.5, 0.0),
                "z": gui.add_slider("Up/down z (cm)", -20.0, 30.0, 0.5, 0.0),
                "yaw": gui.add_slider("Yaw (deg)", -180.0, 180.0, 1.0, 0.0),
                "pitch": gui.add_slider("Pitch (deg)", -45.0, 45.0, 1.0, 0.0),
                "roll": gui.add_slider("Roll (deg)", -45.0, 45.0, 1.0, 0.0),
            }
            self.tube_sliders = {
                "x": gui.add_slider("Tube right/left x (cm)", -15.0, 15.0, 0.5, 0.0),
                "y": gui.add_slider("Tube front/back y (cm)", -15.0, 15.0, 0.5, 0.0),
                "yaw": gui.add_slider("Tube yaw (deg)", -90.0, 90.0, 1.0, 0.0),
            }
            reset = gui.add_button("Reset to saved")
            save = gui.add_button("Save as this demonstration's default")
            self.save_status = gui.add_markdown("")
        with gui.add_folder("Metrics"):
            self.metrics = gui.add_markdown("")
        with gui.add_folder("Export"):
            export = gui.add_button("Export robot demonstration (60 Hz npz)")
            self.export_status = gui.add_markdown("")

        for handle in (
            self.method, self.optimizer, self.scaling, self.filter, self.floating, *self.placement_sliders.values(),
            *self.grasp_sliders, self.grasp_first, self.grasp_last, self.grasp_ramp,
        ):
            handle.on_update(lambda _: self.recompute())
        self.grasp_first.on_update(self._show_frame)
        self.grasp_last.on_update(self._show_frame)
        self.frame.on_update(lambda _: self.render())
        self.overlay.on_update(lambda _: self.render())
        self.show_tube.on_update(lambda _: self.render())
        self.table_height.on_update(lambda _: self.render())
        # Dragging an end of the range shows the frame it is on.
        self.first_frame.on_update(self._show_frame)
        self.last_frame.on_update(self._show_frame)
        for slider in self.tube_sliders.values():
            slider.on_update(lambda _: self.render())
        reset.on_click(lambda _: self._reset_placement())
        save.on_click(lambda _: self._save_placement())
        self._reset_placement()
        export.on_click(lambda _: self._export())

    def _reset_placement(self) -> None:
        a = self.saved.adjustment
        values = {
            "x": a.x * 100, "y": a.y * 100, "z": a.z * 100,
            "yaw": np.degrees(a.yaw), "pitch": np.degrees(a.pitch), "roll": np.degrees(a.roll),
        }
        for name, slider in self.placement_sliders.items():
            slider.value = float(values[name])
        self.table_height.value = self.saved.table_offset * 100
        for name, value in zip(("x", "y", "yaw"), self.saved.tube_offset):
            self.tube_sliders[name].value = float(np.degrees(value) if name == "yaw" else value * 100)
        last = self.demo.num_frames - 1
        self.first_frame.value = min(self.saved.first_frame, last)
        self.last_frame.value = last if self.saved.last_frame is None else min(self.saved.last_frame, last)
        grasp = self.saved.grasp
        for slider, extra in zip(self.grasp_sliders, grasp.extra):
            slider.value = float(np.degrees(extra))
        self.grasp_first.value = min(grasp.first_frame, last)
        self.grasp_last.value = last if grasp.last_frame is None else min(grasp.last_frame, last)
        self.grasp_ramp.value = grasp.ramp_frames

    def grasp(self) -> GraspStrengthening:
        first, last = sorted((int(self.grasp_first.value), int(self.grasp_last.value)))
        return GraspStrengthening(
            tuple(float(np.deg2rad(s.value)) for s in self.grasp_sliders), first, last, int(self.grasp_ramp.value)
        )

    def frame_range(self) -> tuple[int, int]:
        """First and last video frame kept, in order whichever way the sliders were left."""
        first, last = sorted((int(self.first_frame.value), int(self.last_frame.value)))
        return (first, last) if first < last else (0, self.demo.num_frames - 1)

    def _show_frame(self, event) -> None:
        if event.client is None:  # set by "Reset to saved", not dragged
            return
        self.playing.value = False
        self.frame.value = int(event.target.value)

    def _save_placement(self) -> None:
        self.saved = SavedPlacement(
            self.settings().adjustment, self.table_height.value / 100, self.tube_adjustment_values(),
            *self.frame_range(), self.grasp(),
        )
        path = save_placement(self.csv_path, self.saved)
        self.save_status.content = f"Saved to `{path.name}`"

    def tube_adjustment_values(self) -> tuple[float, float, float]:
        t = self.tube_sliders
        return (t["x"].value / 100, t["y"].value / 100, float(np.deg2rad(t["yaw"].value)))

    def tube(self, replay: Replay):
        # From the kept frames only, as in the export.
        return place_tube(
            replay.cut(*self.frame_range()), self.table_top(replay), TubeAdjustment(*self.tube_adjustment_values())
        )

    def _draw_tube(self, replay: Replay) -> None:
        # The tube where it lies when the demonstration starts; without physics it does not move.
        tube = self.tube(replay)
        rotation = _rotation_z(tube.yaw) @ np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        for name, offset in (("/tube", np.zeros(3)), ("/skeleton_tube", SKELETON_TABLE_OFFSET)):
            self.server.scene.add_mesh_trimesh(
                name, self._tube_mesh(tube.spec), wxyz=_wxyz(rotation), position=tube.position + offset,
                visible=self.show_tube.value,
            )

    def _tube_mesh(self, spec):
        mesh = trimesh.creation.cylinder(radius=spec.radius, height=spec.length, sections=32)
        mesh.visual.face_colors = TUBE_COLOR
        return mesh

    def settings(self) -> Settings:
        s = {k: v.value for k, v in self.placement_sliders.items()}
        scaling = None if self.scaling.value == "method default" else self.scaling.value
        return Settings(
            method=self.method.value,
            optimizer=self.optimizer.value,
            scaling_mode=scaling,
            filter=self.filter.value,
            floating_hand=self.floating.value,
            grasp=self.grasp(),
            adjustment=PlacementAdjustment(
                x=s["x"] / 100, y=s["y"] / 100, z=s["z"] / 100,
                yaw=np.deg2rad(s["yaw"]), pitch=np.deg2rad(s["pitch"]), roll=np.deg2rad(s["roll"]),
            ),
        )

    # -- update ----------------------------------------------------------------
    def recompute(self) -> None:
        self._wanted.set()

    def _worker(self) -> None:
        while True:
            self._wanted.wait()
            self._wanted.clear()
            settings = self.settings()
            self.scaling.disabled = settings.method == "joint_mapping"
            self.optimizer.disabled = settings.method == "joint_mapping"
            key = settings.hand_key()
            if key not in self._hands:
                self.status.content = "Retargeting the fingers..."
                self._hands[key] = retarget_hands(self.demo, self.robot, settings)
            replay = retarget_demonstration(self.demo, self.robot, settings, hands=self._hands[key])
            self.status.content = ""
            with self.lock:
                self.replay = replay
            self.render()

    def render(self) -> None:
        with self.lock:
            replay = self.replay
        if replay is None:
            return
        i = int(self.frame.value)
        result = replay.frames[i]
        table_top = self.table_top(replay)
        for box, offset in self.tables:
            box.position = np.array([*TABLE_CENTER_XY, table_top - TABLE_SIZE[2] / 2]) + offset
        floating = replay.settings.floating_hand
        self.robot_root.visible = not floating
        self.floating_root.visible = floating
        if floating:
            self.floating_root.position = result.palm_pose[:3, 3]
            self.floating_root.wxyz = _wxyz(result.palm_pose[:3, :3])
            self.floating_urdf.update_cfg(result.hand_q[self._floating_order])
        else:
            q = np.concatenate([result.arm_q, result.hand_q])
            self.robot_urdf.update_cfg(q[self._urdf_order])

        self._draw_skeleton("/skeleton", result.skeleton_world + SKELETON_TABLE_OFFSET)
        self._draw_skeleton("/overlay", result.skeleton_world, visible=self.overlay.value)
        self.server.scene.add_point_cloud(
            "/overlay/targets", result.fingertip_targets, FINGER_COLORS, point_size=0.007,
            point_shape="diamond", visible=self.overlay.value,
        )
        self._draw_tube(replay)
        self.metrics.content = self._metrics_text(replay, i, table_top)

    def table_top(self, replay: Replay) -> float:
        return TABLE_TOP_Z + self.table_height.value / 100

    def _metrics_text(self, replay: Replay, i: int, table_top: float) -> str:
        result = replay.frames[i]
        errors = replay.stack("fingertip_error") * 1000
        names = ("thumb", "index", "middle", "ring", "pinky")
        rows = "\n".join(
            f"| {n} | {errors[i, k]:.1f} | {errors[:, k].mean():.1f} |" for k, n in enumerate(names)
        )
        skeleton_below = (table_top - result.skeleton_world[:, 2].min()) * 100
        robot_below = (table_top - result.fingertips[:, 2].min()) * 100
        worst_skeleton = (table_top - replay.stack("skeleton_world")[:, :, 2].min()) * 100
        arm = (
            "floating hand"
            if replay.settings.floating_hand
            else f"palm IK error {result.palm_position_error * 1000:.1f} mm, {np.degrees(result.palm_rotation_error):.1f}°"
        )
        penetration = replay.worst_finger_penetration() * 1000
        touching = ", ".join(
            f"{FINGER_NAMES[a]}-{FINGER_NAMES[b]} {depth * 1000:.1f} mm"
            for (a, b), depth in sorted(result.finger_penetration.items(), key=lambda kv: -kv[1])
        ) or "none"
        return (
            f"**Frame {i}** ({i / replay.demo.fps:.2f} s), {arm}\n\n"
            "| fingertip error (mm) | this frame | clip mean |\n|---|---|---|\n"
            f"{rows}\n| **all** | {errors[i].mean():.1f} | {errors.mean():.1f} |\n\n"
            f"Finger penetration (meshes): this frame {touching}. Over the clip: worst "
            f"{penetration.max():.1f} mm, deeper than 2 mm in {int((penetration > 2).sum())} of {len(penetration)} frames\n\n"
            f"Below the table: skeleton {max(skeleton_below, 0):.1f} cm, robot fingertips "
            f"{max(robot_below, 0):.1f} cm (skeleton worst over the clip {max(worst_skeleton, 0):.1f} cm)"
        )

    def _export(self) -> None:
        with self.lock:
            replay = self.replay
        if replay.settings.floating_hand:
            self.export_status.content = "Switch off the floating hand: a robot demonstration needs the arm."
            return
        first, last = self.frame_range()
        path = EXPORT_DIR / f"{replay.demo.name}_{replay.settings.method}.npz"
        export_robot_demonstration(replay.cut(first, last), path, self.table_top(replay), self.tube(replay))
        self.export_status.content = f"Wrote `{path}` (frames {first} to {last})"

    def run(self) -> None:
        while True:
            if self.playing.value:
                first, last = self.frame_range()
                following = int(self.frame.value) + 1
                self.frame.value = following if first <= following <= last else first
            time.sleep(1.0 / (self.demo.fps * self.speed.value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="a *_world_joints.csv from the hand pose estimator")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    ReplayViewer(args.csv, args.port).run()


if __name__ == "__main__":
    main()
