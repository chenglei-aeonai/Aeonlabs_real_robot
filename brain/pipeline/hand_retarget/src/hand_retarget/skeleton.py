"""Human demonstrations: loading skeletons, the human palm frame, and the One-Euro filter."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# The hand pose estimator's 21 keypoints, in its (dpl/MANO) order: tips come last.
KEYPOINT_NAMES = (
    "wrist",
    "index_mcp", "index_pip", "index_dip",
    "middle_mcp", "middle_pip", "middle_dip",
    "pinky_mcp", "pinky_pip", "pinky_dip",
    "ring_mcp", "ring_pip", "ring_dip",
    "thumb_cmc", "thumb_mcp", "thumb_ip",
    "index_tip", "middle_tip", "pinky_tip", "ring_tip", "thumb_tip",
)
K = {name: i for i, name in enumerate(KEYPOINT_NAMES)}
WRIST = K["wrist"]
# Each finger from its base to its tip, thumb to pinky (the robot's finger order).
FINGER_CHAINS = (
    (K["thumb_cmc"], K["thumb_mcp"], K["thumb_ip"], K["thumb_tip"]),
    (K["index_mcp"], K["index_pip"], K["index_dip"], K["index_tip"]),
    (K["middle_mcp"], K["middle_pip"], K["middle_dip"], K["middle_tip"]),
    (K["ring_mcp"], K["ring_pip"], K["ring_dip"], K["ring_tip"]),
    (K["pinky_mcp"], K["pinky_pip"], K["pinky_dip"], K["pinky_tip"]),
)
FINGERTIPS = tuple(chain[-1] for chain in FINGER_CHAINS)
FINGER_BASES = tuple(chain[0] for chain in FINGER_CHAINS)
LONG_FINGER_MCPS = FINGER_BASES[1:]
# Bones as keypoint pairs, for drawing.
BONES = tuple(
    pair
    for chain in FINGER_CHAINS
    for pair in zip((WRIST,) + chain[:-1], chain)
) + ((K["index_mcp"], K["middle_mcp"]), (K["middle_mcp"], K["ring_mcp"]), (K["ring_mcp"], K["pinky_mcp"]))


@dataclass
class HumanDemonstration:
    """One video's right-hand skeletons in the demonstration frame (x forward, y left, z up)."""

    name: str
    fps: float
    skeletons: np.ndarray  # (T, 21, 3), metres
    camera_position: np.ndarray | None = None

    @property
    def num_frames(self) -> int:
        return len(self.skeletons)

    def finger_lengths(self) -> np.ndarray:
        """Median length of each finger from base to tip, thumb to pinky."""
        lengths = []
        for chain in FINGER_CHAINS:
            bones = np.diff(self.skeletons[:, list(chain)], axis=1)
            lengths.append(np.median(np.linalg.norm(bones, axis=2).sum(axis=1)))
        return np.array(lengths)


def load_world_joints(path: str | Path) -> HumanDemonstration:
    """Read a ``*_world_joints.csv``, keeping the right hand's raw (uncorrected) positions.

    The trajectory is used exactly as the estimator predicted it: see
    docs/adr/0001-trajectory-kept-as-predicted.md.
    """
    path = Path(path)
    header = [line for line in path.read_text().splitlines() if line.startswith("#")]
    fps_match = re.search(r"at ([\d.]+) fps", header[0]) if header else None
    fps = float(fps_match.group(1)) if fps_match else 24.0
    camera = None
    for line in header:
        match = re.search(r"camera at (-?[\d.]+), (-?[\d.]+), (-?[\d.]+) m", line)
        if match:
            camera = np.array([float(v) for v in match.groups()])

    rows = csv.DictReader(line for line in path.read_text().splitlines() if not line.startswith("#"))
    frames: dict[int, np.ndarray] = {}
    for row in rows:
        if row["hand"] != "right":
            continue
        frame = frames.setdefault(int(row["frame"]), np.full((21, 3), np.nan))
        frame[int(row["joint"])] = [float(row["raw_x_m"]), float(row["raw_y_m"]), float(row["raw_z_m"])]
    if not frames:
        raise ValueError(f"{path} has no right hand; the DG5F is a right hand")
    skeletons = np.stack([frames[i] for i in sorted(frames)])
    if np.isnan(skeletons).any():
        raise ValueError(f"{path} has frames with missing right-hand keypoints")
    name = path.name.removesuffix("_world_joints.csv")
    return HumanDemonstration(name=name, fps=fps, skeletons=skeletons, camera_position=camera)


def hand_anchor(skeleton: np.ndarray) -> np.ndarray:
    """Centre of the knuckle row: the mean of the four long fingers' MCPs."""
    return skeleton[list(LONG_FINGER_MCPS)].mean(axis=0)


def palm_frame(skeleton: np.ndarray) -> np.ndarray:
    """Human palm frame (4x4) in the robot palm's axis convention, origin at the hand anchor.

    Axes, as in ``rl_dg_palm``: z along the fingers (wrist to hand anchor), y across the
    knuckles toward the thumb (pinky MCP to index MCP), x out of the palm side.
    """
    anchor = hand_anchor(skeleton)
    z = anchor - skeleton[WRIST]
    z /= np.linalg.norm(z)
    y = skeleton[K["index_mcp"]] - skeleton[K["pinky_mcp"]]
    y -= y.dot(z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    frame = np.eye(4)
    frame[:3, :3] = np.column_stack([x, y, z])
    frame[:3, 3] = anchor
    return frame


def to_frame(frame: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Express world points (..., 3) in the given 4x4 frame."""
    return (points - frame[:3, 3]) @ frame[:3, :3]


class OneEuroFilter:
    """One-Euro filter (Casiez et al., 2012) over a whole skeleton, one frame at a time.

    Smooths jitter when the hand is slow and follows quickly when it moves fast.
    """

    def __init__(self, rate_hz: float, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0):
        self.rate_hz = rate_hz
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    def _alpha(self, cutoff: np.ndarray | float) -> np.ndarray | float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.rate_hz)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self._x is None:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self._x) * self.rate_hz
        a_d = self._alpha(self.d_cutoff)
        self._dx = a_d * dx + (1.0 - a_d) * self._dx
        speed = np.linalg.norm(self._dx, axis=-1, keepdims=True)
        a = self._alpha(self.min_cutoff + self.beta * speed)
        self._x = a * x + (1.0 - a) * self._x
        return self._x.copy()
