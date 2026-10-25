"""Our own hand optimizer: one cost with several terms, analytic gradients from pinocchio.

Where dex-retargeting only sees fingertips, this also follows the direction of every
phalanx (which fixes how flexion is shared between the joints and how much a finger
spreads), keeps any two fingers from passing through each other, and never moves a joint
faster than the real hand can.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

from .robot import HAND_LOWER, HAND_UPPER, write_hand_urdf
from .skeleton import FINGER_CHAINS

# Points along each finger, thumb to pinky: the joints after the finger base and the tip.
# The segments between consecutive points are the phalanges.
FINGER_POINTS = (
    ("rj_dg_1_2", "rj_dg_1_3", "rj_dg_1_4", "rl_dg_1_tip"),
    ("rj_dg_2_2", "rj_dg_2_3", "rj_dg_2_4", "rl_dg_2_tip"),
    ("rj_dg_3_2", "rj_dg_3_3", "rj_dg_3_4", "rl_dg_3_tip"),
    ("rj_dg_4_2", "rj_dg_4_3", "rj_dg_4_4", "rl_dg_4_tip"),
    ("rj_dg_5_3", "rj_dg_5_4", "rl_dg_5_tip"),
)
# The human keypoint pair each robot phalanx follows, as indices into FINGER_CHAINS[finger].
# The DG5F pinky has one phalanx fewer: its last one follows the human PIP -> tip chord.
PHALANX_TO_HUMAN = (
    ((0, 1), (1, 2), (2, 3)),
    ((0, 1), (1, 2), (2, 3)),
    ((0, 1), (1, 2), (2, 3)),
    ((0, 1), (1, 2), (2, 3)),
    ((0, 1), (1, 3)),
)
# Each phalanx is kept apart from other fingers as a capsule around its axis. DG5F
# phalanges are 19 x 22.4 mm and the fingertip pads bulge 4.5 mm toward the palm; this
# radius keeps the real meshes (checked in collisions.py) within 2 mm on the sample clip.
FINGER_RADIUS = 0.0125
CONTACT_DISTANCE = 2 * FINGER_RADIUS  # between two phalanx axes whose surfaces touch
# The fingertip frames sit 17.6 mm short of the end of the finger; the last capsule runs
# on to where its rounded end meets the real end.
TIP_END_OFFSET = 0.0176 - FINGER_RADIUS
# Direction from the fingertip frame toward the end of the finger, in the tip frame.
TIP_AXES = (np.array([0.0, 1.0, 0.0]),) + (np.array([0.0, 0.0, 1.0]),) * 4
# Spread and pinky opposition joints (indices into the 20 hand joints).
SPREAD_JOINTS = np.array([4, 8, 12, 16, 17])
MAX_JOINT_SPEED = np.pi  # rad/s, the DG5F's velocity limit in the URDF


@dataclass(frozen=True)
class CostWeights:
    """Each term is (error / scale)^2 times its weight."""

    fingertip: float = 1.0
    fingertip_scale: float = 0.01  # m
    direction: float = 0.5
    direction_scale: float = 0.2  # unit-vector difference, about 11 degrees
    collision: float = 1.0
    collision_scale: float = 0.001  # m of overlap
    spread: float = 0.05
    spread_scale: float = np.deg2rad(30.0)
    previous: float = 0.05
    previous_scale: float = np.deg2rad(30.0)


def closest_between_segments(p1, q1, p2, q2):
    """Closest points between many segment pairs, rows of (N, 3): (s, t, distance), after Ericson (2005)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e = (d1 * d1).sum(1), (d2 * d2).sum(1)
    b, c, f = (d1 * d2).sum(1), (d1 * r).sum(1), (d2 * r).sum(1)
    denom = a * e - b * b
    s = np.where(denom > 1e-12, np.clip((b * f - c * e) / np.where(denom > 1e-12, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e
    below, above = t < 0.0, t > 1.0
    s = np.where(below, np.clip(-c / a, 0.0, 1.0), np.where(above, np.clip((b - c) / a, 0.0, 1.0), s))
    t = np.clip(t, 0.0, 1.0)
    gap = p1 + s[:, None] * d1 - p2 - t[:, None] * d2
    return s, t, np.linalg.norm(gap, axis=1)


class HandModel:
    """Positions and Jacobians of the finger points, in the palm frame.

    Rows are each finger's points in FINGER_POINTS order, then one tip-end point per finger.
    """

    def __init__(self) -> None:
        self.model = pin.buildModelFromUrdf(str(_hand_urdf()))
        tip_ends = []
        for finger, points in enumerate(FINGER_POINTS):
            tip = self.model.frames[self.model.getFrameId(points[-1])]
            name = f"{points[-1]}_end"
            self.model.addFrame(
                pin.Frame(name, tip.parentJoint, tip.placement * pin.SE3(np.eye(3), TIP_END_OFFSET * TIP_AXES[finger]),
                          pin.FrameType.OP_FRAME)
            )
            tip_ends.append(name)
        self.data = self.model.createData()
        names = [name for finger in FINGER_POINTS for name in finger] + tip_ends
        self.frame_ids = [self.model.getFrameId(name) for name in names]
        self.finger_slices = []
        start = 0
        for finger in FINGER_POINTS:
            self.finger_slices.append(slice(start, start + len(finger)))
            start += len(finger)
        self.tip_rows = np.array([s.stop - 1 for s in self.finger_slices])
        self.tip_end_rows = np.arange(start, start + len(FINGER_POINTS))

        # Capsule segments: each phalanx, the last one running on to the tip end.
        segments, fingers = [], []
        for finger, rows in enumerate(self.finger_slices):
            chain = list(range(rows.start, rows.stop - 1)) + [self.tip_end_rows[finger]]
            for a, b in zip(chain[:-1], chain[1:]):
                segments.append((a, b))
                fingers.append(finger)
        pairs = [(i, j) for i in range(len(segments)) for j in range(i + 1, len(segments)) if fingers[i] != fingers[j]]
        seg = np.array(segments)
        self.pair_a = seg[[i for i, _ in pairs]]  # (P, 2) point rows of the first segment
        self.pair_b = seg[[j for _, j in pairs]]

    def points(self, q: np.ndarray, jacobians: bool = True):
        if jacobians:
            pin.computeJointJacobians(self.model, self.data, q)
        else:
            pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        positions = np.stack([self.data.oMf[f].translation for f in self.frame_ids])
        if not jacobians:
            return positions, None
        frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        jac = np.stack([pin.getFrameJacobian(self.model, self.data, f, frame)[:3] for f in self.frame_ids])
        return positions, jac

    def capsule_gaps(self, positions: np.ndarray):
        """For every pair of phalanges on different fingers: (s, t, axis distance)."""
        return closest_between_segments(
            positions[self.pair_a[:, 0]], positions[self.pair_a[:, 1]],
            positions[self.pair_b[:, 0]], positions[self.pair_b[:, 1]],
        )


@lru_cache(maxsize=1)
def _hand_urdf():
    return write_hand_urdf()


class HandOptimizer:
    """Solves one frame: 20 DG5F joint angles for the given references, starting from the last frame."""

    def __init__(self, rate_hz: float, weights: CostWeights = CostWeights()):
        self.model = HandModel()
        self.weights = weights
        self.max_step = MAX_JOINT_SPEED / rate_hz
        self.reset()

    def reset(self) -> None:
        self.last_q: np.ndarray | None = None

    def solve(self, directions: list[np.ndarray], task_cost) -> np.ndarray:
        """``directions``: per finger, the human unit vectors its phalanges should follow.

        ``task_cost(positions, jacobians) -> (cost, gradient)`` is the method's own term
        (fingertip positions or DexPilot vectors).
        """
        w = self.weights
        previous = np.zeros(20) if self.last_q is None else self.last_q
        lower, upper = HAND_LOWER.copy(), HAND_UPPER.copy()
        if self.last_q is not None:
            lower = np.maximum(lower, self.last_q - self.max_step)
            upper = np.minimum(upper, self.last_q + self.max_step)

        def cost(q):
            positions, jac = self.model.points(q)
            total, grad = task_cost(positions, jac)

            k = w.direction / w.direction_scale**2
            for finger, pairs in enumerate(PHALANX_TO_HUMAN):
                rows = range(self.model.finger_slices[finger].start, self.model.finger_slices[finger].stop)
                rows = list(rows)
                for phalanx, target in enumerate(directions[finger]):
                    a, b = rows[phalanx], rows[phalanx + 1]
                    v = positions[b] - positions[a]
                    length = np.linalg.norm(v)
                    u = v / length
                    diff = u - target
                    total += k * diff @ diff
                    du = (np.eye(3) - np.outer(u, u)) / length
                    grad += 2 * k * diff @ du @ (jac[b] - jac[a])

            k = w.collision / w.collision_scale**2
            m = self.model
            s_, t_, dist = m.capsule_gaps(positions)
            for p in np.flatnonzero((dist < CONTACT_DISTANCE) & (dist > 1e-9)):
                (a1, a2), (b1, b2), si, ti = m.pair_a[p], m.pair_b[p], s_[p], t_[p]
                overlap = CONTACT_DISTANCE - dist[p]
                normal = (
                    positions[a1] + si * (positions[a2] - positions[a1])
                    - positions[b1] - ti * (positions[b2] - positions[b1])
                ) / dist[p]
                ddist = normal @ ((1 - si) * jac[a1] + si * jac[a2] - (1 - ti) * jac[b1] - ti * jac[b2])
                total += k * overlap**2
                grad += -2 * k * overlap * ddist

            k = w.spread / w.spread_scale**2
            total += k * q[SPREAD_JOINTS] @ q[SPREAD_JOINTS]
            grad[SPREAD_JOINTS] += 2 * k * q[SPREAD_JOINTS]

            k = w.previous / w.previous_scale**2
            total += k * (q - previous) @ (q - previous)
            grad += 2 * k * (q - previous)
            return total, grad

        start = np.clip(previous, lower, upper)
        result = minimize(
            cost, start, jac=True, method="L-BFGS-B", bounds=list(zip(lower, upper)),
            options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-9},
        )
        self.last_q = np.clip(result.x, lower, upper)
        return self.last_q.copy()


def human_phalanx_directions(skeleton_in_palm: np.ndarray) -> list[np.ndarray]:
    """Unit vectors of the human bones each robot phalanx follows, per finger (thumb to pinky)."""
    directions = []
    for finger, pairs in enumerate(PHALANX_TO_HUMAN):
        chain = FINGER_CHAINS[finger]
        vectors = np.stack([skeleton_in_palm[chain[b]] - skeleton_in_palm[chain[a]] for a, b in pairs])
        directions.append(vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
    return directions


def fingertip_cost(targets: np.ndarray, tip_rows: np.ndarray, weights: CostWeights):
    """Term pulling the five fingertips onto ``targets`` (5, 3)."""
    k = weights.fingertip / weights.fingertip_scale**2

    def term(positions, jac):
        diff = positions[tip_rows] - targets
        grad = 2 * k * np.einsum("fi,fij->j", diff, jac[tip_rows])
        return k * float((diff * diff).sum()), grad

    return term


class DexPilotVectors:
    """DexPilot's vectors (palm to fingertips, fingertip to fingertip) with its pinch projection.

    Same rules and weights as dex-retargeting's DexPilotOptimizer: a thumb pair closer than
    ``project_dist`` is pulled into contact until it is farther than ``escape_dist``; a
    pair of other fingers is pulled together only when both are pinching the thumb.
    """

    # dex-retargeting pulls a pinching pair's fingertip frames to 0.1 mm (eta1), but those
    # frames sit inside the pads: that would push the fingers into each other. Here the
    # pinch target is the distance at which the two fingertips' surfaces touch.
    def __init__(self, project_dist=0.03, escape_dist=0.05, eta1=CONTACT_DISTANCE, eta2=0.03):
        self.pairs = [(j, i) for i in range(5) for j in range(i + 1, 5)]  # (origin, task), thumb pairs first
        self.thumb_pairs = 4
        self.project_dist, self.escape_dist, self.eta1, self.eta2 = project_dist, escape_dist, eta1, eta2
        self.reset()

    def reset(self) -> None:
        self.projected = np.zeros(len(self.pairs), dtype=bool)

    def cost(self, tips: np.ndarray, tip_rows: np.ndarray, weights: CostWeights):
        vectors = np.stack([tips[task] - tips[origin] for origin, task in self.pairs])
        dist = np.linalg.norm(vectors, axis=1)
        n = self.thumb_pairs
        self.projected[:n][dist[:n] < self.project_dist] = True
        self.projected[:n][dist[:n] > self.escape_dist] = False
        for k, (origin, task) in enumerate(self.pairs[n:], start=n):
            both = self.projected[origin - 1] and self.projected[task - 1]
            self.projected[k] = both and dist[k] <= 0.03
        eta = np.where(np.arange(len(self.pairs)) < n, self.eta1, self.eta2)
        targets = np.where(self.projected[:, None], vectors / (dist[:, None] + 1e-6) * eta[:, None], vectors)
        # dex-retargeting weighs pinch pairs 1 (projected: 200 thumb, 400 others) and
        # palm-to-tip vectors 15; here divided by 15 so palm vectors weigh 1.
        pair_weights = np.where(self.projected, np.where(np.arange(len(self.pairs)) < n, 200.0, 400.0), 1.0) / 15.0
        k = weights.fingertip / weights.fingertip_scale**2

        def term(positions, jac):
            p, J = positions[tip_rows], jac[tip_rows]
            total, grad = 0.0, np.zeros(jac.shape[2])
            for (origin, task), target, wp in zip(self.pairs, targets, pair_weights):
                diff = p[task] - p[origin] - target
                total += k * wp * diff @ diff
                grad += 2 * k * wp * diff @ (J[task] - J[origin])
            # Palm origin to each fingertip: the palm is the root, so its point is fixed at 0.
            diff = p - tips
            total += k * float((diff * diff).sum())
            grad += 2 * k * np.einsum("fi,fij->j", diff, J)
            return total, grad

        return term
