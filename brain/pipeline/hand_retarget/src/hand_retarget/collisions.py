"""Collisions between the DG5F's fingers, checked on its real collision meshes.

The optimizer keeps fingers apart with capsules (fast, differentiable); this module is the
independent referee that says whether two fingers' meshes actually overlap.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pinocchio as pin

from .robot import ASSETS_DIR

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _finger_of(link: str) -> int | None:
    """Finger number (0 = thumb) of a finger link, or None for the palm and base."""
    parts = link.split("_")
    return int(parts[2]) - 1 if len(parts) >= 4 and parts[2].isdigit() else None


class FingerCollisionChecker:
    """Which pairs of fingers overlap, on the collision meshes of the hand-only URDF."""

    def __init__(self, hand_urdf: str):
        self.model = pin.buildModelFromUrdf(hand_urdf)
        self.data = self.model.createData()
        self.geometry = pin.buildGeomFromUrdf(
            self.model, hand_urdf, pin.GeometryType.COLLISION, package_dirs=[str(ASSETS_DIR)]
        )
        # coal's penetration depth between two general meshes is unreliable (it reported 9 mm
        # where the meshes overlap by under 1 mm); between convex shapes it is exact, and the
        # phalanges are nearly convex.
        for obj in self.geometry.geometryObjects:
            obj.geometry.buildConvexRepresentation(False)
            obj.geometry = obj.geometry.convex
        self.pair_fingers = []
        for a, b in combinations(range(self.geometry.ngeoms), 2):
            fa = _finger_of(self.model.frames[self.geometry.geometryObjects[a].parentFrame].name)
            fb = _finger_of(self.model.frames[self.geometry.geometryObjects[b].parentFrame].name)
            # Links of the same finger touch at their joints by design; the palm is not a finger.
            if fa is None or fb is None or fa == fb:
                continue
            self.geometry.addCollisionPair(pin.CollisionPair(a, b))
            self.pair_fingers.append((min(fa, fb), max(fa, fb)))
        self.geometry_data = pin.GeometryData(self.geometry)

    def penetrations(self, hand_q: np.ndarray) -> dict[tuple[int, int], float]:
        """How deep (m) the meshes of each overlapping pair of fingers (0 = thumb) go into each other."""
        pin.computeDistances(self.model, self.data, self.geometry, self.geometry_data, hand_q)
        depths: dict[tuple[int, int], float] = {}
        for k, result in enumerate(self.geometry_data.distanceResults):
            if result.min_distance < 0.0:  # signed: negative is the penetration depth
                pair = self.pair_fingers[k]
                depths[pair] = max(depths.get(pair, 0.0), -result.min_distance)
        return depths
