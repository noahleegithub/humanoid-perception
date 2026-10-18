"""Procedural indoor scenes for synthetic data generation.

A scene is an axis-aligned room plus a set of labelled axis-aligned boxes.
Everything is analytic, so the renderer can produce *exact* depth and
per-pixel object ids -- perfect ground truth for testing the change engine
without any GPU or photogrammetry in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Box:
    """An axis-aligned object."""

    lo: np.ndarray
    hi: np.ndarray
    label: str
    color: tuple[int, int, int]
    name: str = ""

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)
        if not self.name:
            self.name = self.label

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.lo + self.hi)

    @property
    def extent(self) -> np.ndarray:
        return self.hi - self.lo

    def copy(self) -> "Box":
        return Box(self.lo.copy(), self.hi.copy(), self.label, self.color, name=self.name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "lo": [float(x) for x in self.lo],
            "hi": [float(x) for x in self.hi],
            "center": [float(x) for x in self.center],
            "extent": [float(x) for x in self.extent],
        }


@dataclass
class Room:
    """An empty room (interior box) and its objects."""

    lo: np.ndarray
    hi: np.ndarray
    objects: list[Box] = field(default_factory=list)
    wall_color: tuple[int, int, int] = (225, 222, 214)
    floor_color: tuple[int, int, int] = (150, 140, 128)
    ceiling_color: tuple[int, int, int] = (240, 240, 240)

    def __post_init__(self) -> None:
        self.lo = np.asarray(self.lo, dtype=np.float64).reshape(3)
        self.hi = np.asarray(self.hi, dtype=np.float64).reshape(3)

    def add(self, box: Box) -> "Room":
        self.objects.append(box)
        return self

    def clone(self) -> "Room":
        return Room(self.lo.copy(), self.hi.copy(), [b.copy() for b in self.objects],
                    self.wall_color, self.floor_color, self.ceiling_color)

    def object_by_name(self, name: str) -> Box | None:
        for b in self.objects:
            if b.name == name:
                return b
        return None

    def to_dict(self) -> dict:
        return {
            "room": {"lo": self.lo.tolist(), "hi": self.hi.tolist()},
            "objects": [b.to_dict() for b in self.objects],
        }


def default_room() -> Room:
    """A 4 x 3 x 2.6 m room with a handful of everyday objects."""
    room = Room(lo=[0, 0, 0], hi=[4.0, 3.0, 2.6])
    room.add(Box([2.6, 1.9, 0.0], [3.4, 2.6, 0.75], label="table", color=(150, 105, 70)))
    room.add(Box([2.05, 1.95, 0.0], [2.45, 2.35, 0.45], label="chair", color=(90, 90, 100)))
    room.add(Box([0.2, 0.2, 0.0], [0.6, 1.4, 0.9], label="cabinet", color=(180, 175, 165)))
    room.add(Box([0.3, 0.4, 0.9], [0.5, 0.6, 1.25], label="lamp", color=(240, 220, 120)))
    room.add(Box([3.4, 0.3, 0.0], [3.7, 0.6, 0.4], label="box", color=(200, 170, 90)))
    room.add(Box([1.7, 2.7, 0.0], [2.0, 3.0, 1.05], label="plant", color=(70, 150, 70)))
    return room
