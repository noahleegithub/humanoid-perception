"""Core data types shared across the pipeline.

Everything is expressed with numpy arrays in a right-handed world frame
(metres, +Z up by convention).  Camera poses are 4x4 cam-to-world matrices
following the OpenCV convention (x right, y down, z forward).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Frame:
    """A single registered RGB-D view.

    ``depth`` is camera-space z in metres (float32, ``HxW``).
    ``pose`` is a 4x4 cam-to-world transform.
    ``intrinsics`` is a 3x3 pinhole matrix.
    """

    index: int
    rgb: Optional[np.ndarray] = None          # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None        # (H, W) float32, metres
    pose: Optional[np.ndarray] = None         # (4, 4) cam-to-world
    intrinsics: Optional[np.ndarray] = None   # (3, 3)
    path: Optional[str] = None

    @property
    def height(self) -> int:
        if self.rgb is not None:
            return int(self.rgb.shape[0])
        if self.depth is not None:
            return int(self.depth.shape[0])
        raise ValueError("frame has neither rgb nor depth")

    @property
    def width(self) -> int:
        if self.rgb is not None:
            return int(self.rgb.shape[1])
        if self.depth is not None:
            return int(self.depth.shape[1])
        raise ValueError("frame has neither rgb nor depth")

    @property
    def camera_center(self) -> np.ndarray:
        if self.pose is None:
            raise ValueError("frame has no pose")
        return self.pose[:3, 3].copy()


@dataclass
class Scene:
    """A reconstructed scene: registered frames plus an optional fused point cloud."""

    name: str
    frames: list[Frame] = field(default_factory=list)
    points: Optional[np.ndarray] = None       # (N, 3) world coordinates
    colors: Optional[np.ndarray] = None       # (N, 3) uint8
    meta: dict = field(default_factory=dict)

    def pointmap_frames(self) -> list[Frame]:
        """Frames that carry both depth and pose (usable for volumetric fusion)."""
        return [f for f in self.frames if f.depth is not None and f.pose is not None]


class VoxelState:
    """Integer labels for a voxel's observation status."""

    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    NAMES = {0: "unknown", 1: "free", 2: "occupied"}


@dataclass
class VoxelGrid:
    """Dense trilinear voxel grid holding free-space and surface observation counts.

    ``origin`` is the world coordinate of the low corner of voxel ``(0, 0, 0)``.
    ``free`` and ``occ`` are int32 arrays of shape ``dims``.
    """

    voxel_size: float
    origin: np.ndarray                        # (3,)
    dims: tuple[int, int, int]
    free: np.ndarray
    occ: np.ndarray

    # ---- construction -----------------------------------------------------
    @classmethod
    def empty(cls, voxel_size: float, origin: np.ndarray, dims: tuple[int, int, int]) -> "VoxelGrid":
        dims = tuple(int(d) for d in dims)
        return cls(
            voxel_size=float(voxel_size),
            origin=np.asarray(origin, dtype=np.float64).reshape(3),
            dims=dims,
            free=np.zeros(dims, dtype=np.int32),
            occ=np.zeros(dims, dtype=np.int32),
        )

    # ---- coordinate helpers ----------------------------------------------
    @property
    def shape(self) -> tuple[int, int, int]:
        return self.dims

    def world_to_index(self, pts: np.ndarray) -> np.ndarray:
        """World ``(..., 3)`` -> integer voxel indices ``(..., 3)`` (unbounded)."""
        pts = np.asarray(pts, dtype=np.float64)
        return np.floor((pts - self.origin) / self.voxel_size).astype(np.int64)

    def index_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Voxel indices ``(..., 3)`` -> world coordinate of the voxel centre."""
        idx = np.asarray(idx, dtype=np.float64)
        return self.origin + (idx + 0.5) * self.voxel_size

    def in_bounds(self, idx: np.ndarray) -> np.ndarray:
        idx = np.asarray(idx)
        return np.all((idx >= 0) & (idx < np.asarray(self.dims)), axis=-1)

    # ---- state ------------------------------------------------------------
    def state(self, min_occ: int = 1, min_free: int = 1) -> np.ndarray:
        """Return per-voxel :class:`VoxelState` labels.

        A voxel is OCCUPIED if it has enough surface hits, else FREE if it has
        enough free-space hits, else UNKNOWN.  Occupied takes precedence, which
        is the conservative choice when a voxel was hit by both.
        """
        st = np.full(self.dims, VoxelState.UNKNOWN, dtype=np.int8)
        st[self.free >= min_free] = VoxelState.FREE
        st[self.occ >= min_occ] = VoxelState.OCCUPIED
        return st

    def occupied_mask(self, min_occ: int = 1) -> np.ndarray:
        return self.occ >= min_occ

    def free_mask(self, min_free: int = 1) -> np.ndarray:
        return self.free >= min_free


@dataclass
class ChangeCluster:
    """A connected region of voxels that differs between the two scans."""

    id: int
    kind: str                                  # "added" | "removed" | "moved"
    centroid: np.ndarray                       # (3,) world
    bbox_min: np.ndarray                       # (3,) world
    bbox_max: np.ndarray                       # (3,) world
    n_voxels: int
    volume: float                              # m^3
    voxel_indices: np.ndarray                  # (K, 3) int
    score: float = 0.0
    color: Optional[np.ndarray] = None         # (3,) mean RGB 0-255, if sampled
    # for "moved": the object's own size (not the union of before/after bboxes)
    object_extent: Optional[np.ndarray] = None
    # populated for "moved" clusters after pairing
    pair_id: Optional[int] = None
    move_from: Optional[np.ndarray] = None
    move_to: Optional[np.ndarray] = None

    @property
    def extent(self) -> np.ndarray:
        """Axis-aligned side lengths in metres."""
        return self.bbox_max - self.bbox_min

    @property
    def size(self) -> np.ndarray:
        """The object's own size in metres (falls back to the bbox extent)."""
        if self.object_extent is not None:
            return np.asarray(self.object_extent, dtype=np.float64)
        return self.extent

    @property
    def center(self) -> np.ndarray:
        return self.centroid

    def to_dict(self) -> dict:
        return {
            "id": int(self.id),
            "kind": self.kind,
            "centroid": [float(x) for x in self.centroid],
            "bbox_min": [float(x) for x in self.bbox_min],
            "bbox_max": [float(x) for x in self.bbox_max],
            "extent_m": [float(x) for x in self.extent],
            "size_m": [float(x) for x in self.size],
            "volume_m3": float(self.volume),
            "n_voxels": int(self.n_voxels),
            "score": float(self.score),
            "color": None if self.color is None else [float(x) for x in self.color],
            "pair_id": None if self.pair_id is None else int(self.pair_id),
            "move_from": None if self.move_from is None else [float(x) for x in self.move_from],
            "move_to": None if self.move_to is None else [float(x) for x in self.move_to],
        }


@dataclass
class LabeledChange:
    """A :class:`ChangeCluster` with a semantic label and a sentence."""

    cluster: ChangeCluster
    label: str
    confidence: float
    description: str
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.cluster.to_dict()
        d.update(
            {
                "label": self.label,
                "confidence": float(self.confidence),
                "description": self.description,
                "evidence": self.evidence,
            }
        )
        return d


@dataclass
class ChangeSet:
    """Full result of comparing a source scene against a target scene."""

    src_name: str
    dst_name: str
    voxel_size: float
    clusters: list[ChangeCluster] = field(default_factory=list)
    labeled: list[LabeledChange] = field(default_factory=list)
    sim3: Optional[np.ndarray] = None          # (4, 4) source -> target
    scale: float = 1.0
    align_info: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def counts(self) -> dict:
        out = {"added": 0, "removed": 0, "moved": 0}
        for c in self.clusters:
            out[c.kind] = out.get(c.kind, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "src": self.src_name,
            "dst": self.dst_name,
            "voxel_size": float(self.voxel_size),
            "scale": float(self.scale),
            "counts": self.counts(),
            "align_info": self.align_info,
            "stats": self.stats,
            "changes": [lc.to_dict() for lc in self.labeled]
            if self.labeled
            else [c.to_dict() for c in self.clusters],
        }
