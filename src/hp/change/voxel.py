"""Volumetric fusion: turn registered RGB-D frames into a visibility-aware voxel grid.

Each voxel accumulates two counts per scan:

* ``free`` -- how many frames observed it as empty space (free-space carving)
* ``occ``  -- how many frames observed a surface inside it

Because the counts come from *observation*, a voxel that was never seen stays
``UNKNOWN`` and is never reported as a change.  This is what lets the pipeline
ignore occlusion and incomplete coverage instead of hallucinating difference.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from hp.types import Frame, Scene, VoxelGrid

__all__ = [
    "frame_world_points",
    "scene_world_points",
    "fit_grid",
    "integrate_frames",
    "integrate_scene",
]


def _backproject(frame: Frame, pixel_stride: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space surface points and unit ray directions for a frame.

    Only pixels with valid depth are returned.
    """
    if frame.depth is None or frame.pose is None or frame.intrinsics is None:
        raise ValueError("frame needs depth, pose and intrinsics to be integrated")
    depth = frame.depth
    height, width = depth.shape
    ys = np.arange(0, height, pixel_stride)
    xs = np.arange(0, width, pixel_stride)
    uu, vv = np.meshgrid(xs, ys)

    z = depth[vv, uu].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)

    k = frame.intrinsics
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])

    u = uu.ravel().astype(np.float64)
    v = vv.ravel().astype(np.float64)
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    dz = np.ones_like(dx)
    norm = np.sqrt(dx * dx + dy * dy + 1.0)

    raycam = np.stack([dx / norm, dy / norm, dz / norm], axis=1)  # (P, 3)
    rot = frame.pose[:3, :3]
    center = frame.pose[:3, 3]
    raydir = raycam @ rot.T                                       # (P, 3) world
    length = z.ravel() * norm                                     # (P,) metres to surface

    flat_valid = valid.ravel()
    raydir = raydir[flat_valid]
    length = length[flat_valid]
    surf = center[None, :] + length[:, None] * raydir
    return surf, raydir, length


def frame_world_points(frame: Frame, pixel_stride: int = 2) -> np.ndarray:
    """World-space surface points for a single frame."""
    surf, _, _ = _backproject(frame, pixel_stride=pixel_stride)
    return surf


def scene_world_points(scene: Scene, pixel_stride: int = 4) -> np.ndarray:
    """Concatenated world-space surface points for every registered frame."""
    chunks = [frame_world_points(f, pixel_stride=pixel_stride) for f in scene.pointmap_frames()]
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def fit_grid(
    point_sets: Iterable[np.ndarray],
    voxel_size: float,
    pad: float = 0.4,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Compute a shared grid bounding box covering every point set.

    Returns ``(origin, dims)`` where ``origin`` is the low corner (already
    snapped to the voxel lattice) and ``dims`` is the integer extent.
    """
    pts = np.concatenate([np.asarray(p) for p in point_sets if len(p)], axis=0)
    if pts.size == 0:
        raise ValueError("cannot fit a grid to zero points")
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    origin = np.floor(lo / voxel_size) * voxel_size
    extent = np.ceil((hi - origin) / voxel_size)
    dims = tuple(int(max(1, d)) for d in extent)
    return origin, dims


def integrate_frames(
    grid: VoxelGrid,
    frames: Iterable[Frame],
    *,
    pixel_stride: int = 3,
    trunc: float = 0.10,
    max_ray: float = 8.0,
    max_samples: int = 200,
    min_occ_obs_frames: bool = True,
) -> VoxelGrid:
    """Carve free space and splat surfaces from ``frames`` into ``grid`` (in place).

    Each frame contributes at most one free vote and one surface vote per voxel,
    so ``min_occ_obs`` in the config means "seen as a surface by N frames".
    """
    origin = grid.origin
    voxel = grid.voxel_size
    dims = np.asarray(grid.dims, dtype=np.int64)
    n_total = int(np.prod(dims))
    free_flat = grid.free.reshape(-1)
    occ_flat = grid.occ.reshape(-1)

    for frame in frames:
        surf, raydir, length = _backproject(frame, pixel_stride=pixel_stride)

        # --- free space -----------------------------------------------------
        t_max = np.clip(length - trunc, 0.0, max_ray)
        active = t_max > 0
        if np.any(active):
            d = raydir[active]
            t = t_max[active]
            c = frame.pose[:3, 3]
            frac = (np.arange(1, max_samples + 1) / max_samples)[None, :]
            pts = c[None, None, :] + (t[:, None, None] * frac[..., None]) * d[:, None, :]
            idx = np.floor((pts - origin) / voxel).astype(np.int64).reshape(-1, 3)
            ok = np.all((idx >= 0) & (idx < dims), axis=1)
            if np.any(ok):
                flat = (
                    (idx[ok, 0] * dims[1] + idx[ok, 1]) * dims[2] + idx[ok, 2]
                )
                uniq = np.unique(flat)
                free_flat += np.bincount(uniq, minlength=n_total)

        # --- surfaces -------------------------------------------------------
        idx = np.floor((surf - origin) / voxel).astype(np.int64)
        ok = np.all((idx >= 0) & (idx < dims), axis=1)
        if np.any(ok):
            flat = (idx[ok, 0] * dims[1] + idx[ok, 1]) * dims[2] + idx[ok, 2]
            uniq = np.unique(flat)
            occ_flat += np.bincount(uniq, minlength=n_total)

    return grid


def integrate_scene(
    scene: Scene,
    grid: VoxelGrid,
    **kwargs,
) -> VoxelGrid:
    """Integrate every registered frame of ``scene`` into ``grid``."""
    return integrate_frames(grid, scene.pointmap_frames(), **kwargs)
