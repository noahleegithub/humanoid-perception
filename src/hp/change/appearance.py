"""Sample the visual appearance of a 3D change cluster from the source frames.

Used to disambiguate *moved* objects: a removal and an addition should only be
paired if they look alike, which stops unrelated changes that happen to be
nearby (a removed plant next to a newly added box) from being merged.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from hp.types import Scene

__all__ = ["sample_cluster_color", "color_similarity"]


def sample_cluster_color(
    scene: Scene,
    world_points: np.ndarray,
    *,
    tol_m: float = 0.06,
    max_points: int = 400,
    rng: Optional[np.random.Generator] = None,
) -> tuple[Optional[np.ndarray], int]:
    """Mean RGB of ``world_points`` as seen (unoccluded) in ``scene`` frames.

    Returns ``(rgb 0-255 float array of shape (3,), number of pixels sampled)``.
    Returns ``(None, 0)`` when the points are never observed.
    """
    if world_points.size == 0:
        return None, 0
    rng = rng or np.random.default_rng(0)
    pts = np.asarray(world_points, dtype=np.float64)
    if pts.shape[0] > max_points:
        sel = rng.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[sel]

    collected: list[np.ndarray] = []
    for frame in scene.frames:
        if frame.rgb is None or frame.depth is None or frame.pose is None or frame.intrinsics is None:
            continue
        rot = frame.pose[:3, :3]
        center = frame.pose[:3, 3]
        pc = (pts - center) @ rot                      # world -> camera
        z = pc[:, 2]
        valid = z > 1e-6

        k = frame.intrinsics
        u = k[0, 0] * pc[:, 0] / np.where(valid, z, 1.0) + k[0, 2]
        v = k[1, 1] * pc[:, 1] / np.where(valid, z, 1.0) + k[1, 2]
        h, w = frame.depth.shape
        inside = valid & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(inside):
            continue
        ui = np.clip(np.round(u[inside]).astype(int), 0, w - 1)
        vi = np.clip(np.round(v[inside]).astype(int), 0, h - 1)
        depth_at = frame.depth[vi, ui].astype(np.float64)
        zi = z[inside]
        consistent = np.abs(depth_at - zi) < tol_m
        if np.any(consistent):
            colors = frame.rgb[vi, ui][consistent].astype(np.float64)
            collected.append(colors)

    if not collected:
        return None, 0
    all_colors = np.concatenate(collected, axis=0)
    return all_colors.mean(axis=0), int(all_colors.shape[0])


def color_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray], sigma: float = 45.0) -> float:
    """Similarity in ``[0, 1]`` between two mean RGB colours (0-255)."""
    if a is None or b is None:
        return 0.0
    dist = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    return float(np.exp(-dist / max(sigma, 1e-6)))
