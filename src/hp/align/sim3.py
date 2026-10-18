"""Similarity transforms (Sim(3)) for aligning two independent reconstructions.

Monocular reconstruction is only defined up to scale, so aligning the "before"
and "after" scans needs a 7-DoF similarity (rotation, translation, scale), not
a rigid SE(3).  Everything here is closed-form (Umeyama) plus RANSAC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

__all__ = ["Sim3", "umeyama", "ransac_sim3"]


@dataclass
class Sim3:
    """A similarity transform ``x -> scale * R x + t``."""

    scale: float
    rotation: np.ndarray          # (3, 3)
    translation: np.ndarray       # (3,)
    inliers: Optional[np.ndarray] = None  # boolean mask over the fitted correspondences

    def __post_init__(self) -> None:
        self.rotation = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        self.translation = np.asarray(self.translation, dtype=np.float64).reshape(3)
        self.scale = float(self.scale)

    @classmethod
    def identity(cls) -> "Sim3":
        return cls(1.0, np.eye(3), np.zeros(3))

    def matrix(self) -> np.ndarray:
        """Return the 4x4 homogeneous matrix (scale folded into the top-left)."""
        m = np.eye(4)
        m[:3, :3] = self.scale * self.rotation
        m[:3, 3] = self.translation
        return m

    def apply_points(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64)
        return self.scale * (pts @ self.rotation.T) + self.translation

    def apply_pose(self, pose: np.ndarray) -> np.ndarray:
        """Transform a cam-to-world pose, keeping it rigid (scale only on translation)."""
        pose = np.asarray(pose, dtype=np.float64)
        out = np.eye(4)
        out[:3, :3] = self.rotation @ pose[:3, :3]
        out[:3, 3] = self.scale * (self.rotation @ pose[:3, 3]) + self.translation
        return out

    def inverse(self) -> "Sim3":
        scale_inv = 1.0 / self.scale
        rot_inv = self.rotation.T
        trans_inv = -scale_inv * (rot_inv @ self.translation)
        return Sim3(scale_inv, rot_inv, trans_inv)

    def compose(self, other: "Sim3") -> "Sim3":
        """Return the transform ``self @ other`` (apply ``other`` first)."""
        scale = self.scale * other.scale
        rot = self.rotation @ other.rotation
        trans = self.scale * (self.rotation @ other.translation) + self.translation
        return Sim3(scale, rot, trans)

    def rms_error(self, src: np.ndarray, dst: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((self.apply_points(src) - dst) ** 2, axis=1))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Closed-form least-squares similarity aligning ``src`` onto ``dst``.

    ``src`` and ``dst`` are ``(N, 3)`` correspondences (``N >= 3``).
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("umeyama needs at least 3 paired points of equal shape")
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    sc = src - mu_src
    dc = dst - mu_dst

    cov = (dc.T @ sc) / n
    u, sing, vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[2] = -1.0
    rot = u @ np.diag(d) @ vt
    var_src = float((sc**2).sum() / n)
    scale = float((sing * d).sum() / var_src) if (with_scale and var_src > 1e-12) else 1.0
    trans = mu_dst - scale * (rot @ mu_src)
    return Sim3(scale, rot, trans)


def ransac_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold: float = 0.08,
    iterations: int = 2000,
    with_scale: bool = True,
    sample_size: int = 3,
    seed: int = 0,
    min_inliers: int = 3,
) -> Sim3:
    """RANSAC over ``(src, dst)`` correspondences, returning the best Sim(3).

    Inliers are re-fitted at the end.  ``Sim3.inliers`` holds the final mask.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < sample_size:
        raise ValueError("not enough correspondences for RANSAC")

    rng = np.random.default_rng(seed)
    best_sim = None
    best_count = -1
    thresh_sq = threshold * threshold

    for _ in range(int(iterations)):
        idx = rng.choice(n, size=sample_size, replace=False)
        try:
            sim = umeyama(src[idx], dst[idx], with_scale=with_scale)
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.sum((sim.apply_points(src) - dst) ** 2, axis=1)
        count = int((residual < thresh_sq).sum())
        if count > best_count:
            best_count, best_sim = count, sim

    if best_sim is None:
        best_sim = umeyama(src, dst, with_scale=with_scale)
        inliers = np.ones(n, dtype=bool)
    else:
        residual = np.sum((best_sim.apply_points(src) - dst) ** 2, axis=1)
        inliers = residual < thresh_sq

    if int(inliers.sum()) >= min_inliers:
        best_sim = umeyama(src[inliers], dst[inliers], with_scale=with_scale)
        residual = np.sum((best_sim.apply_points(src) - dst) ** 2, axis=1)
        inliers = residual < thresh_sq

    best_sim.inliers = inliers
    return best_sim
