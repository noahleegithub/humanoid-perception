"""Top-level change detection between two fused voxel grids."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import ndimage

from hp.change.appearance import color_similarity, sample_cluster_color
from hp.change.cluster import clusters_from_mask
from hp.change.moved import pair_moved
from hp.types import ChangeCluster, Scene, VoxelGrid, VoxelState

__all__ = ["diff_grids", "detect_changes"]


def diff_grids(
    grid_a: VoxelGrid,
    grid_b: VoxelGrid,
    *,
    min_occ: int = 1,
    min_free: int = 1,
    surface_band_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(added_mask, removed_mask)``.

    * added   : observed empty in A, occupied in B
    * removed : occupied in A, observed empty in B

    Voxels that were unobserved in either scan never appear in either mask,
    which is how occlusion / incomplete coverage is discounted.

    ``surface_band_m`` suppresses changes within a tolerance of a surface
    present in the *other* scan.  This absorbs depth noise, sub-voxel
    registration error, and the discretisation of curved/thin surfaces, which
    would otherwise carpet every wall with spurious differences.
    """
    state_a = grid_a.state(min_occ=min_occ, min_free=min_free)
    state_b = grid_b.state(min_occ=min_occ, min_free=min_free)
    a_occ = state_a == VoxelState.OCCUPIED
    b_occ = state_b == VoxelState.OCCUPIED
    a_free = state_a == VoxelState.FREE
    b_free = state_b == VoxelState.FREE

    radius = int(round(surface_band_m / grid_a.voxel_size)) if surface_band_m > 0 else 0
    if radius > 0:
        structure = _ball(radius)
        a_occ_d = ndimage.binary_dilation(a_occ, structure=structure)
        b_occ_d = ndimage.binary_dilation(b_occ, structure=structure)
        # a voxel that is next to a surface *in its own scan* is unreliable as
        # free space (grazing rays skim walls/floors).  Drop it.
        a_free = a_free & ~a_occ_d
        b_free = b_free & ~b_occ_d
    else:
        a_occ_d, b_occ_d = a_occ, b_occ

    added = b_occ & a_free & ~a_occ_d
    removed = a_occ & b_free & ~b_occ_d
    return added, removed


def _ball(radius: int) -> np.ndarray:
    radius = max(int(radius), 0)
    ax = np.arange(-radius, radius + 1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing="ij")
    return (xx * xx + yy * yy + zz * zz) <= radius * radius


def detect_changes(
    grid_a: VoxelGrid,
    grid_b: VoxelGrid,
    *,
    min_occ: int = 1,
    min_free: int = 1,
    surface_band_m: float = 0.0,
    min_cluster_voxels: int = 20,
    moved_max_dist_m: float = 1.5,
    moved_max_volume_ratio: float = 3.0,
    appearance_sim: Optional[np.ndarray] = None,
    min_appearance: float = 0.0,
    scene_a: Optional[Scene] = None,
    scene_b: Optional[Scene] = None,
) -> list[ChangeCluster]:
    """Detect added / removed / moved clusters between two scans.

    ``grid_a`` is the "before" scan, ``grid_b`` the "after" scan.  Both must
    share the same origin, voxel size and dims (they are aligned beforehand).

    When ``scene_a``/``scene_b`` are given, each cluster's mean colour is
    sampled from the frames and used to reject spurious moved pairings.
    """
    added_mask, removed_mask = diff_grids(
        grid_a, grid_b, min_occ=min_occ, min_free=min_free, surface_band_m=surface_band_m
    )

    removed, _ = clusters_from_mask(removed_mask, grid_a, "removed", min_cluster_voxels, start_id=0)
    added, _ = clusters_from_mask(added_mask, grid_b, "added", min_cluster_voxels, start_id=0)

    # sample per-cluster appearance for moved-object disambiguation
    if scene_a is not None:
        for c in removed:
            c.color, _ = sample_cluster_color(scene_a, grid_a.index_to_world(c.voxel_indices))
    if scene_b is not None:
        for c in added:
            c.color, _ = sample_cluster_color(scene_b, grid_b.index_to_world(c.voxel_indices))

    if appearance_sim is None and scene_a is not None and scene_b is not None and removed and added:
        appearance_sim = np.zeros((len(removed), len(added)), dtype=np.float64)
        for i, r in enumerate(removed):
            for j, a in enumerate(added):
                appearance_sim[i, j] = color_similarity(r.color, a.color)

    removed_rem, added_rem, moved = pair_moved(
        removed,
        added,
        max_dist_m=moved_max_dist_m,
        max_volume_ratio=moved_max_volume_ratio,
        appearance_sim=appearance_sim,
        min_appearance=min_appearance,
    )
    clusters: Sequence[ChangeCluster] = list(removed_rem) + list(added_rem) + list(moved)
    clusters = sorted(clusters, key=lambda c: -c.volume)
    for cid, cluster in enumerate(clusters):
        cluster.id = cid
    return clusters
