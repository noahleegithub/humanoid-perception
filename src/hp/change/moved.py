"""Pair a removed cluster with an added cluster to recognise a *moved* object.

A moved object shows up as a removal where it used to be and an addition where
it went.  If the two blobs are close, similarly sized, and (optionally)
visually similar, we merge them into a single "moved" change carrying a
from/to trajectory instead of reporting two unrelated events.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from hp.types import ChangeCluster

__all__ = ["pair_moved"]


def _volume_ratio(a: ChangeCluster, b: ChangeCluster) -> float:
    hi = max(a.volume, b.volume)
    lo = max(min(a.volume, b.volume), 1e-9)
    return hi / lo


def pair_moved(
    removed: Sequence[ChangeCluster],
    added: Sequence[ChangeCluster],
    *,
    max_dist_m: float = 1.5,
    max_volume_ratio: float = 3.0,
    appearance_sim: Optional[np.ndarray] = None,
    min_appearance: float = 0.0,
) -> tuple[list[ChangeCluster], list[ChangeCluster], list[ChangeCluster]]:
    """Greedily match removals to additions.

    ``appearance_sim`` is an optional ``(len(removed), len(added))`` similarity
    matrix in ``[0, 1]`` (e.g. cosine similarity of CLIP crops).  Matches below
    ``min_appearance`` are rejected.

    Returns ``(removed_remaining, added_remaining, moved)``.
    """
    if not removed or not added:
        return list(removed), list(added), []

    big = 1e6
    cost = np.full((len(removed), len(added)), big, dtype=np.float64)
    for i, r in enumerate(removed):
        for j, a in enumerate(added):
            dist = float(np.linalg.norm(r.centroid - a.centroid))
            if dist > max_dist_m or _volume_ratio(r, a) > max_volume_ratio:
                continue
            if appearance_sim is not None and appearance_sim[i, j] < min_appearance:
                continue
            # normalise distance and bias toward appearance similarity when present
            c = dist / max(max_dist_m, 1e-9)
            if appearance_sim is not None:
                c += 0.5 * (1.0 - float(appearance_sim[i, j]))
            cost[i, j] = c

    rows, cols = linear_sum_assignment(cost)
    used_removed: set[int] = set()
    used_added: set[int] = set()
    moved: list[ChangeCluster] = []

    for i, j in zip(rows, cols):
        if cost[i, j] >= big / 2:
            continue
        r, a = removed[i], added[j]
        used_removed.add(i)
        used_added.add(j)
        idx = np.concatenate([r.voxel_indices, a.voxel_indices], axis=0)
        color = None
        if r.color is not None and a.color is not None:
            color = 0.5 * (np.asarray(r.color) + np.asarray(a.color))
        moved.append(
            ChangeCluster(
                id=-1,
                kind="moved",
                centroid=0.5 * (r.centroid + a.centroid),
                bbox_min=np.minimum(r.bbox_min, a.bbox_min),
                bbox_max=np.maximum(r.bbox_max, a.bbox_max),
                n_voxels=int(r.n_voxels + a.n_voxels),
                volume=float(r.volume + a.volume),
                voxel_indices=idx,
                score=float(cost[i, j]),
                color=color,
                object_extent=np.maximum(r.extent, a.extent),
                move_from=r.centroid.copy(),
                move_to=a.centroid.copy(),
                pair_id=None,
            )
        )

    removed_remaining = [c for i, c in enumerate(removed) if i not in used_removed]
    added_remaining = [c for i, c in enumerate(added) if i not in used_added]
    return removed_remaining, added_remaining, moved
