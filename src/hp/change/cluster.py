"""Turn a boolean voxel difference mask into discrete change objects."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from hp.types import ChangeCluster, VoxelGrid

__all__ = ["label_components", "clusters_from_mask"]


def label_components(mask: np.ndarray, connectivity: int = 26) -> tuple[np.ndarray, int]:
    """Connected-component labelling of a 3D boolean mask."""
    if connectivity == 6:
        structure = ndimage.generate_binary_structure(3, 1)
    else:
        structure = np.ones((3, 3, 3), dtype=bool)
    labels, count = ndimage.label(mask, structure=structure)
    return labels, int(count)


def clusters_from_mask(
    mask: np.ndarray,
    grid: VoxelGrid,
    kind: str,
    min_voxels: int,
    start_id: int = 0,
) -> tuple[list[ChangeCluster], int]:
    """Build :class:`ChangeCluster` objects from a boolean mask.

    Clusters smaller than ``min_voxels`` are discarded as noise.
    """
    labels, count = label_components(mask)
    voxel = grid.voxel_size
    origin = grid.origin
    clusters: list[ChangeCluster] = []

    # bincount over label ids is much faster than looping over argwhere
    flat_labels = labels.reshape(-1)
    sizes = np.bincount(flat_labels, minlength=count + 1)
    keep = [lab for lab in range(1, count + 1) if sizes[lab] >= min_voxels]

    idx_by_label = {lab: np.argwhere(labels == lab) for lab in keep}
    for lab in keep:
        idx = idx_by_label[lab]
        imin = idx.min(axis=0)
        imax = idx.max(axis=0)
        world = origin + (idx + 0.5) * voxel
        clusters.append(
            ChangeCluster(
                id=-1,
                kind=kind,
                centroid=world.mean(axis=0),
                bbox_min=origin + imin * voxel,
                bbox_max=origin + (imax + 1) * voxel,
                n_voxels=int(idx.shape[0]),
                volume=float(idx.shape[0] * voxel**3),
                voxel_indices=idx,
                score=float(idx.shape[0]),
            )
        )

    clusters.sort(key=lambda c: -c.volume)
    for offset, cluster in enumerate(clusters):
        cluster.id = start_id + offset
    return clusters, start_id + len(clusters)
