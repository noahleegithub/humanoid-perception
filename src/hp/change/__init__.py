"""3D change detection over visibility-aware voxel grids."""

from hp.change.cluster import clusters_from_mask, label_components
from hp.change.detect import detect_changes, diff_grids
from hp.change.moved import pair_moved
from hp.change.voxel import (
    fit_grid,
    frame_world_points,
    integrate_frames,
    integrate_scene,
    scene_world_points,
)

__all__ = [
    "clusters_from_mask",
    "label_components",
    "detect_changes",
    "diff_grids",
    "pair_moved",
    "fit_grid",
    "frame_world_points",
    "integrate_frames",
    "integrate_scene",
    "scene_world_points",
]
