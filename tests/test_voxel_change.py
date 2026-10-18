"""Unit tests for the visibility-aware voxel change engine."""

import numpy as np

from hp.change import diff_grids
from hp.types import VoxelGrid, VoxelState


def _grid(voxel: float = 0.1, dims=(6, 6, 6)) -> VoxelGrid:
    return VoxelGrid.empty(voxel, np.zeros(3), dims)


def test_unknown_voxels_are_never_reported():
    """A voxel unobserved in one scan must not count as a change."""
    a = _grid()
    b = _grid()
    # A sees a surface at (2,2,2); B never observes that voxel at all
    a.occ[2, 2, 2] = 5
    added, removed = diff_grids(a, b, min_occ=2, min_free=2, surface_band_m=0.0)
    assert not added.any()
    assert not removed.any()


def test_added_and_removed_semantics():
    a = _grid()
    b = _grid()
    # object present in A only -> removed
    a.occ[1, 1, 1] = 5
    b.free[1, 1, 1] = 5
    # object present in B only -> added
    b.occ[4, 4, 4] = 5
    a.free[4, 4, 4] = 5

    added, removed = diff_grids(a, b, min_occ=2, min_free=2, surface_band_m=0.0)
    assert removed[1, 1, 1]
    assert not removed[4, 4, 4]
    assert added[4, 4, 4]
    assert not added[1, 1, 1]
    assert added.sum() == 1 and removed.sum() == 1


def test_surface_band_suppresses_nearby_surface_shift():
    """A surface that merely shifted within the band is not a change."""
    a = _grid()
    b = _grid()
    a.occ[2, 2, 2] = 5
    b.occ[3, 2, 2] = 5          # same wall, one voxel over
    b.free[2, 2, 2] = 5
    a.free[3, 2, 2] = 5

    without_band, _ = diff_grids(a, b, min_occ=2, min_free=2, surface_band_m=0.0)
    with_band, _ = diff_grids(a, b, min_occ=2, min_free=2, surface_band_m=0.2)
    assert without_band[2, 2, 2] or without_band[3, 2, 2]
    assert not with_band.any()


def test_voxel_state_precedence():
    g = _grid()
    g.free[0, 0, 0] = 10
    g.occ[0, 0, 0] = 3
    state = g.state(min_occ=2, min_free=2)
    assert state[0, 0, 0] == VoxelState.OCCUPIED     # surface wins over free
    assert state[1, 1, 1] == VoxelState.UNKNOWN      # untouched


def test_world_index_roundtrip():
    g = _grid(voxel=0.05, dims=(10, 10, 10))
    g.origin = np.array([-0.2, 0.0, 0.1])
    idx = np.array([[0, 0, 0], [3, 4, 5]])
    world = g.index_to_world(idx)
    back = g.world_to_index(world)
    assert np.array_equal(back, idx)
    assert g.in_bounds(np.array([[0, 0, 0], [20, 0, 0]])).tolist() == [True, False]
