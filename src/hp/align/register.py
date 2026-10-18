"""Align two independently reconstructed scenes into a shared coordinate frame.

The "before" scan is transformed onto the "after" scan.  Two families of method
are supported:

* ``identity`` -- both scenes already share a frame (e.g. ground-truth replay).
* ``open3d``   -- global feature-based registration (FPFH + RANSAC + ICP)
  followed by a closed-form scale/anchor refinement, for real, independent
  reconstructions.  Requires ``open3d`` locally.

Alignment must be driven by *static* structure; the pipeline iterates
align -> detect -> re-align with the detected changes masked out.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial import cKDTree

from hp.align.sim3 import Sim3, ransac_sim3, umeyama
from hp.types import Scene

__all__ = [
    "align_scene",
    "point_cloud_from_scene",
    "register_from_correspondences",
    "register_open3d",
    "register_scenes",
]


def align_scene(scene: Scene, sim3: Sim3) -> Scene:
    """Return a copy of ``scene`` with frames and points transformed by ``sim3``."""
    frames = []
    for f in scene.frames:
        g = type(f)(
            index=f.index,
            rgb=f.rgb,
            depth=f.depth,
            pose=None if f.pose is None else sim3.apply_pose(f.pose),
            intrinsics=f.intrinsics,
            path=f.path,
        )
        frames.append(g)
    out = Scene(name=scene.name, frames=frames, meta=dict(scene.meta))
    if scene.points is not None:
        out.points = sim3.apply_points(scene.points)
    out.colors = scene.colors
    return out


def point_cloud_from_scene(scene: Scene, pixel_stride: int = 6) -> np.ndarray:
    """Get a world-space point cloud for a scene (from the cached cloud or frames)."""
    if scene.points is not None and len(scene.points):
        return np.asarray(scene.points, dtype=np.float64)
    from hp.change.voxel import scene_world_points

    return scene_world_points(scene, pixel_stride=pixel_stride)


def register_from_correspondences(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold: float = 0.08,
    iterations: int = 3000,
    with_scale: bool = True,
    seed: int = 0,
) -> Sim3:
    """RANSAC a Sim(3) from explicit 3D-3D correspondences."""
    return ransac_sim3(
        src, dst, threshold=threshold, iterations=iterations, with_scale=with_scale, seed=seed
    )


def _nn_umeyama(src: np.ndarray, dst: np.ndarray, max_dist: float, with_scale: bool) -> Sim3:
    """Nearest-neighbour correspondences then closed-form Sim(3)."""
    tree = cKDTree(dst)
    dist, idx = tree.query(src, k=1, distance_upper_bound=max_dist)
    valid = np.isfinite(dist)
    if valid.sum() < 3:
        return Sim3.identity()
    return umeyama(src[valid], dst[idx[valid]], with_scale=with_scale)


def register_open3d(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    voxel: float = 0.05,
    max_dist: float = 0.25,
    with_scale: bool = True,
    max_points: int = 60000,
    seed: int = 0,
) -> Sim3:
    """Global registration with Open3D (FPFH + RANSAC + ICP), then Sim(3) refine.

    Falls back to a nearest-neighbour Sim(3) if Open3D is unavailable.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) > max_points:
        src = src[np.random.default_rng(seed).choice(len(src), max_points, replace=False)]
    if len(dst) > max_points:
        dst = dst[np.random.default_rng(seed).choice(len(dst), max_points, replace=False)]

    try:
        import open3d as o3d
    except ImportError:
        return _nn_umeyama(src, dst, max_dist=max_dist, with_scale=with_scale)

    def make_pcd(pts: np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd = pcd.voxel_down_sample(voxel)
        pcd.estimate_normals()
        return pcd

    pcd_src = make_pcd(src)
    pcd_dst = make_pcd(dst)
    radius = voxel * 3.0
    feat_src = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_src, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    feat_dst = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_dst, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100)
    )
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_src,
        pcd_dst,
        feat_src,
        feat_dst,
        False,
        voxel * 2.0,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 2.0),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    transform = np.asarray(result.transformation, dtype=np.float64)

    # refine: ICP for the rigid part, then closed-form scale on NN pairs
    icp = o3d.pipelines.registration.registration_icp(
        pcd_src,
        pcd_dst,
        max_dist,
        transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    rough = np.asarray(icp.transformation, dtype=np.float64)
    rot, trans = rough[:3, :3], rough[:3, 3]
    coarse = src @ rot.T + trans
    refined = _nn_umeyama(coarse, dst, max_dist=max_dist, with_scale=with_scale)
    return refined.compose(Sim3(1.0, rot, trans))


def register_scenes(
    scene_a: Scene,
    scene_b: Scene,
    *,
    method: str = "auto",
    threshold: float = 0.08,
    iterations: int = 3000,
    with_scale: bool = True,
    correspondences: Optional[tuple[np.ndarray, np.ndarray]] = None,
    pixel_stride: int = 6,
    seed: int = 0,
) -> tuple[Sim3, dict]:
    """Return ``(sim3_a_to_b, info)`` aligning ``scene_a`` onto ``scene_b``."""
    method = method.lower()
    meta_a = scene_a.meta.get("backend")
    meta_b = scene_b.meta.get("backend")

    if method == "auto":
        if meta_a == meta_b == "gt":
            method = "identity"
        else:
            method = "open3d"

    info = {"method": method}

    if method == "identity":
        return Sim3.identity(), info

    if method == "correspondences":
        if correspondences is None:
            raise ValueError("method='correspondences' needs a (src, dst) pair")
        sim = register_from_correspondences(
            correspondences[0], correspondences[1],
            threshold=threshold, iterations=iterations, with_scale=with_scale, seed=seed,
        )
        info["n_correspondences"] = int(len(correspondences[0]))
    elif method == "open3d":
        pts_a = point_cloud_from_scene(scene_a, pixel_stride=pixel_stride)
        pts_b = point_cloud_from_scene(scene_b, pixel_stride=pixel_stride)
        sim = register_open3d(
            pts_a, pts_b, max_dist=max(0.25, threshold * 3), with_scale=with_scale, seed=seed
        )
        info["n_src"] = int(len(pts_a))
        info["n_dst"] = int(len(pts_b))
    else:
        raise ValueError(f"unknown alignment method: {method!r}")

    info["scale"] = sim.scale
    if sim.inliers is not None:
        info["inlier_ratio"] = float(np.mean(sim.inliers))
    return sim, info
