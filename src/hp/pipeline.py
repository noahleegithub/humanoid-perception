"""End-to-end pipeline: two recordings in, a labelled 3D change set out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from hp.align import Sim3, align_scene, point_cloud_from_scene, register_scenes
from hp.change import detect_changes, fit_grid, integrate_scene, scene_world_points
from hp.config import Config, load_config
from hp.types import ChangeCluster, ChangeSet, LabeledChange, Scene, VoxelGrid

__all__ = ["PipelineResult", "compare_scenes", "run"]


@dataclass
class PipelineResult:
    changeset: ChangeSet
    scene_a: Scene
    scene_b: Scene
    grid_a: VoxelGrid
    grid_b: VoxelGrid
    out_dir: Path
    outputs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# scene construction
# ---------------------------------------------------------------------------
def _load_synthetic(dataset_dir: Path) -> tuple[Scene, Scene]:
    from hp.recon.gt_recon import load_synthetic_pair

    return load_synthetic_pair(dataset_dir)


def _load_videos(video_a: str, video_b: str, cfg: Config) -> tuple[Scene, Scene]:
    from hp.io.keyframes import select_keyframes
    from hp.io.video import extract_frames
    from hp.recon import get_reconstructor

    backend = cfg.recon.get_path("backend", "vggt")
    kwargs = {}
    if backend == "vggt":
        kwargs = {
            "device": cfg.recon.get_path("device", "auto"),
            "frames_chunk_size": cfg.recon.get_path("frames_chunk_size", 6),
            "vram_efficient": cfg.recon.get_path("vram_efficient", True),
        }
    recon = get_reconstructor(backend, **kwargs)

    scenes = []
    for name, path in (("A", video_a), ("B", video_b)):
        frames = extract_frames(
            path,
            stride_s=cfg.io.get_path("frame_stride_s", 0.5),
            long_side=cfg.io.get_path("long_side", 518),
        )
        keyframes = select_keyframes(
            frames,
            max_keyframes=cfg.io.get_path("max_keyframes", 12),
            min_sharpness_pct=cfg.io.get_path("min_sharpness_pct", 20.0),
        )
        scenes.append(recon.reconstruct(keyframes, f"scene_{name}"))
    return scenes[0], scenes[1]


# ---------------------------------------------------------------------------
# fusion + detection
# ---------------------------------------------------------------------------
def _fuse(scene_a: Scene, scene_b: Scene, cfg: Config) -> tuple[VoxelGrid, VoxelGrid]:
    voxel = float(cfg.change.get_path("voxel_size", 0.06))
    stride = int(cfg.change.get_path("pixel_stride", 3)) if cfg.change.get_path("pixel_stride") else 3
    trunc = float(cfg.change.get_path("trunc_m", voxel))
    max_ray = float(cfg.change.get_path("max_ray_m", 9.0))

    pts_a = scene_world_points(scene_a, pixel_stride=6)
    pts_b = scene_world_points(scene_b, pixel_stride=6)
    origin, dims = fit_grid([pts_a, pts_b], voxel, pad=0.3)

    grid_a = VoxelGrid.empty(voxel, origin, dims)
    grid_b = VoxelGrid.empty(voxel, origin, dims)
    integrate_scene(scene_a, grid_a, pixel_stride=stride, trunc=trunc, max_ray=max_ray)
    integrate_scene(scene_b, grid_b, pixel_stride=stride, trunc=trunc, max_ray=max_ray)
    return grid_a, grid_b


def _static_points(scene: Scene, clusters: list[ChangeCluster], pixel_stride: int = 6) -> np.ndarray:
    """Points from ``scene`` that fall outside every detected cluster bbox."""
    pts = point_cloud_from_scene(scene, pixel_stride=pixel_stride)
    if not clusters or len(pts) == 0:
        return pts
    keep = np.ones(len(pts), dtype=bool)
    margin = 0.1
    for c in clusters:
        lo = c.bbox_min - margin
        hi = c.bbox_max + margin
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        keep &= ~inside
    return pts[keep]


def _scene_bounds(scene: Scene):
    pts = scene_world_points(scene, pixel_stride=8)
    if len(pts) == 0:
        return None
    return pts.min(axis=0), pts.max(axis=0)


def _label_fallback(clusters: list[ChangeCluster], scene_a: Scene, scene_b: Scene) -> list[LabeledChange]:
    """Label changes without SAM 3.1 (used when semantics is disabled/unavailable)."""
    from hp.semantics.caption import describe

    bounds_a = _scene_bounds(scene_a)
    bounds_b = _scene_bounds(scene_b)
    out: list[LabeledChange] = []
    for c in clusters:
        bounds = bounds_a if c.kind == "removed" else bounds_b
        evidence = []
        if bounds is not None:
            evidence.append(
                {"scene_bounds": {"min": bounds[0].tolist(), "max": bounds[1].tolist()}}
            )
        lc = LabeledChange(cluster=c, label="object", confidence=0.0, description="", evidence=evidence)
        lc.description = describe(lc)
        out.append(lc)
    return out


def _label(clusters: list[ChangeCluster], scene_a: Scene, scene_b: Scene, cfg: Config) -> list[LabeledChange]:
    if not cfg.semantics.get_path("enabled", True) or not clusters:
        return _label_fallback(clusters, scene_a, scene_b)
    try:
        from hp.semantics.sam3 import Sam3Labeler

        labeler = Sam3Labeler(
            model_id=cfg.semantics.get_path("sam3_model", "facebook/sam3.1"),
            device=cfg.semantics.get_path("device", "auto"),
            min_score=float(cfg.semantics.get_path("min_score", 0.4)),
            use_tracker=bool(cfg.semantics.get_path("use_tracker", False)),
        )
        return labeler.label_changes(
            clusters, scene_a, scene_b, top_k_frames=int(cfg.semantics.get_path("top_k_frames", 3))
        )
    except Exception as exc:  # pragma: no cover - depends on GPU/model availability
        import warnings

        warnings.warn(f"SAM 3.1 labelling unavailable ({exc}); falling back to template labels.")
        return _label_fallback(clusters, scene_a, scene_b)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def compare_scenes(
    scene_a: Scene,
    scene_b: Scene,
    cfg: Optional[Config] = None,
    *,
    align_method: str = "auto",
    correspondences: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[ChangeSet, VoxelGrid, VoxelGrid, Scene, Scene]:
    """Compare two already-reconstructed scenes; returns the change set and stabilised scenes.

    ``scene_a`` (before) is aligned onto ``scene_b`` (after).  Returns
    ``(changeset, grid_a, grid_b, scene_a_aligned, scene_b)``.
    """
    cfg = cfg or load_config()
    min_occ = int(cfg.change.get_path("min_occ_obs", 2))
    min_free = int(cfg.change.get_path("min_free_obs", 3))
    band = float(cfg.change.get_path("surface_band_m", 0.16))
    min_vox = int(cfg.change.get_path("min_cluster_voxels", 60))
    moved_dist = float(cfg.change.get_path("moved_max_dist_m", 1.6))
    moved_ratio = float(cfg.change.get_path("moved_max_volume_ratio", 2.5))
    min_app = float(cfg.change.get_path("min_appearance", 0.55))
    iterations = int(cfg.align.get_path("iterations", 2))

    sim3, align_info = register_scenes(
        scene_a, scene_b, method=align_method, correspondences=correspondences
    )

    a_aligned = align_scene(scene_a, sim3)
    clusters: list[ChangeCluster] = []
    grid_a = grid_b = None

    for it in range(max(1, iterations)):
        grid_a, grid_b = _fuse(a_aligned, scene_b, cfg)
        clusters = detect_changes(
            grid_a,
            grid_b,
            min_occ=min_occ,
            min_free=min_free,
            surface_band_m=band,
            min_cluster_voxels=min_vox,
            moved_max_dist_m=moved_dist,
            moved_max_volume_ratio=moved_ratio,
            min_appearance=min_app,
            scene_a=a_aligned,
            scene_b=scene_b,
        )

        # re-align on static geometry only, then re-detect
        if it < iterations - 1 and align_info.get("method") not in {"identity"} and clusters:
            static_a = _static_points(a_aligned, clusters)
            static_b = point_cloud_from_scene(scene_b)
            if len(static_a) > 50 and len(static_b) > 50:
                from hp.align import register_open3d

                refine = register_open3d(static_a, static_b, seed=0)
                candidate = sim3.compose(refine) if refine.scale > 0 else sim3
                if np.isfinite(candidate.scale) and candidate.scale > 0:
                    sim3 = candidate
                    align_info["iterations"] = it + 1
                    a_aligned = align_scene(scene_a, sim3)

    changeset = ChangeSet(
        src_name=scene_a.name,
        dst_name=scene_b.name,
        voxel_size=float(grid_a.voxel_size),
        clusters=clusters,
        sim3=sim3.matrix(),
        scale=sim3.scale,
        align_info=align_info,
        stats={
            "voxel_size": float(grid_a.voxel_size),
            "grid_dims": list(grid_a.dims),
            "n_clusters": len(clusters),
        },
    )
    changeset.labeled = _label(clusters, a_aligned, scene_b, cfg)
    return changeset, grid_a, grid_b, a_aligned, scene_b


def run(
    *,
    video_a: Optional[str] = None,
    video_b: Optional[str] = None,
    dataset: Optional[str] = None,
    cfg: Optional[Config] = None,
    out_dir: Optional[str] = None,
    align_method: str = "auto",
    config_path: Optional[str] = None,
    overrides: Optional[list[str]] = None,
) -> PipelineResult:
    """Run the full pipeline from two videos or a synthetic dataset directory."""
    cfg = cfg or load_config(config_path, overrides)
    out_path = Path(out_dir or cfg.out.get_path("dir", "outputs"))
    out_path.mkdir(parents=True, exist_ok=True)

    if dataset:
        scene_a, scene_b = _load_synthetic(Path(dataset))
    elif video_a and video_b:
        scene_a, scene_b = _load_videos(video_a, video_b, cfg)
    else:
        raise ValueError("provide either --dataset or both --video-a and --video-b")

    changeset, grid_a, grid_b, a_aligned, b_aligned = compare_scenes(
        scene_a, scene_b, cfg, align_method=align_method
    )

    outputs: dict = {}
    # visualisation + report
    try:
        from hp.viz.plotly_viz import change_figure, save_figure_html
        from hp.viz.report import write_report

        pts = np.concatenate(
            [scene_world_points(a_aligned, pixel_stride=4), scene_world_points(b_aligned, pixel_stride=4)],
            axis=0,
        )
        fmt = cfg.viz.get_path("backend", "plotly")
        if fmt == "plotly":
            fig = change_figure(
                pts,
                changeset,
                grid=grid_b,
                point_size=float(cfg.viz.get_path("point_size", 1.2)),
                max_points=int(cfg.viz.get_path("max_points", 400000)),
                colors=cfg.viz.get_path("colors"),
            )
            outputs["figure_html"] = str(save_figure_html(fig, out_path / "changes.html"))
        elif fmt == "open3d":
            from hp.viz.open3d_viz import show_changes

            show_changes(pts, changeset, grid=grid_b)
        if cfg.out.get_path("save_report", True):
            outputs.update(write_report(changeset, out_path, config=dict(cfg)))
    except Exception as exc:  # pragma: no cover - visualisation is best-effort
        import warnings

        warnings.warn(f"visualisation/report failed: {exc}")

    return PipelineResult(
        changeset=changeset,
        scene_a=a_aligned,
        scene_b=b_aligned,
        grid_a=grid_a,
        grid_b=grid_b,
        out_dir=out_path,
        outputs=outputs,
    )
