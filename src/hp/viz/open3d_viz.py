"""Open3D visualisation helpers (local, interactive display).

Open3D is an optional dependency (``pip install hp[viz]``).  It is imported
lazily inside the functions that need it so that ``import hp.viz`` -- and the
plotly backend -- keep working on machines and Colab runtimes without Open3D.
:func:`save_ply` in particular falls back to a small, dependency-free PLY
writer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from hp.types import ChangeSet, VoxelGrid
from hp.viz.plotly_viz import DEFAULT_COLORS, cluster_points

__all__ = ["save_ply", "show_changes"]

_NO_OPEN3D = (
    "open3d is required for interactive 3D visualisation. Install it with "
    "`pip install open3d` (or `pip install hp[viz]`), or use the plotly "
    "backend, which needs neither a GPU nor a display."
)


def _require_open3d():  # noqa: ANN202 - return type is the optional module
    """Import and return the ``open3d`` module, or raise a clear ImportError."""
    try:
        import open3d as o3d
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(_NO_OPEN3D) from exc
    return o3d


def _rgb_unit(color: object) -> np.ndarray:
    """Return an ``(3,)`` float array in ``[0, 1]`` from an ``(r, g, b)`` 0-255 value."""
    comps = np.asarray(color, dtype=np.float64).reshape(-1)
    if comps.size < 3:
        raise ValueError(f"colour needs 3 components, got {color!r}")
    return np.clip(comps[:3], 0.0, 255.0) / 255.0


def _color_map(colors: Optional[dict] = None) -> dict[str, np.ndarray]:
    """Per-kind unit-RGB colours, with user ``colors`` merged over the defaults."""
    merged: dict[str, object] = dict(DEFAULT_COLORS)
    if colors:
        merged.update(colors)
    return {key: _rgb_unit(value) for key, value in merged.items()}


def _as_uint8_colors(colors: object, n: int) -> np.ndarray:
    """Coerce ``colors`` into an ``(n, 3)`` uint8 array."""
    if colors is None:
        base = np.full((n, 3), 200, dtype=np.uint8)
        return base
    arr = np.asarray(colors)
    if arr.ndim == 1 and arr.size == 3:
        arr = np.tile(arr.reshape(1, 3), (n, 1))
    arr = arr.reshape(-1, 3)
    if arr.shape[0] != n:
        raise ValueError(f"colors has {arr.shape[0]} rows but there are {n} points")
    if np.issubdtype(arr.dtype, np.floating) and arr.size and float(np.max(arr)) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _point_cloud(o3d, points: np.ndarray, color: np.ndarray):
    """Build an Open3D ``PointCloud`` with a uniform colour."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))
    return pcd


def show_changes(
    scene_points: np.ndarray,
    changeset: ChangeSet,
    *,
    grid: Optional[VoxelGrid] = None,
    point_size: float = 1.2,
    colors: Optional[dict] = None,
) -> None:
    """Open an interactive Open3D window showing the scene and the changes.

    The backdrop is drawn in grey and every cluster is coloured by kind.  The
    grid transform is used to place cluster voxels when ``grid`` is given,
    otherwise bounding-box samples are used (see
    :func:`hp.viz.plotly_viz.cluster_points`).
    """
    o3d = _require_open3d()
    palette = _color_map(colors)
    rng = np.random.default_rng(0)

    geometries = []
    if scene_points is not None:
        scene = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
        if len(scene):
            geometries.append(_point_cloud(o3d, scene, palette["unchanged"]))

    for cluster in changeset.clusters:
        pts = cluster_points(cluster, grid, rng=rng)
        if not len(pts):
            continue
        color = palette.get(cluster.kind, palette["added"])
        geometries.append(_point_cloud(o3d, pts, color))

    title = f"Changes: {changeset.src_name} -> {changeset.dst_name}"
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1024, height=768)
    for geom in geometries:
        vis.add_geometry(geom)
    options = vis.get_render_option()
    if options is not None:
        options.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def _write_ply_binary(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Minimal binary little-endian PLY writer (xyz + uchar rgb)."""
    n = int(len(points))
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment written by hp.viz.open3d_viz\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    record = np.empty(
        n,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    record["x"] = points[:, 0]
    record["y"] = points[:, 1]
    record["z"] = points[:, 2]
    record["red"] = colors[:, 0]
    record["green"] = colors[:, 1]
    record["blue"] = colors[:, 2]
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(record.tobytes())


def save_ply(points: np.ndarray, colors: np.ndarray, path: str | Path) -> Path:
    """Write a coloured point cloud to ``path`` as a PLY file.

    Uses Open3D when it is installed; otherwise a pure-python binary PLY writer
    is used so the call works everywhere (including Colab).  ``colors`` may be
    ``(N, 3)`` or a single ``(3,)`` RGB; values in ``[0, 1]`` are scaled to
    ``[0, 255]``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cols = _as_uint8_colors(colors, len(pts))

    try:
        import open3d as o3d
    except ImportError:
        _write_ply_binary(pts, cols, path)
        return path

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(cols.astype(np.float64) / 255.0)
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    return path
