"""Plotly visualisation of detected changes (renders inline in Colab).

The fused/aligned scene is drawn as a grey "unchanged" backdrop while the
voxels belonging to each change cluster are coloured by kind (added=green,
removed=red, moved=blue).  No GPU and no display are needed -- the figure is a
plain :class:`plotly.graph_objects.Figure` that can be shown inline or written
to a standalone HTML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from hp.types import ChangeCluster, ChangeSet, VoxelGrid

if TYPE_CHECKING:  # pragma: no cover - typing only
    import plotly.graph_objects as go

__all__ = ["DEFAULT_COLORS", "change_figure", "cluster_points", "save_figure_html"]

#: Default RGB colours (0-255) keyed by voxel kind.
DEFAULT_COLORS: dict[str, Sequence[int]] = {
    "unchanged": (150, 150, 150),
    "added": (40, 200, 80),
    "removed": (220, 50, 50),
    "moved": (60, 120, 240),
}

_KINDS = ("added", "removed", "moved")


def _as_rgb(color: object) -> str:
    """Return a plotly colour string for an ``(r, g, b)`` value or a string."""
    if isinstance(color, str):
        return color
    comps = np.asarray(color, dtype=np.float64).reshape(-1)
    if comps.size < 3:
        raise ValueError(f"colour needs 3 components, got {color!r}")
    r, g, b = (int(round(float(np.clip(v, 0.0, 255.0)))) for v in comps[:3])
    return f"rgb({r},{g},{b})"


def _color_map(colors: Optional[dict]) -> dict[str, str]:
    """Merge user colours over :data:`DEFAULT_COLORS`, normalised to strings."""
    merged: dict[str, object] = dict(DEFAULT_COLORS)
    if colors:
        merged.update(colors)
    return {key: _as_rgb(value) for key, value in merged.items()}


def _downsample(
    points: np.ndarray, limit: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Return at most ``limit`` rows of ``points`` (random, seeded by default)."""
    n = int(len(points))
    if limit <= 0:
        return points[:0]
    if n <= limit:
        return points
    rng = np.random.default_rng(0) if rng is None else rng
    sel = rng.choice(n, size=int(limit), replace=False)
    sel.sort()
    return points[sel]


def cluster_points(
    cluster: ChangeCluster,
    grid: Optional[VoxelGrid] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    max_points: Optional[int] = None,
) -> np.ndarray:
    """Return world-space points that represent ``cluster``.

    When ``grid`` is provided the cluster's own voxel centres are used
    (``grid.index_to_world(cluster.voxel_indices)``), which reproduces the true
    shape of the region.

    ``ChangeCluster`` does not store the grid transform, so when ``grid`` is
    ``None`` we fall back to distributing ``n_voxels`` points uniformly inside
    the cluster bounding box ``bbox_min..bbox_max``.  That fallback is only an
    approximation: it fills the axis-aligned box rather than the voxel shape.
    """
    idx = getattr(cluster, "voxel_indices", None)
    if grid is not None and idx is not None and len(idx) > 0:
        pts = grid.index_to_world(np.asarray(idx))
    else:
        rng = np.random.default_rng(0) if rng is None else rng
        lo = np.minimum(np.asarray(cluster.bbox_min, dtype=np.float64),
                        np.asarray(cluster.bbox_max, dtype=np.float64)).reshape(3)
        hi = np.maximum(np.asarray(cluster.bbox_min, dtype=np.float64),
                        np.asarray(cluster.bbox_max, dtype=np.float64)).reshape(3)
        n = max(int(cluster.n_voxels), 1)
        pts = rng.uniform(lo, hi, size=(n, 3))

    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
    if max_points is not None and len(pts) > int(max_points):
        pts = _downsample(pts, int(max_points), rng)
    return pts


def change_figure(
    scene_points: np.ndarray,
    changeset: ChangeSet,
    *,
    grid: Optional[VoxelGrid] = None,
    point_size: float = 1.2,
    max_points: int = 400000,
    colors: Optional[dict] = None,
) -> "go.Figure":
    """Build an interactive :class:`plotly.graph_objects.Figure` of the changes.

    Parameters
    ----------
    scene_points:
        ``(N, 3)`` fused/aligned scene, drawn as the grey "unchanged" backdrop.
    changeset:
        Detected changes; every cluster is coloured by its ``kind``.
    grid:
        Voxel grid used to resolve each cluster's voxel centres.  When omitted,
        cluster points are approximated by their bounding boxes (see
        :func:`cluster_points`).
    point_size:
        Marker size in pixels for backdrop points; change points are drawn 1.4x
        larger so they stand out.
    max_points:
        Total point budget for the figure.  Changes get first refusal (up to
        half the budget) and the backdrop receives what is left.  Values ``<= 0``
        disable downsampling.
    colors:
        Optional overrides for :data:`DEFAULT_COLORS`, e.g.
        ``{"added": (40, 200, 80)}`` or ``{"added": "#28c850"}``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import plotly.graph_objects as go

    palette = _color_map(colors)
    rng = np.random.default_rng(0)

    # ---- gather change points per kind (changes are budgeted before backdrop)
    per_kind: dict[str, list[np.ndarray]] = {kind: [] for kind in _KINDS}
    for cluster in changeset.clusters:
        pts = cluster_points(cluster, grid, rng=rng)
        per_kind.setdefault(cluster.kind, []).append(pts)

    kind_pts: dict[str, np.ndarray] = {}
    for kind, chunks in per_kind.items():
        chunks = [c for c in chunks if len(c) > 0]
        if chunks:
            kind_pts[kind] = np.concatenate(chunks, axis=0)

    n_change = sum(int(len(p)) for p in kind_pts.values())
    if max_points > 0 and n_change > max_points // 2:
        change_limit = max(max_points // 2, 1)
        for kind in list(kind_pts):
            quota = max(int(round(len(kind_pts[kind]) * change_limit / n_change)), 1)
            kind_pts[kind] = _downsample(kind_pts[kind], quota, rng)

    # ---- backdrop, shrunk to whatever budget the changes left over
    if scene_points is None:
        scene = np.empty((0, 3), dtype=np.float64)
    else:
        scene = np.asarray(scene_points, dtype=np.float64).reshape(-1, 3)
    if max_points > 0:
        shown = sum(int(len(p)) for p in kind_pts.values())
        scene = _downsample(scene, max(max_points - shown, 0), rng)

    # ---- traces
    traces: list[object] = []
    if len(scene):
        traces.append(
            go.Scatter3d(
                x=scene[:, 0],
                y=scene[:, 1],
                z=scene[:, 2],
                mode="markers",
                name=f"unchanged ({len(scene):,})",
                marker=dict(size=point_size, color=palette["unchanged"], opacity=0.5),
                hoverinfo="name",
            )
        )

    ordered = [k for k in _KINDS if k in kind_pts]
    ordered += sorted(k for k in kind_pts if k not in _KINDS)
    for kind in ordered:
        pts = kind_pts[kind]
        if not len(pts):
            continue
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="markers",
                name=f"{kind} ({len(pts):,})",
                marker=dict(
                    size=point_size * 1.4,
                    color=palette.get(kind, _as_rgb(DEFAULT_COLORS["added"])),
                ),
                hoverinfo="name",
            )
        )

    # ---- motion arrows for paired "moved" clusters
    seg_x: list[object] = []
    seg_y: list[object] = []
    seg_z: list[object] = []
    for cluster in changeset.clusters:
        if cluster.kind != "moved" or cluster.move_from is None or cluster.move_to is None:
            continue
        a = np.asarray(cluster.move_from, dtype=np.float64).reshape(3)
        b = np.asarray(cluster.move_to, dtype=np.float64).reshape(3)
        seg_x += [float(a[0]), float(b[0]), None]
        seg_y += [float(a[1]), float(b[1]), None]
        seg_z += [float(a[2]), float(b[2]), None]
    if seg_x:
        traces.append(
            go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode="lines",
                name="moved (from → to)",
                line=dict(color=palette["moved"], width=4),
                hoverinfo="name",
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Changes: {changeset.src_name} → {changeset.dst_name}",
        scene=dict(
            aspectmode="data",
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="z (m)",
            bgcolor="white",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def save_figure_html(fig: "go.Figure", path: str | Path) -> Path:
    """Write ``fig`` to a standalone HTML file and return the resolved path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    return path
