"""Visualisation and reporting for detected 3D changes.

Two rendering backends are provided:

* :mod:`hp.viz.plotly_viz` -- interactive ``Scatter3d`` figures that render
  inline in Google Colab (no GPU, no display required).  This is the default.
* :mod:`hp.viz.open3d_viz` -- an interactive Open3D window for local machines
  with a display, plus a dependency-free coloured PLY writer.

Open3D is optional and imported lazily, so ``import hp.viz`` works everywhere.
"""

from hp.viz.open3d_viz import save_ply, show_changes
from hp.viz.plotly_viz import DEFAULT_COLORS, change_figure, cluster_points, save_figure_html
from hp.viz.report import changes_to_markdown, write_report

__all__ = [
    "DEFAULT_COLORS",
    "change_figure",
    "changes_to_markdown",
    "cluster_points",
    "save_figure_html",
    "save_ply",
    "show_changes",
    "write_report",
]
