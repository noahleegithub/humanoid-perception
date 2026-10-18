"""Human-readable Markdown and machine-readable JSON change reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from hp.types import ChangeCluster, ChangeSet

__all__ = ["changes_to_markdown", "write_report"]

_KINDS = ("added", "removed", "moved")


def _escape(text: object) -> str:
    """Make ``text`` safe to place inside a Markdown table cell."""
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _fmt_position(vec: object) -> str:
    """Render a world position as ``(x, y, z) m``."""
    comps = [float(v) for v in vec][:3]
    return "({:.2f}, {:.2f}, {:.2f}) m".format(*comps)


def _fmt_size_cm(cluster: ChangeCluster) -> str:
    """Render the bounding-box side lengths in centimetres."""
    dims = [abs(float(v)) * 100.0 for v in cluster.extent][:3]
    return "{:.0f} × {:.0f} × {:.0f}".format(*dims)


def _join(items: list[str]) -> str:
    """Join phrases as ``a, b and c``."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _summary_paragraph(changeset: ChangeSet) -> str:
    """A natural-language summary of the detected changes."""
    counts = changeset.counts()
    total = sum(counts.values())
    voxel_cm = changeset.voxel_size * 100.0
    before, after = changeset.src_name, changeset.dst_name

    if total == 0:
        return (
            f"No 3D changes were detected between **{before}** (before) and "
            f"**{after}** (after) at a {voxel_cm:.1f} cm voxel resolution."
        )

    phrases = [f"{counts.get(kind, 0)} {kind}" for kind in _KINDS if counts.get(kind, 0)]
    for kind in sorted(k for k in counts if k not in _KINDS and counts[k]):
        phrases.append(f"{counts[kind]} {kind}")

    parts = [
        f"Comparing **{before}** (before) with **{after}** (after) at a "
        f"{voxel_cm:.1f} cm voxel resolution, the pipeline detected {total} "
        f"change{'s' if total != 1 else ''}: {_join(phrases)}."
    ]

    largest = max(changeset.clusters, key=lambda c: c.volume) if changeset.clusters else None
    if largest is not None:
        label = largest.kind
        if changeset.labeled:
            for lc in changeset.labeled:
                if lc.cluster.id == largest.id:
                    label = lc.label
                    break
        parts.append(
            f"The largest single change is the **{label}** region of about "
            f"{largest.volume * 1e6:.0f} cm³ near {_fmt_position(largest.centroid)}."
        )
    return " ".join(parts)


def changes_to_markdown(changeset: ChangeSet) -> str:
    """Render a :class:`~hp.types.ChangeSet` as a Markdown report string."""
    counts = changeset.counts()
    total = sum(counts.values())
    lines: list[str] = []

    lines.append(f"# Change report: {changeset.src_name} → {changeset.dst_name}")
    lines.append("")
    lines.append(f"- **Before:** `{changeset.src_name}`")
    lines.append(f"- **After:** `{changeset.dst_name}`")
    lines.append(f"- **Voxel size:** {changeset.voxel_size * 100.0:.1f} cm")
    if changeset.scale and abs(changeset.scale - 1.0) > 1e-9:
        lines.append(f"- **Alignment scale:** {changeset.scale:.4f}")
    lines.append("")

    lines.append("## Counts")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("| --- | ---: |")
    for kind in _KINDS:
        lines.append(f"| {kind} | {counts.get(kind, 0)} |")
    for kind in sorted(k for k in counts if k not in _KINDS):
        lines.append(f"| {kind} | {counts[kind]} |")
    lines.append(f"| **total** | **{total}** |")
    lines.append("")

    lines.append("## Changes")
    lines.append("")
    lines.append("| kind | label | size (cm) | position | confidence |")
    lines.append("| --- | --- | --- | --- | ---: |")
    label_by_id = {lc.cluster.id: lc for lc in changeset.labeled}
    listed = 0
    for cluster in changeset.clusters:
        lc = label_by_id.get(cluster.id)
        label = _escape(lc.label) if lc is not None else "—"
        confidence = f"{lc.confidence:.2f}" if lc is not None else "—"
        lines.append(
            f"| {_escape(cluster.kind)} | {label} | {_fmt_size_cm(cluster)} | "
            f"{_fmt_position(cluster.centroid)} | {confidence} |"
        )
        listed += 1
    # include labeled entries whose cluster was not present in ``clusters``
    known_ids = {cluster.id for cluster in changeset.clusters}
    for lc in changeset.labeled:
        if lc.cluster.id in known_ids:
            continue
        lines.append(
            f"| {_escape(lc.cluster.kind)} | {_escape(lc.label)} | "
            f"{_fmt_size_cm(lc.cluster)} | {_fmt_position(lc.cluster.centroid)} | "
            f"{lc.confidence:.2f} |"
        )
        listed += 1
    if listed == 0:
        lines.append("| — | — | — | — | — |")
    lines.append("")

    if changeset.labeled:
        lines.append("## Summary")
        lines.append("")
        lines.append(_summary_paragraph(changeset))
        lines.append("")
        lines.append("### Descriptions")
        lines.append("")
        for i, lc in enumerate(changeset.labeled, 1):
            description = lc.description.strip() or lc.label
            lines.append(f"{i}. **{_escape(lc.label)}** "
                         f"(confidence {lc.confidence:.2f}): {description.strip()}")
        lines.append("")
    elif changeset.clusters:
        lines.append(_summary_paragraph(changeset))
        lines.append("")

    return "\n".join(lines).rstrip("\n") + "\n"


def write_report(
    changeset: ChangeSet,
    out_dir: str | Path,
    *,
    config: Optional[dict] = None,
) -> dict:
    """Write ``changes.json`` and ``report.md`` into ``out_dir``.

    ``changes.json`` is exactly :meth:`hp.types.ChangeSet.to_dict`; the Markdown
    report is produced by :func:`changes_to_markdown`.  ``config`` is accepted
    for API compatibility with the pipeline config and is currently unused.

    Returns
    -------
    dict
        ``{"json": Path, "markdown": Path}``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "changes.json"
    md_path = out_dir / "report.md"
    json_path.write_text(
        json.dumps(changeset.to_dict(), indent=2, sort_keys=False), encoding="utf-8"
    )
    md_path.write_text(changes_to_markdown(changeset), encoding="utf-8")
    return {"json": json_path, "markdown": md_path}
