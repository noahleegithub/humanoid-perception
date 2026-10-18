"""Grounded natural-language descriptions of labelled 3D changes.

The pipeline produces :class:`~hp.types.LabeledChange` objects whose ``label``
comes from SAM 3.1.  This module turns those into short English sentences that
mention what changed, roughly how big it is, its colour and where in the room it
happened.  Everything is template-based and deterministic (``captioner:
template`` in the configuration); no language model is required.
"""

from __future__ import annotations

from collections import Counter
from typing import Optional, Sequence

import numpy as np

from hp.semantics.labels import color_name
from hp.types import ChangeCluster, LabeledChange

__all__ = ["describe", "summarise"]


def _extent_cm(cluster: ChangeCluster) -> tuple[int, int, int]:
    """Object side lengths of a cluster in centimetres (bbox for non-moved)."""
    extent = np.asarray(cluster.size, dtype=np.float64).reshape(3) * 100.0
    return int(round(float(extent[0]))), int(round(float(extent[1]))), int(round(float(extent[2])))


def _size_phrase(cluster: ChangeCluster) -> str:
    dx, dy, dz = _extent_cm(cluster)
    return f"{dx} x {dy} x {dz} cm"


def _bounds_from_change(change: LabeledChange) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Recover the scene point-cloud bounds stashed in the change's evidence.

    :meth:`hp.semantics.sam3.Sam3Labeler.label_changes` records the bounds of
    the scene each change was observed in, so a description can place it
    relative to the room.  Returns ``None`` when no bounds were recorded.
    """
    for entry in change.evidence or []:
        if not isinstance(entry, dict):
            continue
        bounds = entry.get("scene_bounds")
        if not bounds:
            continue
        try:
            lo = np.asarray(bounds["min"], dtype=np.float64).reshape(3)
            hi = np.asarray(bounds["max"], dtype=np.float64).reshape(3)
        except (KeyError, TypeError, ValueError):
            continue
        if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
            return lo, hi
    return None


def _wall_place(nx: float, ny: float) -> Optional[str]:
    """Named wall or corner when a point sits near the room's edge, else ``None``."""
    left, right = nx < 0.28, nx > 0.72
    back, front = ny < 0.28, ny > 0.72
    if left and back:
        return "the back-left corner"
    if right and back:
        return "the back-right corner"
    if left and front:
        return "the front-left corner"
    if right and front:
        return "the front-right corner"
    if left:
        return "the left wall"
    if right:
        return "the right wall"
    if back:
        return "the back wall"
    if front:
        return "the front wall"
    return None


def _place(point: Sequence[float], bounds: Optional[tuple[np.ndarray, np.ndarray]]) -> str:
    """A place noun phrase for a world point, e.g. ``"the floor near the left wall"``.

    Horizontal placement is relative to the room's point-cloud bounds when they
    are known; without them we fall back to the height above the floor implied
    by the world ``z`` axis (metres, ``+z`` up).
    """
    p = np.asarray(point, dtype=np.float64).reshape(3)
    if bounds is None:
        return "the floor" if p[2] < 0.15 else "the room"

    lo, hi = bounds
    span = np.maximum(hi - lo, 1e-6)
    nx = float(np.clip((p[0] - lo[0]) / span[0], 0.0, 1.0))
    ny = float(np.clip((p[1] - lo[1]) / span[1], 0.0, 1.0))
    nz = float(np.clip((p[2] - lo[2]) / span[2], 0.0, 1.0))
    wall = _wall_place(nx, ny)
    height_m = float(p[2] - lo[2])

    if wall is not None:
        joiner = "in" if wall.endswith("corner") else "near"
        if nz < 0.25:
            return f"the floor {joiner} {wall}"
        if nz > 0.85:
            return f"the ceiling {joiner} {wall}"
        return f"{wall} about {height_m:.1f} m above the floor"

    if nz < 0.25:
        return "the centre of the room"
    if nz > 0.85:
        return "the centre of the room near the ceiling"
    return f"the centre of the room about {height_m:.1f} m above the floor"


def _locative(place: str) -> str:
    """Prepositional form of a :func:`_place` phrase (e.g. ``"on the floor ..."``)."""
    if place.startswith(("the floor", "the ceiling")):
        return f"on {place}"
    if place.startswith("the room"):
        return "in the room"
    if place.endswith("corner"):
        return f"in {place}"
    if "wall" in place:
        return f"near {place}"
    return f"in {place}"


def _noun(change: LabeledChange) -> str:
    """Object noun phrase, prefixed by a coarse colour when one is known."""
    label = (change.label or "object").strip() or "object"
    colour = color_name(change.cluster.color)
    return f"{colour} {label}" if colour else label


def describe(change: LabeledChange) -> str:
    """Return one grounded sentence describing a labelled change.

    The sentence names the change type and object, gives an approximate size in
    centimetres, its colour (when sampled) and a coarse position in the room.
    For ``"moved"`` changes it reports the travel distance and the source and
    destination anchors.
    """
    cluster = change.cluster
    noun = _noun(change)
    size = _size_phrase(cluster)
    bounds = _bounds_from_change(change)
    place = _place(cluster.centroid, bounds)

    if cluster.kind == "removed":
        return f"A {size} {noun} was removed from {place}."

    if cluster.kind == "moved":
        source = cluster.move_from if cluster.move_from is not None else cluster.centroid
        target = cluster.move_to if cluster.move_to is not None else cluster.centroid
        distance = float(np.linalg.norm(np.asarray(target) - np.asarray(source)))
        from_place = _place(source, bounds)
        to_place = _place(target, bounds)
        return (
            f"The {size} {noun} was moved about {distance:.1f} m "
            f"from {from_place} towards {to_place}."
        )

    if cluster.kind == "added":
        return f"A {size} {noun} appeared {_locative(place)}."

    return f"A {size} {noun} changed {_locative(place)}."


def summarise(changes: Sequence[LabeledChange]) -> str:
    """Short paragraph: overall counts followed by one sentence per change."""
    changes = list(changes)
    if not changes:
        return "No changes were detected between the two scans."

    counts = Counter(change.cluster.kind for change in changes)
    total = len(changes)
    plural = "s" if total != 1 else ""
    header = (
        f"Detected {total} change{plural} "
        f"({counts.get('added', 0)} added, {counts.get('removed', 0)} removed, "
        f"{counts.get('moved', 0)} moved)."
    )
    sentences = [describe(change) for change in changes]
    return " ".join([header, *sentences])
