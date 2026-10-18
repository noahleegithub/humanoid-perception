"""Indoor-object prompt vocabulary for SAM 3.1 concept segmentation.

SAM 3.1 segments *concepts*: it is prompted with a short noun phrase such as
``"coffee table"`` rather than a class id.  Labelling a 3D change therefore
reduces to trying a handful of plausible prompts and keeping the one whose mask
best matches the observed region.  Because the model is open-vocabulary we can
stay close to plain language, but the search is still capped by
:data:`INDOOR_LABELS` so that a change is never described with a nonsense
concept.

This module also carries a small colour vocabulary.  Colour is used two ways:
to prioritise prompts that people commonly associate with a hue (a green blob
is more likely a plant than a keyboard) and, in :mod:`hp.semantics.caption`, to
turn a mean RGB triplet into a coarse word such as ``"black"`` or ``"beige"``.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["INDOOR_LABELS", "candidate_labels", "color_name"]

# ~150 common indoor-object noun phrases.  Stored unsorted here and normalised
# (lowercased, de-duplicated, sorted) into ``INDOOR_LABELS`` below so that the
# published list is always the canonical one.
_INDOOR_LABEL_SOURCE: tuple[str, ...] = (
    "air conditioner",
    "air fryer",
    "air purifier",
    "alarm clock",
    "backpack",
    "bag",
    "ball",
    "basket",
    "bath mat",
    "bath towel",
    "bathtub",
    "bed",
    "bedside table",
    "bench",
    "blanket",
    "blender",
    "book",
    "bookshelf",
    "bottle",
    "bowl",
    "box",
    "broom",
    "brush",
    "bucket",
    "cabinet",
    "calculator",
    "camera",
    "candle",
    "can",
    "carpet",
    "cart",
    "chair",
    "charger",
    "chest of drawers",
    "clock",
    "clothes",
    "coat rack",
    "coffee maker",
    "coffee table",
    "computer",
    "container",
    "couch",
    "cup",
    "curtain",
    "cushion",
    "cutting board",
    "desk",
    "desk lamp",
    "detergent",
    "dish",
    "dishwasher",
    "door mat",
    "dresser",
    "dumbbells",
    "dustbin",
    "earphones",
    "fan",
    "floor lamp",
    "flower pot",
    "fork",
    "fridge",
    "garbage can",
    "glass",
    "gloves",
    "guitar",
    "hanger",
    "headphone",
    "heater",
    "jar",
    "jug",
    "kettle",
    "keyboard",
    "knife",
    "ladder",
    "lamp",
    "laptop",
    "laundry basket",
    "laundry detergent",
    "magazine",
    "mat",
    "mattress",
    "microwave",
    "mirror",
    "mixer",
    "monitor",
    "mouse",
    "mug",
    "nightstand",
    "notebook",
    "ottoman",
    "oven",
    "pan",
    "paper",
    "paper towel",
    "pen",
    "pencil",
    "picture frame",
    "pillow",
    "plant",
    "plate",
    "poster",
    "pot",
    "printer",
    "projector",
    "radiator",
    "remote control",
    "rug",
    "router",
    "scale",
    "scissors",
    "screwdriver",
    "screen",
    "shelf",
    "shoe",
    "shoe rack",
    "sink",
    "soap",
    "sofa",
    "speaker",
    "spoon",
    "stool",
    "stove",
    "suitcase",
    "table",
    "table lamp",
    "tablet",
    "teapot",
    "telephone",
    "television",
    "tissue box",
    "toaster",
    "toilet",
    "toilet paper",
    "tool",
    "toolbox",
    "towel",
    "toy",
    "trash bag",
    "trash can",
    "tv",
    "umbrella",
    "vacuum cleaner",
    "vase",
    "wall clock",
    "wardrobe",
    "washing machine",
    "watch",
    "water bottle",
    "window",
    "wine bottle",
    "wrench",
)

INDOOR_LABELS: list[str] = sorted({phrase.lower() for phrase in _INDOOR_LABEL_SOURCE})

# Representative RGB (0-255) for each coarse colour word.
_PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "dark grey": (70, 70, 70),
    "grey": (128, 128, 128),
    "light grey": (190, 190, 190),
    "white": (240, 240, 240),
    "beige": (222, 202, 166),
    "tan": (210, 180, 140),
    "brown": (120, 72, 40),
    "red": (200, 40, 40),
    "orange": (230, 130, 30),
    "yellow": (230, 210, 40),
    "olive": (128, 128, 32),
    "green": (40, 160, 60),
    "teal": (40, 140, 140),
    "cyan": (60, 200, 210),
    "blue": (50, 90, 200),
    "navy": (30, 40, 110),
    "purple": (120, 60, 170),
    "magenta": (200, 50, 170),
    "pink": (240, 150, 190),
}

# Prompts to try first for a given colour word.  These are heuristics, not
# hard constraints: the kind-specific and generic lists follow, so a wrong
# guess merely costs one extra forward pass.
_COLOR_PRIORITY: dict[str, tuple[str, ...]] = {
    "black": ("monitor", "speaker", "keyboard", "laptop", "tv", "backpack", "chair"),
    "dark grey": ("monitor", "laptop", "speaker", "box", "keyboard"),
    "grey": ("laptop", "monitor", "box", "speaker", "shelf", "stool"),
    "light grey": ("chair", "table", "box", "basket"),
    "white": ("mug", "bottle", "paper", "kettle", "toaster", "tissue box", "plate"),
    "beige": ("couch", "sofa", "cushion", "curtain", "basket"),
    "tan": ("basket", "cushion", "bag"),
    "brown": ("box", "chair", "shelf", "bookshelf", "table", "cabinet"),
    "red": ("bottle", "box", "backpack", "cup", "chair"),
    "orange": ("bottle", "ball", "box", "toy"),
    "yellow": ("ball", "bottle", "box", "toy"),
    "olive": ("backpack", "bottle", "box"),
    "green": ("plant", "bottle", "box", "chair", "flower pot"),
    "teal": ("mug", "bottle", "cushion"),
    "cyan": ("bottle", "toy", "container"),
    "blue": ("bottle", "box", "backpack", "cup", "chair", "ball"),
    "navy": ("backpack", "bottle", "box"),
    "purple": ("backpack", "bottle", "toy", "cushion"),
    "magenta": ("toy", "box", "backpack"),
    "pink": ("toy", "backpack", "bottle", "cushion"),
}

# Generic shortlist per change kind.  All three kinds usually denote a movable
# household object, so the lists overlap heavily; the ordering is the only
# difference.
_KIND_PRIORITY: dict[str, tuple[str, ...]] = {
    "added": (
        "box",
        "chair",
        "table",
        "bottle",
        "mug",
        "backpack",
        "bag",
        "lamp",
        "plant",
        "monitor",
        "keyboard",
        "couch",
        "sofa",
        "bookshelf",
        "cabinet",
        "trash can",
        "basket",
        "shelf",
        "cushion",
        "blanket",
        "book",
        "toy",
        "container",
        "cup",
    ),
    "removed": (
        "box",
        "chair",
        "table",
        "bottle",
        "mug",
        "backpack",
        "bag",
        "lamp",
        "plant",
        "monitor",
        "keyboard",
        "couch",
        "sofa",
        "bookshelf",
        "cabinet",
        "trash can",
        "basket",
        "shelf",
        "cushion",
        "blanket",
        "book",
        "toy",
        "container",
        "cup",
    ),
    "moved": (
        "chair",
        "box",
        "table",
        "bottle",
        "mug",
        "backpack",
        "bag",
        "lamp",
        "plant",
        "stool",
        "cushion",
        "toy",
        "basket",
        "bucket",
        "monitor",
        "keyboard",
        "cup",
        "book",
        "container",
        "shelf",
    ),
}


def _as_rgb(rgb: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Coerce ``rgb`` to a length-3 float array, or ``None`` if impossible."""
    if rgb is None:
        return None
    arr = np.asarray(rgb, dtype=np.float64).reshape(-1)
    if arr.size < 3 or not np.all(np.isfinite(arr[:3])):
        return None
    return arr[:3]


def color_name(rgb: Optional[Sequence[float]]) -> Optional[str]:
    """Map a mean RGB triplet (0-255) to the nearest coarse colour word.

    Returns ``None`` when ``rgb`` is missing or malformed.  Distance is plain
    Euclidean in RGB space, which is crude but more than adequate for picking
    ``"black"`` versus ``"white"`` or ``"green"`` versus ``"brown"``.
    """
    arr = _as_rgb(rgb)
    if arr is None:
        return None
    best_name: Optional[str] = None
    best_dist = np.inf
    for name, ref in _PALETTE.items():
        dist = float(np.sum((arr - np.asarray(ref, dtype=np.float64)) ** 2))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def candidate_labels(
    kind: str,
    color: Optional[tuple[float, float, float]] = None,
    max_candidates: int = 24,
) -> list[str]:
    """Return a prioritised shortlist of prompts to try for a change.

    The list starts with prompts suggested by ``color`` (when given), then the
    generic prompts for ``kind`` (``"added"``, ``"removed"`` or ``"moved"``),
    and finally the full :data:`INDOOR_LABELS` vocabulary as a fallback.  The
    result is de-duplicated, keeps its priority order and is truncated to
    ``max_candidates`` (never fewer than one entry).

    ``kind`` is the :attr:`hp.types.ChangeCluster.kind` of the change.  Unknown
    kinds fall back to the ``"added"`` ordering.
    """
    kind_key = (kind or "").strip().lower()
    limit = max(int(max_candidates), 1)

    ordered: list[str] = []
    name = color_name(color)
    if name is not None:
        ordered.extend(_COLOR_PRIORITY.get(name, ()))
    ordered.extend(_KIND_PRIORITY.get(kind_key, _KIND_PRIORITY["added"]))
    ordered.extend(INDOOR_LABELS)

    seen: set[str] = set()
    shortlist: list[str] = []
    for prompt in ordered:
        prompt = prompt.strip().lower()
        if not prompt or prompt in seen:
            continue
        seen.add(prompt)
        shortlist.append(prompt)
        if len(shortlist) >= limit:
            break
    return shortlist
