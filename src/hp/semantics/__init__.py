"""Semantic labelling and captioning of 3D changes with SAM 3.1.

Public surface:

* :data:`hp.semantics.labels.INDOOR_LABELS` / ``candidate_labels`` -- the indoor
  object prompt vocabulary and per-change prompt shortlists.
* :class:`hp.semantics.sam3.Sam3Labeler` -- SAM 3.1 concept segmentation of
  change clusters.
* :func:`hp.semantics.caption.describe` / ``summarise`` -- grounded sentences.

Everything imports cleanly without torch/transformers installed; the model
dependencies are pulled in lazily the first time segmentation runs.
"""

from hp.semantics.caption import describe, summarise
from hp.semantics.labels import INDOOR_LABELS, candidate_labels, color_name
from hp.semantics.sam3 import Sam3Labeler

__all__ = [
    "INDOOR_LABELS",
    "candidate_labels",
    "color_name",
    "Sam3Labeler",
    "describe",
    "summarise",
]
