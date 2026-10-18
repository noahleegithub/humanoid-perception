"""Keyframe selection: keep the sharpest, most informative frames per video.

Phone clips are long and mostly redundant, so decoding every frame is wasteful
and reconstructing all of them is worse.  The pipeline samples a modest
candidate stride with :func:`hp.io.video.extract_frames` and then keeps a small,
temporally spread subset of the sharpest frames.  Sharpness is the variance of
the Laplacian, a cheap proxy for blur and reconstruction quality.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from hp.types import Frame

__all__ = ["sharpness", "select_keyframes"]


def sharpness(frame: Frame) -> float:
    """Variance of the Laplacian of the frame's grayscale image.

    Higher values indicate a sharper (less blurred) image.
    """
    if frame.rgb is None:
        raise ValueError("frame has no rgb image")
    gray = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _evenly_spaced_indices(count: int, keep: int) -> list[int]:
    """``keep`` indices spread evenly over ``range(count)``, in temporal order."""
    if keep >= count:
        return list(range(count))
    picked = np.linspace(0, count - 1, num=keep)
    return sorted({int(round(value)) for value in picked})


def select_keyframes(
    frames: list[Frame],
    *,
    max_keyframes: int = 12,
    min_sharpness_pct: float = 20.0,
) -> list[Frame]:
    """Drop blurry frames and cap the result, preserving temporal order.

    Frames whose :func:`sharpness` falls below the ``min_sharpness_pct``
    percentile of the input are discarded; if more than ``max_keyframes`` remain,
    an evenly-spaced subset is kept.  The survivors are renumbered ``0..N-1`` and
    are otherwise identical copies of the input frames, which are not mutated.
    """
    if not frames or max_keyframes < 1:
        return []

    scores = np.asarray([sharpness(frame) for frame in frames], dtype=np.float64)
    cutoff = float(np.percentile(scores, min_sharpness_pct))
    kept = [frame for frame, score in zip(frames, scores) if score >= cutoff]

    if len(kept) > max_keyframes:
        keep_indices = _evenly_spaced_indices(len(kept), max_keyframes)
        kept = [kept[i] for i in keep_indices]

    return [replace(frame, index=i) for i, frame in enumerate(kept)]
