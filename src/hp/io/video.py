"""Video ingest: decode a phone recording into a list of RGB frames.

OpenCV is used for decoding because it is already a dependency and reads the
``.mp4``/``.mov`` files phones produce without extra tooling.  At this stage the
pipeline only needs RGB images; depth, pose and intrinsics are filled in later
by the reconstruction stage.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from hp.types import Frame

__all__ = ["extract_frames", "probe_video"]


def _resize_long_side(image: np.ndarray, long_side: int | None) -> np.ndarray:
    """Downscale ``image`` so its long side is ``long_side`` (never upscales).

    Aspect ratio is preserved.  A ``long_side`` of ``None`` (or a value not
    smaller than the current long side) returns the image untouched.
    """
    if long_side is None or long_side <= 0:
        return image
    height, width = image.shape[:2]
    scale = long_side / float(max(height, width))
    if scale >= 1.0:
        return image
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def _frame_step(fps: float, stride_s: float) -> int:
    """Number of frames between samples for the requested time stride."""
    if not math.isfinite(fps) or fps <= 0.0:
        return 1
    if not math.isfinite(stride_s) or stride_s <= 0.0:
        return 1
    return max(1, int(round(fps * stride_s)))


def extract_frames(
    video_path: str | Path,
    *,
    stride_s: float = 0.5,
    long_side: int | None = 518,
    max_frames: int | None = None,
) -> list[Frame]:
    """Decode ``video_path`` into a list of RGB :class:`~hp.types.Frame` objects.

    One frame is sampled every ``stride_s`` seconds (using the video's reported
    FPS), resized so the long side is ``long_side`` pixels without upscaling, and
    collection stops early once ``max_frames`` frames have been gathered.  Each
    returned frame carries a sequential ``index``, the BGR->RGB converted image
    and the source ``path``; depth, pose and intrinsics are left ``None``.

    Raises:
        FileNotFoundError: if ``video_path`` does not exist.
        ValueError: if the video cannot be opened or yields zero frames.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    frames: list[Frame] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        step = _frame_step(fps, stride_s)
        source_path = str(path)
        decoded = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if decoded % step == 0:
                if max_frames is not None and len(frames) >= max_frames:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = _resize_long_side(rgb, long_side)
                frames.append(Frame(index=len(frames), rgb=rgb, path=source_path))
            decoded += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return frames


def probe_video(video_path: str | Path) -> dict:
    """Best-effort metadata for ``video_path``.

    Returns ``fps``, ``frame_count``, ``width``, ``height`` and ``duration_s``.
    Any field OpenCV cannot report is returned as ``0`` / ``0.0`` rather than
    raising.

    Raises:
        FileNotFoundError: if ``video_path`` does not exist.
        ValueError: if the video cannot be opened.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"could not open video: {path}")

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if not math.isfinite(fps) or fps <= 0.0:
        fps = 0.0
        duration_s = 0.0
    else:
        duration_s = max(0, frame_count) / fps

    return {
        "fps": fps,
        "frame_count": max(0, frame_count),
        "width": max(0, width),
        "height": max(0, height),
        "duration_s": duration_s,
    }
