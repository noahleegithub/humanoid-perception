"""Reconstruction backends.

A backend turns a sequence of RGB frames (or a recorded dataset) into a
:class:`hp.types.Scene` containing registered cameras and depth.  Backends are
interchangeable so the same change engine runs on ground-truth geometry in
tests, on VGGT on a Colab GPU, or on COLMAP on CPU.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from hp.types import Frame, Scene

__all__ = ["Reconstructor"]


class Reconstructor(ABC):
    """Base class for scene reconstruction backends."""

    name: str = "base"

    @abstractmethod
    def reconstruct(self, frames: Sequence[Frame], scene_name: str) -> Scene:
        """Return a :class:`Scene` with registered frames (depth + pose + intrinsics)."""
        raise NotImplementedError
