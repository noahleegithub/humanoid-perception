"""Ground-truth reconstruction backend.

Reads a synthetic dataset produced by :func:`hp.synthetic.generate_dataset`
(``scene_A.npz`` / ``scene_B.npz``) and exposes it as a fully registered
:class:`Scene`.  This lets the change engine be validated end-to-end with
exact geometry and no GPU, model weights, or photogrammetry.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from hp.recon.base import Reconstructor
from hp.types import Frame, Scene

__all__ = ["GTReconstructor", "load_synthetic_scene", "load_synthetic_pair"]

_WHICH = {"A": "scene_A.npz", "B": "scene_B.npz"}


def load_synthetic_scene(dataset_dir: str | Path, which: str) -> Scene:
    """Load ``scene_A.npz`` (``which="A"``) or ``scene_B.npz`` (``which="B"``)."""
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / _WHICH[which.upper()]
    data = np.load(path)
    rgbs = data["rgb"]
    depths = data["depth"]
    poses = data["poses"]
    intrinsics = data["intrinsics"]
    n = rgbs.shape[0]

    frames = [
        Frame(
            index=i,
            rgb=rgbs[i],
            depth=depths[i].astype(np.float32),
            pose=poses[i].astype(np.float64),
            intrinsics=intrinsics[i].astype(np.float64),
        )
        for i in range(n)
    ]
    scene = Scene(name=f"{dataset_dir.name}:{which}", frames=frames)
    scene.meta["source"] = str(path)
    scene.meta["backend"] = "gt"
    return scene


def load_synthetic_pair(dataset_dir: str | Path) -> tuple[Scene, Scene]:
    """Convenience helper returning ``(scene_a, scene_b)``."""
    return load_synthetic_scene(dataset_dir, "A"), load_synthetic_scene(dataset_dir, "B")


class GTReconstructor(Reconstructor):
    """``Reconstructor`` that replays ground-truth geometry from a synthetic dataset."""

    name = "gt"

    def __init__(self, dataset_dir: str | Path, which: str = "A") -> None:
        self.dataset_dir = Path(dataset_dir)
        self.which = which

    def reconstruct(self, frames: Sequence[Frame], scene_name: str) -> Scene:  # noqa: ARG002
        return load_synthetic_scene(self.dataset_dir, self.which)
