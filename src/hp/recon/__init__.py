"""Reconstruction backends (gt / vggt / colmap)."""

from hp.recon.base import Reconstructor
from hp.recon.gt_recon import GTReconstructor, load_synthetic_pair, load_synthetic_scene

__all__ = [
    "Reconstructor",
    "GTReconstructor",
    "load_synthetic_scene",
    "load_synthetic_pair",
]


def get_reconstructor(backend: str, **kwargs) -> Reconstructor:
    """Factory for reconstruction backends.  Heavy backends are imported lazily."""
    backend = backend.lower()
    if backend == "gt":
        return GTReconstructor(**kwargs)
    if backend == "vggt":
        from hp.recon.vggt_recon import VGGTReconstructor

        return VGGTReconstructor(**kwargs)
    if backend == "colmap":
        from hp.recon.colmap_recon import COLMAPReconstructor

        return COLMAPReconstructor(**kwargs)
    raise ValueError(f"unknown reconstruction backend: {backend!r}")
