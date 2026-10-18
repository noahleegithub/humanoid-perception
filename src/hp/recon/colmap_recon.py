"""Classic COLMAP structure-from-motion backend.

This backend runs the textbook COLMAP sparse pipeline entirely through the
``pycolmap`` bindings, so it works on a CPU-only machine (no GPU, no model
weights): the RGB frames are written to a temporary image folder, then

1. ``pycolmap.extract_features`` extracts SIFT features,
2. ``pycolmap.match_exhaustive`` matches every image pair, and
3. ``pycolmap.incremental_mapping`` triangulates a sparse reconstruction.

The registered images are mapped back to the input :class:`hp.types.Frame`\\ s,
producing per-frame ``pose`` (cam-to-world), ``intrinsics`` and ``rgb``.  The
sparse ``points3D`` become ``scene.points`` / ``scene.colors``.

Dense reconstruction
--------------------
Sparse SfM has **no depth maps**, so the returned frames carry ``pose``,
``intrinsics`` and ``rgb`` but ``depth=None``.  A *dense* point cloud would need
a separate multi-view-stereo pass after the sparse model is written to disk::

    pycolmap.undistort_images(mvs_path, sparse_dir, image_dir)
    pycolmap.patch_match_stereo(mvs_path)          # requires a CUDA build
    pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)

That step is deliberately **not** performed here (it needs a CUDA-compiled
COLMAP, is slow, and is out of scope for the CPU fallback); ``dense=True`` is
accepted and recorded in ``scene.meta`` so callers can tell that sparse output
was produced on request.  Use the VGGT backend for dense, per-frame depth.

``pycolmap`` is imported lazily, so ``import hp.recon`` still works without it.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from hp.recon.base import Reconstructor
from hp.types import Frame, Scene

__all__ = ["COLMAPReconstructor"]


# --------------------------------------------------------------------------- #
# lazy import + small compatibility helpers
# --------------------------------------------------------------------------- #
def _require_pycolmap() -> Any:
    """Import pycolmap or explain how to install it."""
    try:
        import pycolmap
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The COLMAP backend needs pycolmap. Install it with "
            "`pip install -e '.[colmap]'` (or `pip install pycolmap`). "
            "Importing hp.recon itself does not require pycolmap."
        ) from exc
    return pycolmap


def _write_image(path: Path, rgb: np.ndarray) -> None:
    """Write an ``HxWx3`` uint8 RGB array to ``path`` (cv2, else Pillow)."""
    try:
        import cv2

        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if not cv2.imwrite(str(path), bgr):
            raise OSError(f"cv2 could not write {path}")
        return
    except ImportError:  # pragma: no cover - cv2 is a core dependency
        pass
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(rgb)).save(path)


def _frame_index(name: str) -> int:
    """Recover the input frame index from the COLMAP image name we wrote."""
    stem = Path(name).stem
    try:
        return int(stem)
    except ValueError:
        digits = "".join(ch for ch in stem if ch.isdigit())
        if not digits:
            raise ValueError(f"cannot map COLMAP image {name!r} back to a frame") from None
        return int(digits)


def _first_float(value: Any, default: float = 0.0) -> float:
    """First element of a scalar/array-like as a float, or ``default``."""
    if value is None:
        return default
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return float(array[0]) if array.size else default


def _rigid3d_matrix(rigid: Any) -> np.ndarray:
    """Convert a pycolmap ``Rigid3d`` to a homogeneous 4x4 matrix."""
    if hasattr(rigid, "matrix"):
        matrix = np.asarray(rigid.matrix(), dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix
        if matrix.shape == (3, 4):
            out = np.eye(4, dtype=np.float64)
            out[:3, :4] = matrix
            return out
    # Older bindings expose .qvec/.tvec-style components via .rotation/.translation.
    rotation = np.asarray(rigid.rotation.matrix(), dtype=np.float64)
    translation = np.asarray(rigid.translation, dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation
    out[:3, 3] = translation
    return out


def _invert_se3(matrix: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform (world-to-cam <-> cam-to-world)."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def _camera_matrix(camera: Any) -> np.ndarray:
    """3x3 pinhole intrinsics for a pycolmap ``Camera``."""
    try:
        matrix = np.asarray(camera.calibration_matrix(), dtype=np.float64)
        if matrix.shape == (3, 3):
            return matrix
    except Exception:  # pragma: no cover - older bindings may lack the helper
        pass

    width = _first_float(getattr(camera, "width", None), 0.0)
    height = _first_float(getattr(camera, "height", None), 0.0)
    focal_x = _first_float(getattr(camera, "focal_length_x", None))
    if focal_x == 0.0:
        focal_x = _first_float(getattr(camera, "focal_length", None), max(width, height))
    focal_y = _first_float(getattr(camera, "focal_length_y", None), focal_x)
    center_x = _first_float(getattr(camera, "principal_point_x", None), width / 2.0)
    center_y = _first_float(getattr(camera, "principal_point_y", None), height / 2.0)
    return np.array(
        [[focal_x, 0.0, center_x], [0.0, focal_y, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _method_or_value(obj: Any, attribute: str) -> Any:
    """Fetch ``obj.attribute``, calling it when the binding exposes a method."""
    value = getattr(obj, attribute)
    return value() if callable(value) else value


def _call_with_fallback(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with ``kwargs``, retrying without them on ``TypeError``.

    pycolmap's ``device`` argument only exists in some releases; this keeps the
    pipeline working across 3.9 through 4.x.
    """
    if kwargs:
        try:
            return fn(*args, **kwargs)
        except TypeError:
            pass
    return fn(*args)


# --------------------------------------------------------------------------- #
# reconstruction -> Scene
# --------------------------------------------------------------------------- #
def _points_from_reconstruction(reconstruction: Any) -> tuple[np.ndarray, np.ndarray]:
    """Gather the sparse ``points3D`` as world coordinates and uint8 colours."""
    xyz: list[np.ndarray] = []
    rgb: list[np.ndarray] = []
    for point in reconstruction.points3D.values():
        xyz.append(np.asarray(point.xyz, dtype=np.float32).reshape(3))
        rgb.append(np.asarray(point.color, dtype=np.uint8).reshape(3))
    if not xyz:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    return np.stack(xyz, axis=0), np.stack(rgb, axis=0)


def _largest_reconstruction(reconstructions: Any) -> Any:
    """Pick the reconstruction with the most registered images."""
    values = list(reconstructions.values()) if hasattr(reconstructions, "values") else list(
        reconstructions
    )
    if not values:
        raise RuntimeError(
            "COLMAP produced no reconstruction. The frames may share too little texture "
            "or overlap; try more/sharper keyframes."
        )

    def registered(rec: Any) -> int:
        try:
            return int(_method_or_value(rec, "num_reg_images"))
        except Exception:  # pragma: no cover - defensive
            return len(getattr(rec, "images", {}))

    return max(values, key=registered)


def _scene_from_reconstruction(
    reconstruction: Any,
    rgb_frames: Sequence[Frame],
    scene_name: str,
    meta: dict,
) -> Scene:
    """Build a :class:`Scene` from a pycolmap ``Reconstruction``."""
    cameras = reconstruction.cameras
    frames: list[Frame] = []
    for image in reconstruction.images.values():
        if not getattr(image, "has_pose", True):
            continue
        try:
            cam_from_world = _rigid3d_matrix(
                _method_or_value(image, "cam_from_world")
            )
        except Exception:  # pragma: no cover - unregistered / rig edge cases
            continue
        try:
            order = _frame_index(image.name)
        except ValueError:
            continue
        if not 0 <= order < len(rgb_frames):
            continue
        source = rgb_frames[order]
        frames.append(
            Frame(
                index=source.index,
                rgb=source.rgb,
                depth=None,  # sparse SfM has no depth maps (see module docstring)
                pose=_invert_se3(cam_from_world),
                intrinsics=_camera_matrix(cameras[image.camera_id]),
                path=source.path,
            )
        )
    frames.sort(key=lambda frame: frame.index)

    points, colors = _points_from_reconstruction(reconstruction)
    scene = Scene(name=scene_name, frames=frames, points=points, colors=colors)
    scene.meta.update(meta)
    scene.meta.update({"n_registered": len(frames), "n_points3D": int(points.shape[0])})
    return scene


class COLMAPReconstructor(Reconstructor):
    """CPU-friendly :class:`~hp.recon.base.Reconstructor` using classic COLMAP SfM."""

    name = "colmap"

    def __init__(self, *, use_gpu: bool = False, dense: bool = False) -> None:
        """Configure the backend.

        Args:
            use_gpu: request the CUDA device for SIFT extraction/matching (needs
                a CUDA-enabled COLMAP); the sparse mapping itself runs on CPU.
            dense: recorded in ``scene.meta``.  Dense MVS would additionally need
                ``patch_match_stereo`` + ``stereo_fusion`` (see module docstring),
                which this backend does not run.
        """
        self.use_gpu = bool(use_gpu)
        self.dense = bool(dense)

    def reconstruct(self, frames: Sequence[Frame], scene_name: str) -> Scene:
        """Run COLMAP SfM on ``frames`` and return the registered :class:`Scene`."""
        pycolmap = _require_pycolmap()
        rgb_frames = [f for f in frames if f.rgb is not None]
        if not rgb_frames:
            raise ValueError("COLMAP reconstruction needs at least one frame with an RGB image")

        device: Optional[Any] = None
        device_enum = getattr(pycolmap, "Device", None)
        if device_enum is not None:
            device = device_enum.cuda if self.use_gpu else device_enum.cpu

        with tempfile.TemporaryDirectory(prefix="hp_colmap_") as tmp:
            tmp_path = Path(tmp)
            image_dir = tmp_path / "images"
            image_dir.mkdir()
            sparse_dir = tmp_path / "sparse"
            sparse_dir.mkdir()
            database_path = tmp_path / "database.db"

            for order, frame in enumerate(rgb_frames):
                _write_image(image_dir / f"{order:05d}.png", np.asarray(frame.rgb))

            if hasattr(pycolmap, "set_random_seed"):
                try:
                    pycolmap.set_random_seed(0)
                except Exception:  # pragma: no cover - version drift
                    pass

            extract_kwargs = {} if device is None else {"device": device}
            match_kwargs = {} if device is None else {"device": device}
            _call_with_fallback(
                pycolmap.extract_features, database_path, image_dir, **extract_kwargs
            )
            _call_with_fallback(
                pycolmap.match_exhaustive, database_path, **match_kwargs
            )
            reconstructions = pycolmap.incremental_mapping(
                database_path, image_dir, sparse_dir
            )
            best = _largest_reconstruction(reconstructions)

        meta = {
            "backend": "colmap",
            "use_gpu": self.use_gpu,
            "dense": self.dense,
            "device": "cuda" if self.use_gpu else "cpu",
        }
        return _scene_from_reconstruction(best, rgb_frames, scene_name, meta)
