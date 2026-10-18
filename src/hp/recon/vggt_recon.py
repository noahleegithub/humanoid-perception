"""VGGT feed-forward reconstruction backend.

This backend wraps Meta's *Visual Geometry Grounded Transformer* (VGGT, CVPR
2025) from ``facebookresearch/vggt``.  It is intended to run on a **Google
Colab GPU** (the project targets a 16 GB T4): a single forward pass turns the
selected keyframes into camera poses, per-pixel depth and a coloured point
cloud, with no per-scene optimisation.

Notes
-----
* The pretrained weights are downloaded from Hugging Face on **first use** and
  cached under ``~/.cache/huggingface``; nothing is downloaded at import time.
* ``torch`` and ``vggt`` are imported lazily, so ``import hp.recon`` still works
  on a CPU-only machine that has neither installed.  Only
  :meth:`VGGTReconstructor.reconstruct` needs them.
* VGGT returns OpenCV *world-to-camera* extrinsics (``[R | t]`` with
  ``x_cam = R x_world + t``) while :class:`hp.types.Frame` stores **cam-to-world**
  poses, so the extrinsics are inverted here.
* Images are resized so the long side is 518 px (VGGT's native resolution, a
  multiple of the ViT patch size 14) before being fed to the model.

The model itself is loaded once and cached on the instance, so reconstructing
several scenes from one process only pays the weight-loading cost once.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from hp.recon.base import Reconstructor
from hp.types import Frame, Scene

__all__ = ["VGGTReconstructor"]

_LONG_SIDE = 518          # VGGT's native long side (divisible by the patch size)
_PATCH_SIZE = 14          # ViT patch size: H and W must be multiples of this
_MAX_POINTS = 4_000_000   # upper bound on the fused point cloud, to guard VRAM/RAM
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# lazy heavy imports
# --------------------------------------------------------------------------- #
def _require_torch() -> Any:
    """Import torch or explain how to install the reconstruction extras."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The VGGT backend needs PyTorch. On the Colab GPU machine run "
            "`pip install -e '.[recon]'` (or `pip install torch torchvision`). "
            "Importing hp.recon itself does not require torch."
        ) from exc
    return torch


def _require_vggt() -> Any:
    """Import the upstream VGGT model class or explain how to install it."""
    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "The VGGT backend needs the facebookresearch/vggt package. Install it on the "
            "Colab GPU machine with "
            "`pip install git+https://github.com/facebookresearch/vggt.git`."
        ) from exc
    return VGGT


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _resolve_device(torch_mod: Any, device: str) -> str:
    """Turn ``"auto"`` into a concrete device string and validate CUDA requests."""
    device = (device or "auto").lower()
    if device == "auto":
        return "cuda" if torch_mod.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch_mod.cuda.is_available():
        raise RuntimeError(
            "device='cuda' was requested but torch.cuda.is_available() is False. "
            "Run on a GPU runtime (e.g. a Colab T4) or pass device='cpu'."
        )
    return device


def _autocast_dtype(torch_mod: Any, device_type: str) -> Any:
    """bfloat16 on Ampere+ GPUs, float16 on older GPUs (e.g. the T4)."""
    if device_type != "cuda":
        return torch_mod.bfloat16
    try:
        major = int(torch_mod.cuda.get_device_capability()[0])
    except Exception:  # pragma: no cover - defensive
        major = 0
    return torch_mod.bfloat16 if major >= 8 else torch_mod.float16


def _to_numpy(value: Any) -> np.ndarray:
    """Convert a torch tensor (or pass through a numpy array) to numpy."""
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value.numpy() if hasattr(value, "numpy") else value)


def _to_se3_4x4(matrix_3x4: np.ndarray) -> np.ndarray:
    """Embed a ``3x4`` ``[R | t]`` into a homogeneous ``4x4`` transform."""
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = np.asarray(matrix_3x4, dtype=np.float64)[:3, :4]
    return out


def _invert_se3(matrix_4x4: np.ndarray) -> np.ndarray:
    """Closed-form inverse of a rigid transform (mirrors VGGT's helper)."""
    rot = matrix_4x4[:3, :3]
    trans = matrix_4x4[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -rot.T @ trans
    return inv


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """Quaternion (scalar-last ``xyzw``) to rotation matrix, batched."""
    i, j, k, r = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    two_s = 2.0 / np.sum(quat * quat, axis=-1)
    rows = [
        [1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r)],
        [two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r)],
        [two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)],
    ]
    return np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)


def _pose_enc_to_extri_intri(
    pose_enc: np.ndarray, image_size_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy fallback for ``vggt.utils.pose_enc.pose_encoding_to_extri_intri``.

    ``pose_enc`` is ``(..., 9)`` holding ``[T(3), quat(4, xyzw), fov_h, fov_w]``.
    Returns OpenCV world-to-camera extrinsics ``(..., 3, 4)`` and intrinsics
    ``(..., 3, 3)``.
    """
    trans = pose_enc[..., :3]
    quat = pose_enc[..., 3:7]
    fov_h = pose_enc[..., 7]
    fov_w = pose_enc[..., 8]
    extrinsics = np.concatenate([_quat_to_mat(quat), trans[..., None]], axis=-1)
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    fx = (width / 2.0) / np.tan(fov_w / 2.0)
    fy = (height / 2.0) / np.tan(fov_h / 2.0)
    intrinsics = np.zeros(pose_enc.shape[:-1] + (3, 3), dtype=np.float64)
    intrinsics[..., 0, 0] = fx
    intrinsics[..., 1, 1] = fy
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0
    return extrinsics, intrinsics


def _decode_poses(
    pose_enc: Any, image_size_hw: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Decode VGGT's pose encoding using the official helper when available."""
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError:
        pose_encoding_to_extri_intri = None  # type: ignore[assignment]

    if pose_encoding_to_extri_intri is not None:
        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, image_size_hw)
        return _to_numpy(extrinsics), _to_numpy(intrinsics)
    return _pose_enc_to_extri_intri(_to_numpy(pose_enc), image_size_hw)


def _unproject_depth_local(
    depths: np.ndarray, poses_c2w: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    """Back-project metric-ish depth maps into world coordinates (``S,H,W,3``)."""
    num, height, width = depths.shape
    ys, xs = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    world = np.zeros((num, height, width, 3), dtype=np.float32)
    for idx in range(num):
        k = intrinsics[idx]
        depth = depths[idx].astype(np.float64)
        x = (xs - k[0, 2]) * depth / k[0, 0]
        y = (ys - k[1, 2]) * depth / k[1, 1]
        cam = np.stack([x, y, depth], axis=-1)
        pose = poses_c2w[idx]
        world[idx] = (cam @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)
    return world


def _unproject_depth(
    depths: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
) -> np.ndarray:
    """World point maps via the official helper, with a numpy fallback."""
    try:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
    except ImportError:
        unproject_depth_map_to_point_map = None  # type: ignore[assignment]

    if unproject_depth_map_to_point_map is not None:
        try:
            return np.asarray(
                unproject_depth_map_to_point_map(depths[..., None], extrinsics_w2c, intrinsics)
            )
        except Exception:  # pragma: no cover - version drift; fall through
            pass
    return _unproject_depth_local(depths, poses_c2w, intrinsics)


# --------------------------------------------------------------------------- #
# image preprocessing
# --------------------------------------------------------------------------- #
def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint8 ``HxWx3`` image using cv2 (falls back to Pillow)."""
    try:
        import cv2

        return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    except ImportError:  # pragma: no cover - cv2 is a core dependency
        from PIL import Image

        return np.asarray(Image.fromarray(rgb).resize((width, height), Image.BICUBIC))


def _target_size(height: int, width: int) -> tuple[int, int]:
    """Scale so the long side is 518 and the short side a multiple of 14."""
    if width >= height:
        new_w = _LONG_SIDE
        new_h = int(round(height * _LONG_SIDE / width / _PATCH_SIZE)) * _PATCH_SIZE
    else:
        new_h = _LONG_SIDE
        new_w = int(round(width * _LONG_SIDE / height / _PATCH_SIZE)) * _PATCH_SIZE
    return max(_PATCH_SIZE, new_h), max(_PATCH_SIZE, new_w)


def _preprocess_frames(frames: Sequence[Frame]) -> np.ndarray:
    """Resize frames to a common ``518``-long-side canvas, padded with white.

    VGGT expects ``(S, 3, H, W)`` floats in ``[0, 1]`` but internally normalises
    with ImageNet statistics, so we only need an ``S,H,W,3`` uint8 batch here.
    Frames are padded to the largest prepared size (centre padded with white,
    matching the official ``mode="pad"`` preprocessor) so they can be stacked.
    """
    prepared = []
    for frame in frames:
        rgb = np.asarray(frame.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 RGB frame, got shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        height, width = _target_size(rgb.shape[0], rgb.shape[1])
        prepared.append(_resize_rgb(rgb, width, height))

    canvas_h = max(img.shape[0] for img in prepared)
    canvas_w = max(img.shape[1] for img in prepared)

    batch = np.full((len(prepared), canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    for idx, img in enumerate(prepared):
        top = (canvas_h - img.shape[0]) // 2
        left = (canvas_w - img.shape[1]) // 2
        batch[idx, top : top + img.shape[0], left : left + img.shape[1]] = img
    return batch


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def _predict(model: Any, images: Any) -> dict:
    """Run VGGT's documented ``model(images)`` entry point.

    Falls back to calling the aggregator and the camera/depth/point heads
    directly if a trimmed build does not return everything through ``forward``.
    """
    predictions = model(images)
    if "pose_enc" in predictions and "depth" in predictions:
        return predictions

    tokens, patch_start_idx = model.aggregator(images)
    predictions = {"pose_enc": model.camera_head(tokens)[-1]}
    depth, depth_conf = model.depth_head(tokens, images=images, patch_start_idx=patch_start_idx)
    predictions["depth"] = depth
    predictions["depth_conf"] = depth_conf
    point_head = getattr(model, "point_head", None)
    if point_head is not None:
        points, points_conf = point_head(tokens, images=images, patch_start_idx=patch_start_idx)
        predictions["world_points"] = points
        predictions["world_points_conf"] = points_conf
    return predictions


class VGGTReconstructor(Reconstructor):
    """Feed-forward :class:`~hp.recon.base.Reconstructor` backed by VGGT-1B."""

    name = "vggt"

    def __init__(
        self,
        *,
        device: str = "auto",
        frames_chunk_size: int = 6,
        vram_efficient: bool = True,
        model_id: str = "facebook/VGGT-1B",
    ) -> None:
        """Configure the backend.

        Args:
            device: ``"auto"`` (CUDA if available, else CPU), ``"cuda"`` or ``"cpu"``.
            frames_chunk_size: frames processed per aggregator chunk when the
                installed VGGT exposes the ``frames_chunk_size`` memory hook.
            vram_efficient: when true, request the frame-chunked aggregation
                hook (if available) to keep the T4's 16 GB in budget.
            model_id: Hugging Face model id or local path for the VGGT weights.
        """
        self.device = device
        self.frames_chunk_size = int(frames_chunk_size)
        self.vram_efficient = bool(vram_efficient)
        self.model_id = model_id
        self._model: Optional[Any] = None
        self._model_device: Optional[str] = None

    # -- model lifecycle ----------------------------------------------------
    def _configure_memory(self, model: Any) -> None:
        """Apply VGGT's optional ``frames_chunk_size`` hook when it exists."""
        aggregator = getattr(model, "aggregator", None)
        if aggregator is None or not hasattr(aggregator, "frames_chunk_size"):
            return
        chunk = self.frames_chunk_size if self.vram_efficient else 0
        try:
            aggregator.frames_chunk_size = chunk
        except AttributeError:  # pragma: no cover - read-only on some builds
            pass

    def _load_model(self, torch_mod: Any, device: str) -> Any:
        """Load (once) and place the VGGT model, downloading weights if needed."""
        if self._model is None:
            vggt_cls = _require_vggt()
            model = vggt_cls.from_pretrained(self.model_id)
            model.eval()
            self._configure_memory(model)
            self._model = model
        if self._model_device != device:
            self._model = self._model.to(device)
            self._model_device = device
        return self._model

    # -- main entry point ---------------------------------------------------
    def reconstruct(self, frames: Sequence[Frame], scene_name: str) -> Scene:
        """Reconstruct ``frames`` with VGGT and return a registered :class:`Scene`."""
        torch_mod = _require_torch()
        rgb_frames = [f for f in frames if f.rgb is not None]
        if not rgb_frames:
            raise ValueError("VGGT reconstruction needs at least one frame with an RGB image")

        device = _resolve_device(torch_mod, self.device)
        device_type = "cuda" if device.startswith("cuda") else "cpu"
        model = self._load_model(torch_mod, device)

        images_np = _preprocess_frames(rgb_frames)                      # (S, H, W, 3)
        tensor = torch_mod.from_numpy(np.ascontiguousarray(images_np))
        tensor = tensor.permute(0, 3, 1, 2).to(device=device, dtype=torch_mod.float32)
        tensor = tensor.div_(255.0)

        dtype = _autocast_dtype(torch_mod, device_type)
        try:
            with torch_mod.no_grad():
                with torch_mod.autocast(
                    device_type=device_type, dtype=dtype, enabled=device_type == "cuda"
                ):
                    predictions = _predict(model, tensor)

            extrinsics, intrinsics = _decode_poses(
                predictions["pose_enc"], tuple(int(v) for v in images_np.shape[1:3])
            )
            depth = _to_numpy(predictions["depth"])
            extrinsics = extrinsics[0] if extrinsics.ndim == 4 else extrinsics   # (S, 3, 4)
            intrinsics = intrinsics[0] if intrinsics.ndim == 4 else intrinsics   # (S, 3, 3)
            depths = depth[0, ..., 0] if depth.ndim == 5 else depth               # (S, H, W)

            poses_c2w = np.stack(
                [
                    _invert_se3(_to_se3_4x4(extrinsics[i]))
                    for i in range(extrinsics.shape[0])
                ]
            ).astype(np.float64)

            points, colors = self._fuse_points(
                depths, extrinsics, intrinsics, poses_c2w, images_np
            )
            scene = self._build_scene(
                scene_name, rgb_frames, images_np, depths, poses_c2w, intrinsics,
                points, colors, device,
            )
        finally:
            if device_type == "cuda":
                torch_mod.cuda.empty_cache()
        return scene

    # -- output assembly ----------------------------------------------------
    @staticmethod
    def _fuse_points(
        depths: np.ndarray,
        extrinsics: np.ndarray,
        intrinsics: np.ndarray,
        poses_c2w: np.ndarray,
        images: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Unproject depth maps into a coloured world point cloud."""
        world = _unproject_depth(depths, extrinsics, intrinsics, poses_c2w)
        valid = np.isfinite(world).all(axis=-1) & np.isfinite(depths) & (depths > _EPS)
        points = world[valid].reshape(-1, 3).astype(np.float32)
        colors = images[valid].reshape(-1, 3).astype(np.uint8)
        if points.shape[0] > _MAX_POINTS:
            stride = int(np.ceil(points.shape[0] / _MAX_POINTS))
            points = points[::stride]
            colors = colors[::stride]
        return points, colors

    def _build_scene(
        self,
        scene_name: str,
        rgb_frames: Sequence[Frame],
        images: np.ndarray,
        depths: np.ndarray,
        poses_c2w: np.ndarray,
        intrinsics: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray,
        device: str,
    ) -> Scene:
        """Assemble the :class:`Scene` with aligned rgb/depth/pose/intrinsics."""
        height, width = int(images.shape[1]), int(images.shape[2])
        out_frames = [
            Frame(
                index=src.index,
                rgb=images[i],
                depth=depths[i].astype(np.float32),
                pose=poses_c2w[i].astype(np.float64),
                intrinsics=intrinsics[i].astype(np.float64),
                path=src.path,
            )
            for i, src in enumerate(rgb_frames)
        ]
        scene = Scene(name=scene_name, frames=out_frames, points=points, colors=colors)
        scene.meta.update(
            {
                "backend": "vggt",
                "model_id": self.model_id,
                "device": device,
                "num_frames": len(out_frames),
                "frames_chunk_size": self.frames_chunk_size,
                "vram_efficient": self.vram_efficient,
                "image_size_hw": [height, width],
            }
        )
        return scene
