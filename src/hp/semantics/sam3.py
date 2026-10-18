"""Open-vocabulary labelling of 3D change clusters with SAM 3.1.

SAM 3.1 (``facebook/sam3.1``) performs *Promptable Concept Segmentation* (PCS):
given an image and a short text phrase it returns a mask for every instance of
that concept.  This module wraps that model behind a small, torch-free facade so
the rest of the pipeline can import it on a CPU-only machine and only pull in
the heavy dependencies when a label is actually requested.

For each :class:`~hp.types.ChangeCluster` we:

1. pick the frames of the scene in which the cluster is observed best, by
   projecting the cluster's centroid and bounding-box corners into every
   registered RGB-D frame, ranking by projected size and depth consistency (the
   same camera projection that :mod:`hp.change.appearance` uses);
2. crop a padded region around the projected cluster;
3. run SAM 3.1 on the crop for each prompt in
   :func:`hp.semantics.labels.candidate_labels`, keeping the prompt whose best
   mask overlaps the projected region most confidently.

Two model backends are supported, in order of preference:

* **transformers** -- ``Sam3Model`` + ``Sam3Processor`` with
  ``processor(images=..., text=...)``, ``model(**inputs)`` and
  ``processor.post_process_instance_segmentation(...)``.  This is the backend the
  code below uses when available.
* **the official sam3 repo** -- as an alternative, ``sam3.model_builder`` offers
  ``build_sam3_image_model()`` paired with
  ``sam3.model.sam3_image_processor.Sam3Processor`` (``set_image`` /
  ``set_text_prompt``).  Note that ``facebook/sam3.1`` itself is a video
  checkpoint without a transformers integration, so on that repository the
  native backend is usually the one that loads; the transformers path is used
  when a transformer-compatible checkpoint is supplied.

``torch`` and ``transformers``/``sam3`` are imported lazily inside the methods,
so ``import hp.semantics`` works with none of them installed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from hp.semantics.labels import candidate_labels
from hp.types import ChangeCluster, LabeledChange, Scene

__all__ = ["Sam3Labeler"]

# Smallest projected cluster footprint (square pixels) worth labelling.
_MIN_AREA_PX = 100.0

# Minimum fraction of a predicted mask that must fall inside the projected
# change region for the mask to count as "the object we detected".
_MIN_OVERLAP = 0.15

# Stop searching frames/prompts once a result is this confident.
_STRONG_SCORE = 0.9

# Base depth agreement tolerance (metres) for deciding a projected point is
# actually visible rather than occluded or stale.
_DEPTH_TOL_M = 0.08


class Sam3Labeler:
    """Label 3D change clusters with SAM 3.1 concept segmentation.

    The model is loaded on first use, so constructing the object never imports
    torch.  ``device`` may be ``"auto"`` (CUDA when available, else CPU),
    ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        *,
        model_id: str = "facebook/sam3.1",
        device: str = "auto",
        min_score: float = 0.4,
        use_tracker: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.min_score = float(min_score)
        self.use_tracker = bool(use_tracker)
        self._model: Any = None
        self._processor: Any = None
        self._backend: Optional[str] = None
        self._loaded_model_id: Optional[str] = None
        self._torch: Any = None
        self._resolved_device: Optional[str] = None

    # -- model lifecycle ----------------------------------------------------
    def _resolve_device(self, torch: Any) -> str:
        """Turn the requested device string into a concrete torch device name."""
        if self.device and self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        """Load the segmentation backend on first use.

        Raises a clear :class:`ImportError` when neither the ``transformers``
        image-PCS classes nor the official ``sam3`` package are installed.
        """
        if self._model is not None and self._processor is not None:
            return

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Sam3Labeler requires PyTorch; install the semantics extra with "
                "`pip install 'humanoid-perception[semantics]'` (or `pip install torch`)."
            ) from exc

        self._torch = torch
        self._resolved_device = self._resolve_device(torch)
        errors: list[str] = []

        # Preferred backend: transformers' image PCS classes.  facebook/sam3.1
        # ships as a video checkpoint, so if loading it fails we also try the
        # image-PCS checkpoint facebook/sam3 (same SAM 3 family).
        model_ids = [self.model_id]
        if "sam3.1" in self.model_id:
            model_ids.append("facebook/sam3")
        try:
            from transformers import Sam3Model, Sam3Processor
        except Exception as exc:  # noqa: BLE001 - report and fall through
            errors.append(f"transformers import: {type(exc).__name__}: {exc}")
        else:
            for mid in model_ids:
                try:
                    model = Sam3Model.from_pretrained(mid)
                    model = model.to(self._resolved_device).eval()
                    processor = Sam3Processor.from_pretrained(mid)
                except Exception as exc:  # noqa: BLE001 - report and try the next id
                    errors.append(f"transformers[{mid}]: {type(exc).__name__}: {exc}")
                    continue
                self._model, self._processor, self._backend = model, processor, "transformers"
                self._loaded_model_id = mid
                return

        # Alternative backend: the official facebookresearch/sam3 repo.
        try:
            from sam3.model.sam3_image_processor import Sam3Processor as NativeProcessor
            from sam3.model_builder import build_sam3_image_model

            model = build_sam3_image_model(device=self._resolved_device)
            processor = NativeProcessor(model)
        except Exception as exc:  # noqa: BLE001 - report and fall through
            errors.append(f"sam3: {type(exc).__name__}: {exc}")
        else:
            self._model, self._processor, self._backend = model, processor, "sam3"
            return

        raise ImportError(
            "Could not load SAM 3.1. Install either `transformers` (for Sam3Model/"
            "Sam3Processor) or the official `sam3` package (github.com/facebookresearch/sam3). "
            "Encountered: " + "; ".join(errors)
        )

    # -- segmentation -------------------------------------------------------
    def segment(self, image_rgb: np.ndarray, text_prompt: str) -> list[dict]:
        """Segment every instance of ``text_prompt`` in one RGB image.

        ``image_rgb`` is an ``HxWx3`` uint8 array; ``text_prompt`` a short noun
        phrase such as ``"coffee table"``.  Returns a list of
        ``{"mask": bool ndarray HxW, "box": [x0, y0, x1, y1], "score": float}``,
        one per detected instance (empty when nothing matches).

        The preferred call path is ``transformers``: ``Sam3Processor(images=...,
        text=...)`` -> ``Sam3Model(**inputs)`` ->
        ``Sam3Processor.post_process_instance_segmentation``.  When those classes
        are unavailable the official ``sam3`` repository is used instead, via
        ``sam3.model_builder.build_sam3_image_model`` plus
        ``sam3.model.sam3_image_processor.Sam3Processor`` (``set_image`` /
        ``set_text_prompt``).
        """
        if image_rgb is None:
            return []
        arr = np.asarray(image_rgb)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("image_rgb must be an HxWx3 array")
        if not isinstance(text_prompt, str) or not text_prompt.strip():
            return []

        self._ensure_loaded()
        if self._backend == "transformers":
            return self._segment_transformers(arr, text_prompt)
        return self._segment_native(arr, text_prompt)

    def _segment_transformers(self, arr: np.ndarray, text_prompt: str) -> list[dict]:
        from PIL import Image

        torch = self._torch
        image = Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = inputs.to(self._resolved_device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        sizes = inputs.get("original_sizes")
        if sizes is not None:
            target_sizes = sizes.tolist()
        else:
            target_sizes = [[int(arr.shape[0]), int(arr.shape[1])]]
        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=self.min_score,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]
        return self._pack(results["masks"], results["boxes"], results["scores"], arr.shape[:2])

    def _segment_native(self, arr: np.ndarray, text_prompt: str) -> list[dict]:
        from PIL import Image

        image = Image.fromarray(np.ascontiguousarray(arr.astype(np.uint8)))
        state = self._processor.set_image(image)
        output = self._processor.set_text_prompt(state=state, prompt=text_prompt)
        masks = output.get("masks")
        boxes = output.get("boxes")
        scores = output.get("scores")
        if masks is None or boxes is None or scores is None:
            return []
        return self._pack(masks, boxes, scores, arr.shape[:2])

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """Detach a torch tensor (if needed) and convert it to a numpy array."""
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        return np.asarray(value)

    def _pack(
        self,
        masks: Any,
        boxes: Any,
        scores: Any,
        shape: tuple[int, int],
    ) -> list[dict]:
        """Normalise raw model outputs into the public detection dictionaries."""
        height, width = int(shape[0]), int(shape[1])
        masks_np = self._to_numpy(masks)
        boxes_np = self._to_numpy(boxes)
        scores_np = self._to_numpy(scores).reshape(-1)
        if scores_np.size == 0:
            return []

        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None]
        if boxes_np.ndim == 1:
            boxes_np = boxes_np[None]

        detections: list[dict] = []
        for i in range(int(scores_np.shape[0])):
            if i >= masks_np.shape[0]:
                break
            mask = masks_np[i] > 0.5
            if mask.shape[:2] != (height, width):
                mask = self._resize_mask(mask, height, width)
            if not mask.any():
                continue
            box = boxes_np[i].reshape(-1)[:4].astype(np.float64)
            detections.append(
                {
                    "mask": mask,
                    "box": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "score": float(scores_np[i]),
                }
            )
        return detections

    @staticmethod
    def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
        """Nearest-neighbour resize of a boolean mask (cv2 if available)."""
        try:
            import cv2

            resized = cv2.resize(
                mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
            )
            return resized.astype(bool)
        except ImportError:  # pragma: no cover - cv2 is a core dependency
            ys = np.linspace(0, mask.shape[0] - 1, height).round().astype(int)
            xs = np.linspace(0, mask.shape[1] - 1, width).round().astype(int)
            return mask[np.ix_(ys, xs)]

    # -- frame selection ----------------------------------------------------
    @staticmethod
    def _bbox_corners(cluster: ChangeCluster) -> np.ndarray:
        lo = np.asarray(cluster.bbox_min, dtype=np.float64).reshape(3)
        hi = np.asarray(cluster.bbox_max, dtype=np.float64).reshape(3)
        xs = (lo[0], hi[0])
        ys = (lo[1], hi[1])
        zs = (lo[2], hi[2])
        return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float64)

    def _cluster_points(self, cluster: ChangeCluster) -> np.ndarray:
        """World points whose projection bounds the change in a frame."""
        centroid = np.asarray(cluster.centroid, dtype=np.float64).reshape(1, 3)
        points = np.vstack([centroid, self._bbox_corners(cluster)])
        if cluster.kind == "moved":
            for endpoint in (cluster.move_from, cluster.move_to):
                if endpoint is not None:
                    endpoint_arr = np.asarray(endpoint, dtype=np.float64).reshape(1, 3)
                    points = np.vstack([points, endpoint_arr])
        return points

    def _anchor_points(self, cluster: ChangeCluster) -> np.ndarray:
        """Points whose depth we test to judge visibility of the change."""
        centroid = np.asarray(cluster.centroid, dtype=np.float64).reshape(1, 3)
        if cluster.kind == "moved":
            endpoints = [
                np.asarray(p, dtype=np.float64).reshape(1, 3)
                for p in (cluster.move_from, cluster.move_to)
                if p is not None
            ]
            if endpoints:
                return np.vstack([centroid] + endpoints)
        return centroid

    @staticmethod
    def _project(
        points: np.ndarray, frame: Any
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project world points into a frame (same maths as appearance.py).

        Returns ``(u, v, z_camera, valid)`` where ``valid`` marks points in
        front of the camera; ``u``/``v`` are NaN elsewhere.
        """
        pose = np.asarray(frame.pose, dtype=np.float64)
        rot = pose[:3, :3]
        center = pose[:3, 3]
        pc = (points - center) @ rot
        z = pc[:, 2]
        valid = z > 1e-6
        intrinsics = np.asarray(frame.intrinsics, dtype=np.float64)
        u = np.full(points.shape[0], np.nan)
        v = np.full(points.shape[0], np.nan)
        if np.any(valid):
            zs = z[valid]
            u[valid] = intrinsics[0, 0] * pc[valid, 0] / zs + intrinsics[0, 2]
            v[valid] = intrinsics[1, 1] * pc[valid, 1] / zs + intrinsics[1, 2]
        return u, v, z, valid

    def _depth_consistency(self, points: np.ndarray, frame: Any) -> float:
        """Fraction of projected points that agree with the frame's depth map."""
        u, v, z, valid = self._project(points, frame)
        if not np.any(valid):
            return 0.0
        height, width = frame.depth.shape
        ui = np.clip(np.round(u[valid]).astype(int), 0, width - 1)
        vi = np.clip(np.round(v[valid]).astype(int), 0, height - 1)
        zi = z[valid]
        good = 0
        for j in range(zi.size):
            depth_at = float(frame.depth[vi[j], ui[j]])
            if depth_at > 1e-6 and abs(depth_at - zi[j]) <= _DEPTH_TOL_M + 0.01 * zi[j]:
                good += 1
        return good / float(zi.size)

    def _rank_frames(
        self,
        cluster: ChangeCluster,
        scene: Scene,
        top_k: int,
    ) -> list[tuple[Any, tuple[float, float, float, float]]]:
        """Return up to ``top_k`` ``(frame, projected_bbox)`` pairs, best first."""
        points = self._cluster_points(cluster)
        anchors = self._anchor_points(cluster)
        scored: list[tuple[float, float, Any, tuple[float, float, float, float]]] = []

        for frame in scene.frames:
            if (
                frame.rgb is None
                or frame.depth is None
                or frame.pose is None
                or frame.intrinsics is None
            ):
                continue
            height, width = frame.depth.shape
            u, v, _z, valid = self._project(points, frame)
            if not np.any(valid):
                continue
            us = u[valid]
            vs = v[valid]
            x0 = float(np.clip(us.min(), 0.0, width - 1))
            x1 = float(np.clip(us.max(), 0.0, width - 1))
            y0 = float(np.clip(vs.min(), 0.0, height - 1))
            y1 = float(np.clip(vs.max(), 0.0, height - 1))
            area = (x1 - x0 + 1.0) * (y1 - y0 + 1.0)
            if area < _MIN_AREA_PX:
                continue
            consistency = self._depth_consistency(anchors, frame)
            scored.append((consistency, area, frame, (x0, y0, x1, y1)))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [(item[2], item[3]) for item in scored[: max(int(top_k), 1)]]

    # -- cropping / overlap -------------------------------------------------
    @staticmethod
    def _crop(
        rgb: np.ndarray,
        box: tuple[float, float, float, float],
        *,
        pad_frac: float = 0.15,
        min_pad: int = 6,
    ) -> tuple[Optional[np.ndarray], tuple[int, int]]:
        """Crop a padded region around ``box`` (xyxy) from ``rgb``."""
        if rgb is None:
            return None, (0, 0)
        height, width = rgb.shape[:2]
        x0, y0, x1, y1 = box
        bw = max(x1 - x0 + 1.0, 1.0)
        bh = max(y1 - y0 + 1.0, 1.0)
        pad = max(int(min_pad), int(round(pad_frac * max(bw, bh))))
        cx0 = int(max(0, np.floor(x0 - pad)))
        cy0 = int(max(0, np.floor(y0 - pad)))
        cx1 = int(min(width, np.ceil(x1 + 1.0 + pad)))
        cy1 = int(min(height, np.ceil(y1 + 1.0 + pad)))
        if cx1 <= cx0 or cy1 <= cy0:
            return None, (0, 0)
        return np.ascontiguousarray(rgb[cy0:cy1, cx0:cx1]), (cx0, cy0)

    @staticmethod
    def _region_mask(
        box: tuple[float, float, float, float],
        offset: tuple[int, int],
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Boolean mask of the projected cluster inside the crop."""
        height, width = int(shape[0]), int(shape[1])
        x0, y0, x1, y1 = box
        ox, oy = offset
        rx0 = int(np.clip(np.floor(x0 - ox), 0, width))
        rx1 = int(np.clip(np.ceil(x1 - ox + 1.0), 0, width))
        ry0 = int(np.clip(np.floor(y0 - oy), 0, height))
        ry1 = int(np.clip(np.ceil(y1 - oy + 1.0), 0, height))
        region = np.zeros((height, width), dtype=bool)
        if rx1 > rx0 and ry1 > ry0:
            region[ry0:ry1, rx0:rx1] = True
        return region

    @staticmethod
    def _overlap(mask: np.ndarray, region: np.ndarray) -> float:
        """Fraction of the predicted ``mask`` that lies inside ``region``."""
        mask = np.asarray(mask, dtype=bool)
        area = int(mask.sum())
        if area == 0:
            return 0.0
        return int(np.logical_and(mask, region).sum()) / float(area)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "label": "object",
            "confidence": 0.0,
            "frame_index": -1,
            "box": None,
            "mask": None,
            "prompt": "",
        }

    # -- public labelling API ----------------------------------------------
    def label_change(
        self,
        cluster: ChangeCluster,
        scene: Scene,
        *,
        top_k_frames: int = 3,
    ) -> dict:
        """Pick the best label for one cluster from one scene.

        The cluster is projected into the ``top_k_frames`` best frames of
        ``scene``; a padded crop is segmented with each candidate prompt and the
        prompt whose best-overlapping mask has the highest confidence wins.
        When nothing beats ``min_score`` the result is
        ``{"label": "object", "confidence": 0.0, ...}``.
        """
        ranked = self._rank_frames(cluster, scene, top_k_frames)
        if not ranked:
            return self._empty_result()

        prompts = candidate_labels(cluster.kind, cluster.color)
        best_score = -1.0
        best: Optional[dict] = None

        for frame, projected in ranked:
            crop, offset = self._crop(frame.rgb, projected)
            if crop is None or crop.size == 0:
                continue
            region = self._region_mask(projected, offset, crop.shape[:2])
            if not region.any():
                continue
            for prompt in prompts:
                for detection in self.segment(crop, prompt):
                    if self._overlap(detection["mask"], region) < _MIN_OVERLAP:
                        continue
                    if detection["score"] > best_score:
                        best_score = float(detection["score"])
                        box = detection["box"]
                        best = {
                            "label": prompt,
                            "confidence": best_score,
                            "frame_index": int(frame.index),
                            "box": [
                                box[0] + offset[0],
                                box[1] + offset[1],
                                box[2] + offset[0],
                                box[3] + offset[1],
                            ],
                            "mask": detection["mask"],
                            "prompt": prompt,
                        }
            if best_score >= _STRONG_SCORE:
                break

        if best is None or best_score < self.min_score:
            fallback = self._empty_result()
            fallback["frame_index"] = int(ranked[0][0].index)
            return fallback
        return best

    def label_changes(
        self,
        changes: Sequence[ChangeCluster],
        scene_a: Scene,
        scene_b: Scene,
        *,
        top_k_frames: int = 3,
    ) -> list[LabeledChange]:
        """Label every change, choosing the scene each kind is observed in.

        Removals are only visible in the "before" scene ``scene_a``, additions in
        the "after" scene ``scene_b``.  A moved object is labelled in ``scene_b``
        (where it now is) and falling back to ``scene_a`` (where it was).
        """
        labeled: list[LabeledChange] = []
        for cluster in changes:
            if cluster.kind == "removed":
                scene = scene_a
                info = self.label_change(cluster, scene, top_k_frames=top_k_frames)
            elif cluster.kind == "moved":
                scene = scene_b
                info = self.label_change(cluster, scene, top_k_frames=top_k_frames)
                if info["label"] == "object" or info["confidence"] <= 0.0:
                    alt = self.label_change(cluster, scene_a, top_k_frames=top_k_frames)
                    if alt["confidence"] > info["confidence"]:
                        info, scene = alt, scene_a
            else:
                scene = scene_b
                info = self.label_change(cluster, scene, top_k_frames=top_k_frames)

            labeled.append(
                LabeledChange(
                    cluster=cluster,
                    label=str(info["label"]),
                    confidence=float(info["confidence"]),
                    description="",
                    evidence=[self._evidence(info, scene)],
                )
            )
        return labeled

    @staticmethod
    def _evidence(info: dict, scene: Scene) -> dict:
        """Small evidence record, plus scene bounds used by the captioner."""
        entry = {
            "frame_index": int(info.get("frame_index", -1)),
            "box": info.get("box"),
            "prompt": str(info.get("prompt", "")),
            "score": float(info.get("confidence", 0.0)),
        }
        points = scene.points
        if points is not None and len(points) > 0:
            points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
            entry["scene_bounds"] = {
                "min": [float(val) for val in points.min(axis=0)],
                "max": [float(val) for val in points.max(axis=0)],
            }
        return entry
