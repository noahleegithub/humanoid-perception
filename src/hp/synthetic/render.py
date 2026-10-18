"""Analytic RGB-D renderer for :mod:`hp.synthetic.room`.

Rays are intersected against axis-aligned boxes in pure numpy, giving exact
depth, per-pixel object ids, and simple Lambertian shading with a checkerboard
floor so there is at least some texture for reconstruction to latch onto.

All ray maths uses a shared camera origin ``o`` of shape ``(3,)`` and per-pixel
unit directions ``d`` of shape ``(P, 3)``.
"""

from __future__ import annotations

import numpy as np

from hp.synthetic.room import Room

__all__ = ["make_intrinsics", "look_at", "render_view", "ray_box"]

_LIGHT = np.array([0.35, -0.45, 0.82])
_LIGHT = _LIGHT / np.linalg.norm(_LIGHT)


def make_intrinsics(width: int, height: int, hfov_deg: float = 60.0) -> np.ndarray:
    """Pinhole intrinsics from a horizontal field of view."""
    fx = 0.5 * width / np.tan(np.deg2rad(hfov_deg) / 2.0)
    fy = fx
    return np.array([[fx, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]])


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Return a 4x4 OpenCV-convention cam-to-world matrix (x right, y down, z fwd)."""
    up = np.array([0.0, 0.0, 1.0]) if up is None else np.asarray(up, dtype=np.float64)
    eye = np.asarray(eye, dtype=np.float64)
    z = target - eye
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def ray_box(
    o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Slab test. ``o`` is ``(3,)`` (or ``(P,3)``), ``d`` is ``(P,3)``.

    Returns ``(tmin, tmax, enter_axis, exit_axis)`` each of shape ``(P,)``.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tlo = np.minimum(t1, t2)
    thi = np.maximum(t1, t2)
    tmin = tlo.max(axis=-1)
    tmax = thi.min(axis=-1)
    enter_axis = tlo.argmax(axis=-1)
    exit_axis = thi.argmin(axis=-1)
    return tmin, tmax, enter_axis, exit_axis


def _rays(pose: np.ndarray, intrinsics: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0:height, 0:width]
    u = xs.reshape(-1).astype(np.float64)
    v = ys.reshape(-1).astype(np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    dc /= np.linalg.norm(dc, axis=1, keepdims=True)
    origin = pose[:3, 3].copy()
    d = dc @ pose[:3, :3].T
    return origin, d


def render_view(
    room: Room,
    pose: np.ndarray,
    intrinsics: np.ndarray,
    height: int,
    width: int,
    *,
    depth_noise_std: float = 0.0,
    checker_size: float = 0.35,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render one RGB-D frame.

    Returns ``(rgb HxWx3 uint8, depth HxW float32 metres, sem HxW int16)`` where
    ``sem`` is ``-1`` for room structure and ``k`` for ``room.objects[k]``.
    """
    rng = rng or np.random.default_rng(0)
    o, d = _rays(pose, intrinsics, height, width)
    p = d.shape[0]

    # room interior: the visible wall/floor/ceiling is the exit face
    _, tmax_r, _, exit_axis = ray_box(o, d, room.lo, room.hi)
    t_room = np.where(tmax_r > 0, tmax_r, np.inf)

    best_t = t_room
    sem = np.full(p, -1, dtype=np.int64)
    axis = exit_axis.copy()

    for k, box in enumerate(room.objects):
        tmin, tmax, enter_axis, _ = ray_box(o, d, box.lo, box.hi)
        hit = (tmax >= np.maximum(tmin, 0.0)) & (tmin > 1e-6)
        t = np.where(hit, tmin, np.inf)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        sem = np.where(closer, k, sem)
        axis = np.where(closer, enter_axis, axis)

    # --- assemble colours ---------------------------------------------------
    base = np.zeros((p, 3), dtype=np.float64)
    is_room = sem < 0
    is_horizontal_room = is_room & (exit_axis == 2)
    hit_z = o[2] + t_room * d[:, 2]
    floor_face = is_horizontal_room & (np.abs(hit_z - room.lo[2]) < np.abs(hit_z - room.hi[2]))
    ceil_face = is_horizontal_room & ~floor_face

    base[is_room] = np.array(room.wall_color, dtype=np.float64)
    base[floor_face] = np.array(room.floor_color, dtype=np.float64)
    base[ceil_face] = np.array(room.ceiling_color, dtype=np.float64)
    for k, box in enumerate(room.objects):
        base[sem == k] = np.array(box.color, dtype=np.float64)

    # Lambertian shading (boxes are convex and axis aligned, so |n.L| suffices)
    shade = 0.55 + 0.45 * np.abs(_LIGHT[axis])
    rgb = base * shade[:, None]

    # checkerboard floor texture for non-degenerate features
    if np.any(floor_face):
        px = o[0] + t_room * d[:, 0]
        py = o[1] + t_room * d[:, 1]
        checker = ((np.floor(px / checker_size) + np.floor(py / checker_size)) % 2).astype(np.float64)
        rgb[floor_face] *= (0.82 + 0.18 * checker)[floor_face][:, None]

    depth = best_t.astype(np.float64)
    if depth_noise_std > 0:
        depth = depth + rng.normal(0.0, depth_noise_std, size=depth.shape)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.0)

    return (
        np.clip(rgb, 0, 255).astype(np.uint8).reshape(height, width, 3),
        depth.astype(np.float32).reshape(height, width),
        sem.astype(np.int16).reshape(height, width),
    )
