"""Build before/after synthetic scenarios and write them to disk with ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from hp.synthetic.render import look_at, make_intrinsics, render_view
from hp.synthetic.room import Box, Room, default_room

__all__ = ["Edit", "Scenario", "default_scenario", "generate_dataset", "arc_trajectory"]


@dataclass
class Edit:
    kind: str                       # "added" | "removed" | "moved"
    name: str
    label: str
    before: Optional[Box] = None
    after: Optional[Box] = None


@dataclass
class Scenario:
    room_before: Room
    room_after: Room
    edits: list[Edit]

    def ground_truth(self) -> dict:
        return {
            "changes": [
                {
                    "kind": e.kind,
                    "name": e.name,
                    "label": e.label,
                    "before": None if e.before is None else e.before.to_dict(),
                    "after": None if e.after is None else e.after.to_dict(),
                }
                for e in self.edits
            ]
        }


def arc_trajectory(
    room: Room,
    n_frames: int,
    *,
    angle_start_deg: float,
    angle_end_deg: float,
    radius: float = 1.1,
    eye_z: float = 1.35,
    height_jitter: float = 0.12,
    look_z: float = 0.9,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Camera poses along an arc looking roughly at the room centre."""
    rng = rng or np.random.default_rng(0)
    center = 0.5 * (room.lo + room.hi)
    eye_z0 = eye_z
    angles = np.deg2rad(np.linspace(angle_start_deg, angle_end_deg, n_frames))
    poses: list[np.ndarray] = []
    for i, ang in enumerate(angles):
        eye = np.array(
            [
                center[0] + radius * np.cos(ang),
                center[1] + radius * np.sin(ang),
                eye_z0 + rng.uniform(-height_jitter, height_jitter),
            ]
        )
        eye = np.clip(eye, room.lo + 0.15, room.hi - 0.15)
        target = center + np.array(
            [rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), look_z - center[2]]
        )
        poses.append(look_at(eye, target))
    return poses


def default_scenario() -> Scenario:
    """A before/after pair: move a chair, add a box, remove the plant."""
    before = default_room()
    after = before.clone()

    # 1) move the chair across the room
    chair = after.object_by_name("chair")
    assert chair is not None
    before_chair = chair.copy()
    chair.lo = np.array([0.9, 2.3, 0.0])
    chair.hi = np.array([1.3, 2.7, 0.45])
    after_chair = chair.copy()

    # 2) remove the plant
    plant = after.object_by_name("plant")
    assert plant is not None
    before_plant = plant.copy()
    after.objects = [b for b in after.objects if b.name != "plant"]

    # 3) add a new box near the table
    new_box = Box([1.9, 0.3, 0.0], [2.3, 0.7, 0.55], label="box", color=(200, 90, 60), name="box_new")
    after.add(new_box)

    edits = [
        Edit("moved", "chair", "chair", before=before_chair, after=after_chair),
        Edit("removed", "plant", "plant", before=before_plant, after=None),
        Edit("added", "box_new", "box", before=None, after=new_box.copy()),
    ]
    return Scenario(room_before=before, room_after=after, edits=edits)


def _render_sequence(
    room: Room,
    poses: list[np.ndarray],
    intrinsics: np.ndarray,
    height: int,
    width: int,
    depth_noise: float,
    rng: np.random.Generator,
) -> dict:
    rgbs, depths, sems = [], [], []
    for pose in poses:
        rgb, depth, sem = render_view(
            room, pose, intrinsics, height, width, depth_noise_std=depth_noise, rng=rng
        )
        rgbs.append(rgb)
        depths.append(depth)
        sems.append(sem)
    return {
        "rgb": np.stack(rgbs, axis=0),
        "depth": np.stack(depths, axis=0),
        "sem": np.stack(sems, axis=0),
        "poses": np.stack(poses, axis=0),
        "intrinsics": np.repeat(intrinsics[None, :, :], len(poses), axis=0),
    }


def generate_dataset(
    out_dir: str | Path,
    *,
    name: str = "room01",
    scenario: Scenario | None = None,
    height: int = 240,
    width: int = 320,
    hfov_deg: float = 60.0,
    n_frames: int = 10,
    depth_noise_std: float = 0.0,
    seed: int = 0,
) -> Path:
    """Render a before/after scenario and write ``scene_A.npz``, ``scene_B.npz``, ``gt.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario = scenario or default_scenario()
    rng = np.random.default_rng(seed)
    intrinsics = make_intrinsics(width, height, hfov_deg)

    # different-but-overlapping viewpoints for the two passes: both sweep
    # most of the way around the room so every object is observed, but with a
    # phase offset and a different eye height so the viewpoints genuinely differ
    poses_a = arc_trajectory(scenario.room_before, n_frames, angle_start_deg=200, angle_end_deg=515,
                             radius=1.05, eye_z=1.30, look_z=0.90, rng=rng)
    poses_b = arc_trajectory(scenario.room_after, n_frames, angle_start_deg=232, angle_end_deg=547,
                             radius=1.05, eye_z=1.45, look_z=0.95, rng=rng)

    seq_a = _render_sequence(scenario.room_before, poses_a, intrinsics, height, width, depth_noise_std, rng)
    seq_b = _render_sequence(scenario.room_after, poses_b, intrinsics, height, width, depth_noise_std, rng)

    np.savez_compressed(out_dir / "scene_A.npz", **seq_a)
    np.savez_compressed(out_dir / "scene_B.npz", **seq_b)

    gt = {
        "name": name,
        "voxel_size": None,
        "room_before": scenario.room_before.to_dict(),
        "room_after": scenario.room_after.to_dict(),
        **scenario.ground_truth(),
    }
    (out_dir / "gt.json").write_text(json.dumps(gt, indent=2), encoding="utf-8")
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "height": height,
                "width": width,
                "hfov_deg": hfov_deg,
                "n_frames": n_frames,
                "depth_noise_std": depth_noise_std,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_dir
