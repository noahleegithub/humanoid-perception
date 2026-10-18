"""Synthetic indoor data generation with exact 3D ground truth."""

from hp.synthetic.render import look_at, make_intrinsics, render_view, ray_box
from hp.synthetic.room import Box, Room, default_room
from hp.synthetic.scenario import Edit, Scenario, arc_trajectory, default_scenario, generate_dataset

__all__ = [
    "Box",
    "Room",
    "default_room",
    "Edit",
    "Scenario",
    "default_scenario",
    "generate_dataset",
    "arc_trajectory",
    "render_view",
    "make_intrinsics",
    "look_at",
    "ray_box",
]
