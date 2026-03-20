"""Collision snapping engine — ray-cast points onto scene geometry."""

from blender_cli.snap.axis import AXIS_DIR, active_snap_axes, resolve_direction, snap_ray
from blender_cli.snap.objects import FilteredScene, snap, snap_object
from blender_cli.snap.results import (
    SnapObjectResult,
    SnapPolicy,
    SnapResult,
    SnapResults,
    SnapSummary,
)

__all__ = [
    "AXIS_DIR",
    "FilteredScene",
    "SnapObjectResult",
    "SnapPolicy",
    "SnapResult",
    "SnapResults",
    "SnapSummary",
    "active_snap_axes",
    "resolve_direction",
    "snap",
    "snap_object",
    "snap_ray",
]
