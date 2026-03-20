"""Utility functions — sweep, spline strips, placement helpers, strings."""

from blender_cli.utils.placement import (
    circle_points,
    face_points,
    grid_points,
    line_points,
    perimeter_points,
    random_points,
    rect_mask,
    sample_along_spline,
)
from blender_cli.utils.spline_strip import spline_strip
from blender_cli.utils.strings import stem_matches_keywords
from blender_cli.utils.sweep import sweep

__all__ = [
    "circle_points",
    "face_points",
    "grid_points",
    "line_points",
    "perimeter_points",
    "random_points",
    "rect_mask",
    "sample_along_spline",
    "spline_strip",
    "stem_matches_keywords",
    "sweep",
]
