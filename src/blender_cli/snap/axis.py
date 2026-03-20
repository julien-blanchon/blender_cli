"""Snap axis helpers — direction vectors and ray construction."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blender_cli.types import Vec3

# -- Snap axis helpers -------------------------------------------------

AXIS_DIR: dict[str, tuple[float, float, float]] = {
    "-X": (-1.0, 0.0, 0.0),
    "+X": (1.0, 0.0, 0.0),
    "-Y": (0.0, -1.0, 0.0),
    "+Y": (0.0, 1.0, 0.0),
    "-Z": (0.0, 0.0, -1.0),
    "+Z": (0.0, 0.0, 1.0),
}

_FAR = 10_000.0


def snap_ray(
    point: Vec3, axis: str
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """
    Return *(origin, direction)* for a snap raycast along *axis*.

    The point's coordinates are used directly as the ray origin.
    The caller (:func:`snap`) places ray origins at ``_FAR`` along the
    snap axis so rays always start outside the scene geometry.
    """
    if axis not in AXIS_DIR:
        msg = f"Unknown snap axis {axis!r}, expected one of {sorted(AXIS_DIR)}"
        raise ValueError(msg)
    dx, dy, dz = AXIS_DIR[axis]
    return (point.x, point.y, point.z), (dx, dy, dz)


def resolve_direction(
    axis: str | tuple[float, float, float],
) -> tuple[float, float, float]:
    """
    Normalize *axis* to a unit direction vector.

    Accepts a string axis name (``"-Z"``, ``"+X"``, …) or an arbitrary
    ``(dx, dy, dz)`` direction tuple.  Tuple directions are normalized to
    unit length.

    Raises :class:`ValueError` for unknown string names or zero-length
    vectors.
    """
    if isinstance(axis, str):
        if axis not in AXIS_DIR:
            msg = (
                f"Unknown snap axis {axis!r}, "
                f"expected one of {sorted(AXIS_DIR)} or a direction tuple"
            )
            raise ValueError(msg)
        return AXIS_DIR[axis]
    dx, dy, dz = float(axis[0]), float(axis[1]), float(axis[2])
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-9:
        msg = "Snap direction must be non-zero"
        raise ValueError(msg)
    return (dx / length, dy / length, dz / length)


def active_snap_axes(
    direction: tuple[float, float, float],
) -> frozenset[int]:
    """
    Return axis indices with non-zero direction components.

    For example ``(0, 0, -1)`` → ``frozenset({2})`` and
    ``(1, 1, 0)`` → ``frozenset({0, 1})``.
    """
    return frozenset(i for i, d in enumerate(direction) if abs(d) > 1e-9)
