"""Placement helpers — position generators, spline sampling, rect_mask."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, cast

import numpy as np

from blender_cli.geometry.field2d import Field2D
from blender_cli.geometry.mask import Mask
from blender_cli.geometry.pointset import PointSet
from blender_cli.types import Vec3, Vec3Like, as_vec3

if TYPE_CHECKING:
    from blender_cli.geometry.spline import Spline
    from blender_cli.scene.selection import Selection
    from blender_cli.types import SupportsVec3


# ---------------------------------------------------------------------------
# Bounding box extraction helpers
# ---------------------------------------------------------------------------


def _try_extract_bounds(
    target: object,
) -> tuple[float, float, float, float, float, float] | None:
    """
    Try to extract world-space AABB from a Selection or bpy object.

    Returns ``(x_min, x_max, y_min, y_max, z_min, z_max)`` or ``None``
    if *target* is not a recognised spatial object.
    """
    import mathutils as _mu

    from blender_cli.scene.selection import Selection

    obj = None
    if isinstance(target, Selection):
        obj = target.first()
    elif hasattr(target, "bound_box") and hasattr(target, "matrix_world"):
        obj = target

    if obj is None:
        return None

    corners = [obj.matrix_world @ _mu.Vector(c) for c in obj.bound_box]
    xs = [float(c.x) for c in corners]
    ys = [float(c.y) for c in corners]
    zs = [float(c.z) for c in corners]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


# ---------------------------------------------------------------------------
# circle_points
# ---------------------------------------------------------------------------


def circle_points(
    center: Vec3Like | SupportsVec3,
    radius: float,
    count: int,
) -> PointSet:
    """
    Generate *count* evenly spaced points on a circle around *center*.

    Returns a :class:`PointSet` with points at Z = center.z.
    """
    origin = as_vec3(center)
    pts: list[Vec3] = []
    for i in range(count):
        a = 2.0 * math.pi * i / max(count, 1)
        pts.append(origin + Vec3(radius * math.cos(a), radius * math.sin(a), 0.0))
    return PointSet(pts)


# ---------------------------------------------------------------------------
# perimeter_points
# ---------------------------------------------------------------------------


def perimeter_points(
    corner1: Vec3Like | SupportsVec3 | Selection,
    corner2: Vec3Like | None = None,
    *,
    count: int,
    inset: float = 0.0,
    face: str = "inward",
    seed: int = 0,
    rng: random.Random | None = None,
) -> PointSet:
    """
    Generate oriented positions on the vertical faces of a bounding box.

    Each point carries a ``yaw`` attribute (degrees) indicating the facing
    direction of the sampled face.

    Supports multiple calling conventions:

    - ``perimeter_points(corner1, corner2, count=...)`` — two bbox corners.
    - ``perimeter_points(selection_or_bpy_object, count=...)`` — extract
      bounding box from a Blender object or Selection.
    - ``perimeter_points(corner1, count=...)`` — single Vec3 treated as
      corner with an implicit second corner at the origin.

    Returns a :class:`PointSet` with a ``yaw`` attribute (degrees) set
    via :meth:`PointSet.set_attr`.
    """
    if count <= 0:
        return PointSet([])

    # Extract bounding box from Selection or bpy object.
    bounds = _try_extract_bounds(corner1)
    if bounds is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
    elif corner2 is not None:
        c1 = as_vec3(cast("Vec3Like", corner1))
        c2 = as_vec3(corner2)
        x_min, x_max = min(c1.x, c2.x), max(c1.x, c2.x)
        y_min, y_max = min(c1.y, c2.y), max(c1.y, c2.y)
        z_min, z_max = min(c1.z, c2.z), max(c1.z, c2.z)
    else:
        c1 = as_vec3(cast("Vec3Like", corner1))
        c2 = Vec3(0.0, 0.0, 0.0)
        x_min, x_max = min(c1.x, c2.x), max(c1.x, c2.x)
        y_min, y_max = min(c1.y, c2.y), max(c1.y, c2.y)
        z_min, z_max = min(c1.z, c2.z), max(c1.z, c2.z)

    dx = x_max - x_min
    dy = y_max - y_min
    dz = max(z_max - z_min, 1.0)
    if dx <= 1e-9 or dy <= 1e-9:
        return PointSet([])

    walls = [
        ("-X", Vec3(-1.0, 0.0, 0.0), y_min, y_max),
        ("+X", Vec3(1.0, 0.0, 0.0), y_min, y_max),
        ("-Y", Vec3(0.0, -1.0, 0.0), x_min, x_max),
        ("+Y", Vec3(0.0, 1.0, 0.0), x_min, x_max),
    ]
    weights = [
        dy * dz,
        dy * dz,
        dx * dz,
        dx * dz,
    ]

    rand = rng if rng is not None else random.Random(seed)
    positions: list[Vec3] = []
    yaws: list[float] = []

    for _ in range(count):
        wall_index = rand.choices(range(4), weights=weights, k=1)[0]
        wall_name, outward, a_min, a_max = walls[wall_index]
        a0 = a_min + inset
        a1 = a_max - inset
        if a1 < a0:
            a0, a1 = a_min, a_max
        along = rand.uniform(a0, a1)

        if wall_name == "-X":
            pos = Vec3(x_min, along, z_min)
        elif wall_name == "+X":
            pos = Vec3(x_max, along, z_min)
        elif wall_name == "-Y":
            pos = Vec3(along, y_min, z_min)
        else:
            pos = Vec3(along, y_max, z_min)

        normal = -outward if face.lower() == "inward" else outward
        yaw = math.degrees(math.atan2(normal.x, normal.y))
        positions.append(pos)
        yaws.append(yaw)

    return PointSet(positions).set_attr("yaw", yaws)


# ---------------------------------------------------------------------------
# face_points
# ---------------------------------------------------------------------------

_FACE_ALIASES = {
    "top": "top",
    "bottom": "bottom",
    "+x": "+x",
    "-x": "-x",
    "+y": "+y",
    "-y": "-y",
}


def face_points(
    corner1: Vec3Like,
    corner2: Vec3Like,
    count: int,
    *,
    face: str = "top",
    seed: int = 0,
) -> PointSet:
    """
    Generate *count* random points on a face of a bounding box.

    *face* selects which face to sample:

    - ``"top"`` — max-Z plane (default, same as old ``ceiling_points``).
    - ``"bottom"`` — min-Z plane.
    - ``"+x"`` / ``"-x"`` — vertical side faces along the X axis.
    - ``"+y"`` / ``"-y"`` — vertical side faces along the Y axis.
    """
    if count <= 0:
        return PointSet([])

    c1 = as_vec3(corner1)
    c2 = as_vec3(corner2)
    x_min, x_max = min(c1.x, c2.x), max(c1.x, c2.x)
    y_min, y_max = min(c1.y, c2.y), max(c1.y, c2.y)
    z_min, z_max = min(c1.z, c2.z), max(c1.z, c2.z)

    face_key = face.lower().strip()
    if face_key not in _FACE_ALIASES:
        msg = f"Unknown face {face!r}; expected one of {list(_FACE_ALIASES)}"
        raise ValueError(msg)

    rng_inst = random.Random(seed)
    pts: list[Vec3] = []

    for _ in range(count):
        if face_key == "top":
            pts.append(
                Vec3(
                    rng_inst.uniform(x_min, x_max),
                    rng_inst.uniform(y_min, y_max),
                    z_max,
                )
            )
        elif face_key == "bottom":
            pts.append(
                Vec3(
                    rng_inst.uniform(x_min, x_max),
                    rng_inst.uniform(y_min, y_max),
                    z_min,
                )
            )
        elif face_key == "+x":
            pts.append(
                Vec3(
                    x_max,
                    rng_inst.uniform(y_min, y_max),
                    rng_inst.uniform(z_min, z_max),
                )
            )
        elif face_key == "-x":
            pts.append(
                Vec3(
                    x_min,
                    rng_inst.uniform(y_min, y_max),
                    rng_inst.uniform(z_min, z_max),
                )
            )
        elif face_key == "+y":
            pts.append(
                Vec3(
                    rng_inst.uniform(x_min, x_max),
                    y_max,
                    rng_inst.uniform(z_min, z_max),
                )
            )
        elif face_key == "-y":
            pts.append(
                Vec3(
                    rng_inst.uniform(x_min, x_max),
                    y_min,
                    rng_inst.uniform(z_min, z_max),
                )
            )

    return PointSet(pts)


# ---------------------------------------------------------------------------
# line_points
# ---------------------------------------------------------------------------


def line_points(
    start: Vec3Like,
    end: Vec3Like,
    count: int,
) -> PointSet:
    """
    Generate *count* evenly spaced points between *start* and *end*.

    Useful for fences, lamp posts, bridge supports, and other linear features.
    """
    if count <= 0:
        return PointSet([])

    s = as_vec3(start)
    e = as_vec3(end)

    if count == 1:
        mid = s.lerp(e, 0.5)
        return PointSet([mid])

    pts: list[Vec3] = []
    for i in range(count):
        t = i / (count - 1)
        pts.append(s.lerp(e, t))
    return PointSet(pts)


# ---------------------------------------------------------------------------
# grid_points
# ---------------------------------------------------------------------------


def grid_points(
    corner1: Vec3Like,
    corner2: Vec3Like,
    step_x: float,
    step_y: float,
    *,
    jitter: float = 0.0,
    seed: int = 0,
) -> PointSet:
    """
    Generate a regular grid of points within a rectangle.

    Points are placed at *step_x* / *step_y* intervals starting from the
    min corner.  Optional *jitter* adds random displacement up to that
    distance in each axis.  Z is interpolated linearly between the two
    corners.

    Useful for orchards, market stalls, tile patterns, etc.
    """
    if step_x <= 0 or step_y <= 0:
        msg = "step_x and step_y must be positive"
        raise ValueError(msg)

    c1 = as_vec3(corner1)
    c2 = as_vec3(corner2)
    x_min, x_max = min(c1.x, c2.x), max(c1.x, c2.x)
    y_min, y_max = min(c1.y, c2.y), max(c1.y, c2.y)
    z = (c1.z + c2.z) * 0.5

    rng_inst = random.Random(seed) if jitter > 0 else None
    pts: list[Vec3] = []

    x = x_min
    while x <= x_max + 1e-9:
        y = y_min
        while y <= y_max + 1e-9:
            jx = rng_inst.uniform(-jitter, jitter) if rng_inst else 0.0
            jy = rng_inst.uniform(-jitter, jitter) if rng_inst else 0.0
            pts.append(Vec3(x + jx, y + jy, z))
            y += step_y
        x += step_x

    return PointSet(pts)


# ---------------------------------------------------------------------------
# random_points
# ---------------------------------------------------------------------------


def random_points(
    corner1: Vec3Like,
    corner2: Vec3Like,
    count: int,
    *,
    seed: int = 0,
) -> PointSet:
    """
    Generate *count* uniformly random points within a rectangle.

    Z is set to the average of the two corners.

    Useful for debris, leaf litter, quick prototyping, etc.
    """
    if count <= 0:
        return PointSet([])

    c1 = as_vec3(corner1)
    c2 = as_vec3(corner2)
    x_min, x_max = min(c1.x, c2.x), max(c1.x, c2.x)
    y_min, y_max = min(c1.y, c2.y), max(c1.y, c2.y)
    z = (c1.z + c2.z) * 0.5

    rng_inst = random.Random(seed)
    pts: list[Vec3] = [
        Vec3(rng_inst.uniform(x_min, x_max), rng_inst.uniform(y_min, y_max), z)
        for _ in range(count)
    ]
    return PointSet(pts)


# ---------------------------------------------------------------------------
# sample_along_spline
# ---------------------------------------------------------------------------


def sample_along_spline(
    spline: Spline,
    every_m: float,
    jitter_m: float = 0.0,
    seed: int = 0,
    rng: random.Random | None = None,
) -> PointSet:
    """
    Sample positions at regular arc-length intervals along *spline*.

    *every_m* — spacing in metres between samples.
    *jitter_m* — random lateral offset (perpendicular to spline, XY plane).
    Returns a :class:`PointSet`.
    """
    from blender_cli.geometry.spline import Spline as _Spline

    if not isinstance(spline, _Spline):
        msg = f"Expected Spline, got {type(spline).__name__}"
        raise TypeError(msg)
    if every_m <= 0:
        msg = "every_m must be positive"
        raise ValueError(msg)

    total_len = spline.length()
    if total_len < 1e-9:
        return PointSet([spline.sample(0.0)])

    # Build arc-length → t LUT.
    n_lut = 512
    lut: list[tuple[float, float]] = [(0.0, 0.0)]
    prev = spline.sample(0.0)
    arc = 0.0
    for i in range(1, n_lut + 1):
        t = i / n_lut
        curr = spline.sample(t)
        arc += prev.distance(curr)
        lut.append((arc, t))
        prev = curr

    def _arc_to_t(target_arc: float) -> float:
        """Binary-search the LUT for the t corresponding to *target_arc*."""
        lo, hi = 0, len(lut) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if lut[mid][0] < target_arc:
                lo = mid
            else:
                hi = mid
        a0, t0 = lut[lo]
        a1, t1 = lut[hi]
        frac = (target_arc - a0) / (a1 - a0) if (a1 - a0) > 1e-12 else 0.0
        return t0 + (t1 - t0) * frac

    rand = rng if rng is not None else random.Random(seed)
    pts: list[Vec3] = []
    d = 0.0
    while d <= total_len + 1e-9:
        t = _arc_to_t(min(d, total_len))
        pos = spline.sample(t)

        if jitter_m > 0:
            tang = spline.tangent(t)
            # Perpendicular in XY plane (90° CCW).
            perp = Vec3(-tang.y, tang.x, 0.0).normalized()
            offset = rand.uniform(-jitter_m, jitter_m)
            pos = pos + perp * offset

        pts.append(pos)
        d += every_m

    return PointSet(pts)


# ---------------------------------------------------------------------------
# rect_mask
# ---------------------------------------------------------------------------


def rect_mask(
    corner1: Vec3,
    corner2: Vec3,
    resolution: int = 256,
    meters_per_px: float = 1.0,
) -> Mask:
    """
    Create a rectangular Mask covering the area between *corner1* and *corner2*.

    The mask has value 1.0 inside the rectangle and 0.0 outside.
    *resolution* controls the pixel dimensions of the mask (square).
    *meters_per_px* maps pixel coordinates to world coordinates.
    """
    data = np.zeros((resolution, resolution), dtype=np.float32)

    x_min = min(corner1.x, corner2.x)
    x_max = max(corner1.x, corner2.x)
    y_min = min(corner1.y, corner2.y)
    y_max = max(corner1.y, corner2.y)

    # Map world coords → pixel indices.
    px_x_min = max(0, math.floor(x_min / meters_per_px))
    px_x_max = min(resolution, math.ceil(x_max / meters_per_px))
    px_y_min = max(0, math.floor(y_min / meters_per_px))
    px_y_max = min(resolution, math.ceil(y_max / meters_per_px))

    data[px_y_min:px_y_max, px_x_min:px_x_max] = 1.0

    return Mask(Field2D(data, meters_per_px))
