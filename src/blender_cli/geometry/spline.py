"""Spline — Catmull-Rom curves for roads, rivers, and paths."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blender_cli.snap import snap_ray as _snap_ray
from blender_cli.types import Vec3

if TYPE_CHECKING:
    from blender_cli.scene.scene import Scene


# -- Catmull-Rom math --------------------------------------------------


def _cr_eval(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """Evaluate Catmull-Rom segment at *t* ∈ [0,1] (curve through p1→p2)."""
    t2 = t * t
    t3 = t2 * t

    def _f(a: float, b: float, c: float, d: float) -> float:
        return 0.5 * (
            2.0 * b
            + (-a + c) * t
            + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2
            + (-a + 3.0 * b - 3.0 * c + d) * t3
        )

    return Vec3(
        _f(p0.x, p1.x, p2.x, p3.x),
        _f(p0.y, p1.y, p2.y, p3.y),
        _f(p0.z, p1.z, p2.z, p3.z),
    )


def _cr_tangent(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """Derivative of Catmull-Rom segment at *t* ∈ [0,1]."""
    t2 = t * t

    def _f(a: float, b: float, c: float, d: float) -> float:
        return 0.5 * (
            (-a + c)
            + (4.0 * a - 10.0 * b + 8.0 * c - 2.0 * d) * t
            + (-3.0 * a + 9.0 * b - 9.0 * c + 3.0 * d) * t2
        )

    return Vec3(
        _f(p0.x, p1.x, p2.x, p3.x),
        _f(p0.y, p1.y, p2.y, p3.y),
        _f(p0.z, p1.z, p2.z, p3.z),
    )


# -- Spline class ------------------------------------------------------


class Spline:
    """
    Catmull-Rom spline through 3D control points.

    Use :meth:`catmull` to create, then chain *resample*, *snap*, *offset*.
    """

    __slots__ = ("_closed", "_points")

    def __init__(self, points: list[Vec3], closed: bool = False) -> None:
        if len(points) < 2:
            msg = "Spline requires at least 2 control points"
            raise ValueError(msg)
        self._points = list(points)
        self._closed = closed

    @staticmethod
    def catmull(points: list[Vec3], closed: bool = False) -> Spline:
        """Create a Catmull-Rom spline through *points*."""
        return Spline(points, closed)

    # -- properties ----------------------------------------------------

    @property
    def points(self) -> list[Vec3]:
        """Control points (copy)."""
        return list(self._points)

    @property
    def closed(self) -> bool:
        return self._closed

    # -- internal helpers ----------------------------------------------

    def _num_segments(self) -> int:
        return len(self._points) if self._closed else len(self._points) - 1

    def _segment_points(self, seg: int) -> tuple[Vec3, Vec3, Vec3, Vec3]:
        """Return *(p0, p1, p2, p3)* for segment *seg*."""
        pts = self._points
        n = len(pts)
        if self._closed:
            return (
                pts[(seg - 1) % n],
                pts[seg % n],
                pts[(seg + 1) % n],
                pts[(seg + 2) % n],
            )
        # Open curve: mirror endpoints to create phantom control points.

        def _pt(idx: int) -> Vec3:
            if idx < 0:
                return pts[0] + (pts[0] - pts[1])
            if idx >= n:
                return pts[-1] + (pts[-1] - pts[-2])
            return pts[idx]

        return (_pt(seg - 1), _pt(seg), _pt(seg + 1), _pt(seg + 2))

    def _decompose_t(self, t: float) -> tuple[int, float]:
        """Map global *t* ∈ [0,1] → (segment_index, local_t)."""
        t = max(0.0, min(1.0, t))
        ns = self._num_segments()
        scaled = t * ns
        seg = int(scaled)
        if seg >= ns:
            seg = ns - 1
        return seg, scaled - seg

    # -- sampling ------------------------------------------------------

    def sample(self, t: float) -> Vec3:
        """Position at parameter *t* ∈ [0,1]."""
        seg, local = self._decompose_t(t)
        p0, p1, p2, p3 = self._segment_points(seg)
        return _cr_eval(p0, p1, p2, p3, local)

    def tangent(self, t: float) -> Vec3:
        """Tangent direction at parameter *t* ∈ [0,1]."""
        seg, local = self._decompose_t(t)
        p0, p1, p2, p3 = self._segment_points(seg)
        return _cr_tangent(p0, p1, p2, p3, local)

    # -- queries -------------------------------------------------------

    def length(self, samples: int = 256) -> float:
        """Approximate arc length by summing *samples* linear segments."""
        total = 0.0
        prev = self.sample(0.0)
        for i in range(1, samples + 1):
            curr = self.sample(i / samples)
            total += prev.distance(curr)
            prev = curr
        return total

    # -- transforms ----------------------------------------------------

    def resample(self, step_m: float) -> Spline:
        """New spline with evenly-spaced points at *step_m* metre intervals."""
        if step_m <= 0:
            msg = "step_m must be positive"
            raise ValueError(msg)

        # Build arc-length → t lookup table.
        n_lut = 512
        lut: list[tuple[float, float]] = [(0.0, 0.0)]
        prev = self.sample(0.0)
        arc = 0.0
        for i in range(1, n_lut + 1):
            t = i / n_lut
            curr = self.sample(t)
            arc += prev.distance(curr)
            lut.append((arc, t))
            prev = curr

        total_len = arc
        if total_len < 1e-9:
            return Spline([self._points[0], self._points[-1]], self._closed)

        # Walk the LUT at equal arc-length intervals.
        new_pts: list[Vec3] = [self.sample(0.0)]
        target = step_m
        li = 1
        while target <= total_len - step_m * 0.01:
            while li < len(lut) and lut[li][0] < target:
                li += 1
            if li >= len(lut):
                break
            a0, t0 = lut[li - 1]
            a1, t1 = lut[li]
            frac = (target - a0) / (a1 - a0) if (a1 - a0) > 1e-12 else 0.0
            new_pts.append(self.sample(t0 + (t1 - t0) * frac))
            target += step_m

        # Always include the endpoint.
        end = self.sample(1.0)
        if new_pts[-1].distance(end) > step_m * 0.01:
            new_pts.append(end)

        if len(new_pts) < 2:
            new_pts = [self.sample(0.0), self.sample(1.0)]

        return Spline(new_pts, self._closed)

    def offset(self, distance: float) -> Spline:
        """
        Parallel curve offset by *distance* in the XY plane.

        Positive = left of travel direction, negative = right.
        """
        n = len(self._points)
        ns = self._num_segments()
        new_pts: list[Vec3] = []
        for i in range(n):
            t = i / ns if ns > 0 else 0.0
            tang = self.tangent(t)
            # 90° CCW rotation in XY plane.
            normal = Vec3(-tang.y, tang.x, 0.0).normalized()
            new_pts.append(self._points[i] + normal * distance)
        return Spline(new_pts, self._closed)

    def smooth_z(self, sigma: float = 3.0) -> Spline:
        """
        Gaussian-smooth the Z values of control points.

        Fixes Z discontinuities that appear after snapping to terrain.
        *sigma* controls the smoothing width in number of control points.
        """
        import numpy as np

        zs = np.array([p.z for p in self._points], dtype=np.float64)
        n = len(zs)
        if n < 3 or sigma < 0.01:
            return Spline(list(self._points), self._closed)

        # Build 1D Gaussian kernel
        radius = int(np.ceil(sigma * 3))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        kernel /= kernel.sum()

        # Pad with edge values (nearest mode) and convolve
        padded = np.pad(zs, radius, mode="edge")
        zs_smooth = np.convolve(padded, kernel, mode="valid")

        new_pts = [
            Vec3(p.x, p.y, float(z))
            for p, z in zip(self._points, zs_smooth, strict=False)
        ]
        return Spline(new_pts, self._closed)

    def snap(self, scene: Scene, axis: str = "-Z") -> Spline:
        """
        Ray-cast each control point onto *scene* geometry along *axis*.

        Returns a new spline with snapped positions.
        Points that miss keep their original position.
        """
        import bpy

        # view_layer.update() is called once per Spline.snap() invocation
        # to ensure the depsgraph reflects current scene state before
        # raycasting.  One call per batch, not per control point.
        view_layer = bpy.context.view_layer
        if view_layer is None:
            msg = "No active view layer"
            raise RuntimeError(msg)
        view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bpy_scene = scene.bpy_scene

        new_pts: list[Vec3] = []
        for pt in self._points:
            origin, direction = _snap_ray(pt, axis)
            hit, loc, _normal, _idx, _obj, _mat = bpy_scene.ray_cast(
                depsgraph, origin, direction
            )
            if hit:
                new_pts.append(Vec3(float(loc.x), float(loc.y), float(loc.z)))
            else:
                new_pts.append(pt)
        return Spline(new_pts, self._closed)
