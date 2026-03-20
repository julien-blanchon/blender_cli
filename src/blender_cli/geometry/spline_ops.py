"""SplineOp — generic spline-driven heightfield operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from blender_cli.geometry.field2d import Field2D
from blender_cli.geometry.heightfield import Heightfield

if TYPE_CHECKING:
    from collections.abc import Callable

    from blender_cli.geometry.spline import Spline


def _smoothstep(t: float) -> float:
    """Hermite smoothstep: 0→0, 1→1 with zero derivative at endpoints."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _prepare_spline_samples(
    spline: Spline, mpp: float
) -> tuple[
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
    npt.NDArray[np.floating[Any]],
]:
    """Return (sp_xy, sp_z, sp_t) arrays for dense spline sampling."""
    n_samples = max(256, int(spline.length() / (mpp * 0.5)))
    sp_xy = np.empty((n_samples + 1, 2), dtype=np.float64)
    sp_z = np.empty(n_samples + 1, dtype=np.float64)
    sp_t = np.empty(n_samples + 1, dtype=np.float64)
    for i in range(n_samples + 1):
        t = i / n_samples
        pt = spline.sample(t)
        sp_xy[i, 0] = pt.x
        sp_xy[i, 1] = pt.y
        sp_z[i] = pt.z
        sp_t[i] = t
    return sp_xy, sp_z, sp_t


class SplineOp:
    """Heightfield operations along a spline corridor."""

    @staticmethod
    def grade(
        hf: Heightfield,
        spline: Spline,
        width: float,
        shoulder: float = 0.0,
        cut: float = float("inf"),
        fill: float = float("inf"),
    ) -> Heightfield:
        """
        Flatten terrain to spline elevation.

        Within *width*/2 of the spline centre-line, terrain is set to the
        spline Z (clamped by *cut*/*fill*).  Within *shoulder* beyond that,
        it blends smoothly back to the original.
        """
        if width <= 0:
            msg = "width must be positive"
            raise ValueError(msg)

        data = hf.to_field().to_numpy()
        h, w = data.shape
        mpp = hf.meters_per_px
        half_w = width / 2.0
        outer = half_w + shoulder

        sp_xy, sp_z, _sp_t = _prepare_spline_samples(spline, mpp)

        for row in range(h):
            y_world = row * mpp
            for col in range(w):
                x_world = col * mpp

                dx = sp_xy[:, 0] - x_world
                dy = sp_xy[:, 1] - y_world
                d2 = dx * dx + dy * dy
                nearest = int(np.argmin(d2))
                dist = float(np.sqrt(d2[nearest]))

                if dist > outer:
                    continue

                target_z = float(sp_z[nearest])
                orig_z = float(data[row, col])

                delta = target_z - orig_z
                delta = max(delta, -cut) if delta < 0 else min(delta, fill)
                clamped_z = orig_z + delta

                if dist <= half_w:
                    data[row, col] = clamped_z
                else:
                    blend = _smoothstep((dist - half_w) / shoulder)
                    data[row, col] = clamped_z + (orig_z - clamped_z) * blend

        return Heightfield(Field2D(data, mpp))

    @staticmethod
    def carve(
        hf: Heightfield,
        spline: Spline,
        width: float,
        depth: float,
        shoulder: float = 0.0,
        profile: str = "parabolic",
    ) -> Heightfield:
        """
        Dig a channel below spline elevation.

        *depth* is the maximum depth at the centre.
        *profile*: ``"parabolic"`` (default), ``"flat"``, ``"v_shape"``.
        """
        if width <= 0:
            msg = "width must be positive"
            raise ValueError(msg)

        data = hf.to_field().to_numpy()
        h, w = data.shape
        mpp = hf.meters_per_px
        half_w = width / 2.0
        outer = half_w + shoulder

        sp_xy, sp_z, _sp_t = _prepare_spline_samples(spline, mpp)

        for row in range(h):
            y_world = row * mpp
            for col in range(w):
                x_world = col * mpp

                dx = sp_xy[:, 0] - x_world
                dy = sp_xy[:, 1] - y_world
                d2 = dx * dx + dy * dy
                nearest = int(np.argmin(d2))
                dist = float(np.sqrt(d2[nearest]))

                if dist > outer:
                    continue

                spline_z = float(sp_z[nearest])
                orig_z = float(data[row, col])

                if dist <= half_w:
                    lateral_t = dist / half_w if half_w > 0 else 0.0
                    if profile == "parabolic":
                        depth_at = depth * (1.0 - lateral_t * lateral_t)
                    elif profile == "v_shape":
                        depth_at = depth * (1.0 - lateral_t)
                    else:  # flat
                        depth_at = depth

                    target_z = spline_z - depth_at
                    data[row, col] = min(orig_z, target_z)
                elif shoulder > 0:
                    blend = _smoothstep((dist - half_w) / shoulder)
                    target_z = spline_z - depth
                    carved_z = min(orig_z, target_z)
                    data[row, col] = carved_z + (orig_z - carved_z) * blend

        return Heightfield(Field2D(data, mpp))

    @staticmethod
    def embank(
        hf: Heightfield,
        spline: Spline,
        width: float,
        height: float,
        shoulder: float = 0.0,
    ) -> Heightfield:
        """
        Raise terrain along spline (levees, wall foundations).

        *height* is the amount added on top of the spline Z.
        """
        if width <= 0:
            msg = "width must be positive"
            raise ValueError(msg)

        data = hf.to_field().to_numpy()
        h, w = data.shape
        mpp = hf.meters_per_px
        half_w = width / 2.0
        outer = half_w + shoulder

        sp_xy, sp_z, _sp_t = _prepare_spline_samples(spline, mpp)

        for row in range(h):
            y_world = row * mpp
            for col in range(w):
                x_world = col * mpp

                dx = sp_xy[:, 0] - x_world
                dy = sp_xy[:, 1] - y_world
                d2 = dx * dx + dy * dy
                nearest = int(np.argmin(d2))
                dist = float(np.sqrt(d2[nearest]))

                if dist > outer:
                    continue

                spline_z = float(sp_z[nearest])
                orig_z = float(data[row, col])
                target_z = spline_z + height

                if dist <= half_w:
                    data[row, col] = max(orig_z, target_z)
                elif shoulder > 0:
                    blend = _smoothstep((dist - half_w) / shoulder)
                    raised_z = max(orig_z, target_z)
                    data[row, col] = raised_z + (orig_z - raised_z) * blend

        return Heightfield(Field2D(data, mpp))

    @staticmethod
    def smooth(
        hf: Heightfield,
        spline: Spline,
        width: float,
        radius: int,
        iterations: int = 1,
    ) -> Heightfield:
        """Smooth terrain only within the spline corridor."""
        smoothed = hf.smooth(radius, iterations)
        orig_data = hf.to_field().to_numpy()
        smooth_data = smoothed.to_field().to_numpy()
        h, w = orig_data.shape
        mpp = hf.meters_per_px
        half_w = width / 2.0

        sp_xy, _sp_z, _sp_t = _prepare_spline_samples(spline, mpp)
        result = orig_data.copy()

        for row in range(h):
            y_world = row * mpp
            for col in range(w):
                x_world = col * mpp

                dx = sp_xy[:, 0] - x_world
                dy = sp_xy[:, 1] - y_world
                d2 = dx * dx + dy * dy
                dist = float(np.sqrt(d2.min()))

                if dist <= half_w:
                    t = dist / half_w if half_w > 0 else 0.0
                    blend = _smoothstep(t)
                    result[row, col] = (
                        smooth_data[row, col] * (1.0 - blend)
                        + orig_data[row, col] * blend
                    )

        return Heightfield(Field2D(result, mpp))

    @staticmethod
    def apply(
        hf: Heightfield,
        spline: Spline,
        width: float,
        shoulder: float = 0.0,
        *,
        op: Callable[[float, float, float, float], float] | None = None,
    ) -> Heightfield:
        """
        Generic spline-driven operation.

        *op(terrain_z, spline_z, lateral_t, along_t) → new_z*

        - ``lateral_t``: 0 = centre, 1 = edge
        - ``along_t``: 0 = start, 1 = end
        """
        if op is None:
            return hf

        if width <= 0:
            msg = "width must be positive"
            raise ValueError(msg)

        data = hf.to_field().to_numpy()
        h, w = data.shape
        mpp = hf.meters_per_px
        half_w = width / 2.0
        outer = half_w + shoulder

        sp_xy, sp_z, sp_t = _prepare_spline_samples(spline, mpp)

        for row in range(h):
            y_world = row * mpp
            for col in range(w):
                x_world = col * mpp

                dx = sp_xy[:, 0] - x_world
                dy = sp_xy[:, 1] - y_world
                d2 = dx * dx + dy * dy
                nearest = int(np.argmin(d2))
                dist = float(np.sqrt(d2[nearest]))

                if dist > outer:
                    continue

                terrain_z = float(data[row, col])
                spline_z = float(sp_z[nearest])
                lateral_t = min(dist / half_w, 1.0) if half_w > 0 else 0.0
                along_t = float(sp_t[nearest])

                new_z = op(terrain_z, spline_z, lateral_t, along_t)

                if dist <= half_w:
                    data[row, col] = new_z
                elif shoulder > 0:
                    blend = _smoothstep((dist - half_w) / shoulder)
                    data[row, col] = new_z + (terrain_z - new_z) * blend

        return Heightfield(Field2D(data, mpp))
