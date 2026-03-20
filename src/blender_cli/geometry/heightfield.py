"""Heightfield — terrain elevation with procedural operations and mesh generation."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import numpy.typing as npt

from blender_cli.geometry.field2d import Field2D

if TYPE_CHECKING:
    from collections.abc import Callable

    from blender_cli.assets.material import Material
    from blender_cli.geometry.mask import Mask
    from blender_cli.geometry.spline import Spline
    from blender_cli.scene.entity import Entity

NoiseType = Literal["fbm", "ridged"]
ErosionType = Literal["hydraulic", "thermal"]
StampShape = Literal["circle", "ring"]
StampOp = Literal["add", "sub", "set"]
FalloffCurve = Literal["smooth", "linear", "sharp"]


# -- Noise helpers --


def _noise_octave(
    w: int, h: int, freq: float, rng: np.random.Generator
) -> npt.NDArray[np.float32]:
    """Single octave of smooth value noise via bicubic upsampling."""
    gw = max(2, int(np.ceil(w * freq)) + 1)
    gh = max(2, int(np.ceil(h * freq)) + 1)
    grid = rng.standard_normal((gh, gw)).astype(np.float32)
    return np.asarray(
        cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC), dtype=np.float32
    )


def generate_noise(
    w: int,
    h: int,
    noise_type: NoiseType,
    amp: float,
    freq: float,
    seed: int,
    octaves: int = 6,
) -> npt.NDArray[np.float32]:
    """Generate 2D procedural noise (fbm or ridged)."""
    rng = np.random.default_rng(seed)
    total = np.zeros((h, w), dtype=np.float32)
    a = amp
    f = freq
    for _ in range(octaves):
        n = _noise_octave(w, h, f, rng)
        if noise_type == "ridged":
            total += a * (1.0 - np.abs(n))
        else:  # fbm
            total += a * n
        a *= 0.5
        f *= 2.0
    return total


class Heightfield:
    """
    Terrain elevation grid with procedural operations and mesh generation.

    Wraps a Field2D whose values represent elevation in world units.
    All operations return new instances (immutable).
    """

    __slots__ = ("_field",)

    def __init__(self, field: Field2D) -> None:
        self._field = field

    # -- properties --

    @property
    def width(self) -> int:
        return self._field.width

    @property
    def height(self) -> int:
        return self._field.height

    @property
    def meters_per_px(self) -> float:
        return self._field.meters_per_px

    @property
    def shape(self) -> tuple[int, int]:
        return self._field.shape

    # -- world-space metrics --

    @property
    def world_width(self) -> float:
        """Width of the terrain in world units (metres)."""
        return self._field.world_width

    @property
    def world_height(self) -> float:
        """Height of the terrain in world units (metres)."""
        return self._field.world_height

    @property
    def world_extent(self) -> tuple[float, float]:
        """``(world_width, world_height)`` in metres."""
        return self._field.world_extent

    @property
    def world_area(self) -> float:
        """Ground area in square metres."""
        return self._field.world_area

    # -- factory --

    @staticmethod
    def from_field(field: Field2D, z_scale: float = 1.0) -> Heightfield:
        """Create a heightfield from a Field2D, scaling values by *z_scale*."""
        data = field.to_numpy() * z_scale
        return Heightfield(Field2D(data, field.meters_per_px))

    def at_lod(self, lod: int) -> Heightfield:
        """
        Return a decimated heightfield matching ``to_mesh(lod=...)``.

        Bilinear interpolation on the returned field will agree with the
        planar interpolation of the terrain mesh quads, eliminating Z
        mismatch between sweep vertices and the rendered terrain surface.
        """
        data = self._field.to_numpy()
        h, w = data.shape
        step = max(1, lod)
        rows = list(range(0, h, step))
        cols = list(range(0, w, step))
        decimated = data[np.ix_(rows, cols)]
        return Heightfield(Field2D(decimated, self.meters_per_px * step))

    @staticmethod
    def flat(
        width: int, height: int, z: float = 0.0, meters_per_px: float = 1.0
    ) -> Heightfield:
        """Create a flat heightfield at *z* elevation."""
        return Heightfield(Field2D.constant(width, height, z, meters_per_px))

    # -- operations (all return new Heightfield) --

    def add_noise(
        self,
        type: NoiseType,
        amp: float,
        freq: float,
        seed: int = 0,
        octaves: int = 6,
    ) -> Heightfield:
        """Add procedural noise ('fbm' or 'ridged')."""
        noise = generate_noise(self.width, self.height, type, amp, freq, seed, octaves)
        data = self._field.to_numpy() + noise
        return Heightfield(Field2D(data, self.meters_per_px))

    def smooth(self, radius: int, iters: int = 1) -> Heightfield:
        """Gaussian smoothing. *radius* is pixel radius, applied *iters* times."""
        field = self._field
        for _ in range(iters):
            field = field.blur(radius)
        return Heightfield(field)

    def terrace(self, steps: int, strength: float = 1.0) -> Heightfield:
        """Create terrace/plateau effect by quantizing elevation."""
        data = self._field.to_numpy()
        lo, hi = float(data.min()), float(data.max())
        span = hi - lo
        if span < 1e-12 or steps < 1:
            return Heightfield(Field2D(data.copy(), self.meters_per_px))
        t = (data - lo) / span
        quantized = np.floor(t * steps) / steps
        blended = t + (quantized - t) * strength
        result = lo + blended * span
        return Heightfield(Field2D(result.astype(np.float32), self.meters_per_px))

    def clamp(self, min_z: float, max_z: float) -> Heightfield:
        """Clamp elevation to [min_z, max_z]."""
        data = np.clip(self._field.to_numpy(), min_z, max_z)
        return Heightfield(Field2D(data, self.meters_per_px))

    def erode(
        self,
        type: ErosionType = "hydraulic",
        iterations: int = 50,
        *,
        rain_rate: float = 0.01,
        sediment_capacity: float = 0.05,
        evaporation: float = 0.02,
        seed: int = 0,
        talus_angle: float = 35.0,
    ) -> Heightfield:
        """
        Apply terrain erosion.

        ``type="hydraulic"``: uses *rain_rate*, *sediment_capacity*,
        *evaporation*, *seed*.

        ``type="thermal"``: uses *talus_angle* (degrees).
        """
        from blender_cli.geometry._erosion import hydraulic_erosion, thermal_erosion

        data = self._field.to_numpy()
        if type == "hydraulic":
            result = hydraulic_erosion(
                data,
                self.meters_per_px,
                iterations=iterations,
                rain_rate=rain_rate,
                sediment_capacity=sediment_capacity,
                evaporation=evaporation,
                seed=seed,
            )
        elif type == "thermal":
            result = thermal_erosion(
                data,
                self.meters_per_px,
                iterations=iterations,
                talus_angle=talus_angle,
            )
        else:
            msg = f"Unknown erosion type '{type}', expected: hydraulic, thermal"
            raise ValueError(msg)
        return Heightfield(Field2D(result, self.meters_per_px))

    def radial_falloff(
        self,
        center: tuple[float, float],
        radius: float,
        edge_width: float = 0.0,
        curve: Literal["smooth", "linear"] = "smooth",
    ) -> Heightfield:
        """
        Multiply elevation by a radial gradient.

        Full strength inside *radius*, fading to 0 over *edge_width*.
        Useful for islands, craters, mounds.
        """
        data = self._field.to_numpy()
        h, w = data.shape
        mpp = self.meters_per_px
        cx, cy = center

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        xx = xx * mpp
        yy = yy * mpp
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if edge_width <= 0:
            mask = (dist <= radius).astype(np.float32)
        else:
            t = np.clip((dist - radius) / edge_width, 0.0, 1.0)
            if curve == "smooth":
                mask = (1.0 - t * t * (3.0 - 2.0 * t)).astype(np.float32)
            else:
                mask = (1.0 - t).astype(np.float32)

        return Heightfield(Field2D(data * mask, mpp))

    def remap_curve(self, points: list[tuple[float, float]]) -> Heightfield:
        """
        Piecewise-linear transfer function on elevation values.

        *points* is a sorted list of ``(input_z, output_z)`` pairs.
        Values below the first point or above the last are clamped.
        """
        if len(points) < 2:
            msg = "remap_curve requires at least 2 points"
            raise ValueError(msg)
        pts = sorted(points, key=operator.itemgetter(0))
        xs = np.array([p[0] for p in pts], dtype=np.float64)
        ys = np.array([p[1] for p in pts], dtype=np.float64)
        data = self._field.to_numpy().astype(np.float64)
        result = np.interp(data, xs, ys).astype(np.float32)
        return Heightfield(Field2D(result, self.meters_per_px))

    def stamp(
        self,
        shape: StampShape,
        center: tuple[float, float],
        radius: float,
        operation: StampOp = "add",
        amount: float = 1.0,
        falloff: FalloffCurve = "smooth",
        *,
        inner_radius: float = 0.0,
    ) -> Heightfield:
        """
        Localized terrain edit (circle or ring).

        *operation*: ``"add"``/``"sub"`` offset elevation, ``"set"`` sets
        absolute elevation.  *falloff* controls the edge ramp.
        """
        data = self._field.to_numpy().astype(np.float64)
        h, w = data.shape
        mpp = self.meters_per_px
        cx, cy = center

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        xx = xx * mpp
        yy = yy * mpp
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        if radius <= 0:
            return Heightfield(Field2D(data.astype(np.float32), mpp))

        # Compute falloff mask (1 at centre, 0 at edge)
        t = np.clip(dist / radius, 0.0, 1.0)
        if falloff == "smooth":
            mask = 1.0 - t * t * (3.0 - 2.0 * t)
        elif falloff == "linear":
            mask = 1.0 - t
        elif falloff == "sharp":
            mask = (dist <= radius * 0.8).astype(np.float64)
        else:
            msg = f"Unknown falloff '{falloff}'"
            raise ValueError(msg)

        if shape == "ring" and inner_radius > 0:
            ring_mask = 1.0 - np.clip(inner_radius / (dist + 1e-12), 0.0, 1.0)
            mask = mask * ring_mask

        if operation == "add":
            data += amount * mask
        elif operation == "sub":
            data -= amount * mask
        elif operation == "set":
            data = data * (1.0 - mask) + amount * mask
        else:
            msg = f"Unknown operation '{operation}'"
            raise ValueError(msg)

        return Heightfield(Field2D(data.astype(np.float32), mpp))

    def warp(
        self,
        field_x: Field2D,
        field_y: Field2D,
        strength: float = 1.0,
    ) -> Heightfield:
        """
        Domain warping for organic terrain distortion.

        *field_x*/*field_y* are displacement fields (in pixels) scaled by *strength*.
        """
        data = self._field.to_numpy()
        h, w = data.shape

        dx = field_x.to_numpy()[:h, :w].astype(np.float32) * strength
        dy = field_y.to_numpy()[:h, :w].astype(np.float32) * strength

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x = xx + dx
        map_y = yy + dy

        warped = cv2.remap(
            data,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        return Heightfield(
            Field2D(np.asarray(warped, dtype=np.float32), self.meters_per_px)
        )

    @staticmethod
    def blend(a: Heightfield, b: Heightfield, mask: Mask) -> Heightfield:
        """Masked interpolation between two heightfields: ``a*(1-mask) + b*mask``."""
        data_a = a._field.to_numpy().astype(np.float64)
        data_b = b._field.to_numpy().astype(np.float64)
        m = mask.to_numpy().astype(np.float64)
        result = data_a * (1.0 - m) + data_b * m
        return Heightfield(Field2D(result.astype(np.float32), a.meters_per_px))

    def apply(
        self, fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
    ) -> Heightfield:
        """Generic escape hatch: *fn* receives a copy of the elevation array."""
        result = fn(self._field.to_numpy())
        if result.shape != self._field.shape:
            msg = f"apply() result shape {result.shape} != original {self._field.shape}"
            raise ValueError(msg)
        return Heightfield(Field2D(result.astype(np.float32), self.meters_per_px))

    # -- spline convenience methods --

    def grade_along(
        self,
        spline: Spline,
        width: float,
        shoulder: float = 0.0,
        cut: float = float("inf"),
        fill: float = float("inf"),
    ) -> Heightfield:
        """Flatten terrain along spline. Delegates to ``SplineOp.grade()``."""
        from blender_cli.geometry.spline_ops import SplineOp

        return SplineOp.grade(
            self, spline, width=width, shoulder=shoulder, cut=cut, fill=fill
        )

    def carve_along(
        self,
        spline: Spline,
        width: float,
        depth: float,
        shoulder: float = 0.0,
        profile: str = "parabolic",
    ) -> Heightfield:
        """Dig a channel along spline. Delegates to ``SplineOp.carve()``."""
        from blender_cli.geometry.spline_ops import SplineOp

        return SplineOp.carve(
            self, spline, width=width, depth=depth, shoulder=shoulder, profile=profile
        )

    def embank_along(
        self,
        spline: Spline,
        width: float,
        height: float,
        shoulder: float = 0.0,
    ) -> Heightfield:
        """Raise terrain along spline. Delegates to ``SplineOp.embank()``."""
        from blender_cli.geometry.spline_ops import SplineOp

        return SplineOp.embank(
            self, spline, width=width, height=height, shoulder=shoulder
        )

    def stamp_along(
        self,
        spline: Spline,
        width: float,
        shoulder: float = 0.0,
        *,
        op: Callable[[float, float, float, float], float] | None = None,
    ) -> Heightfield:
        """Generic spline-driven operation. Delegates to ``SplineOp.apply()``."""
        from blender_cli.geometry.spline_ops import SplineOp

        return SplineOp.apply(self, spline, width=width, shoulder=shoulder, op=op)

    # -- sampling --

    def sample_at(self, x: float, y: float) -> float:
        """
        Bilinear-interpolated elevation at world coordinates (x, y).

        Same coordinate space as to_mesh(): x = col * mpp, y = row * mpp.
        Out-of-bounds coordinates are clamped to the nearest edge.
        """
        return self._field.sample(x, y)

    # -- conversions --

    def to_field(self, normalize: bool = False) -> Field2D:
        """Convert back to Field2D. If *normalize*, map to 0..1."""
        if normalize:
            return Field2D(self._field.to_numpy(), self.meters_per_px).normalize()
        return Field2D(self._field.to_numpy(), self.meters_per_px)

    def save_debug(self, path: str) -> None:
        """Save elevation as grayscale PNG for visual inspection."""
        self._field.save_debug(path)

    def to_mesh(
        self,
        lod: int = 1,
        skirts: float = 0.0,
        material: Material | None = None,
        tile_scale: float | None = None,
    ) -> Entity:
        """
        Generate a Blender mesh from this heightfield.

        *lod*: decimation step (1=full, 2=half resolution, etc.)
        *skirts*: vertical drop for edge skirts (0=no skirts)
        *material*: optional Material to assign
        *tile_scale*: UV tiling factor (units per UV tile). Default tiles every 10 metres.
        """
        import bpy

        data = self._field.to_numpy()
        h, w = data.shape
        mpp = self.meters_per_px
        step = max(1, lod)

        rows = list(range(0, h, step))
        cols = list(range(0, w, step))
        nr, nc = len(rows), len(cols)

        # Build vertices: (x, y, z) with Z = elevation
        verts: list[tuple[float, float, float]] = []
        for r in rows:
            verts.extend((c * mpp, r * mpp, float(data[r, c])) for c in cols)

        # Build quad faces
        faces: list[tuple[int, int, int, int]] = []
        for ri in range(nr - 1):
            for ci in range(nc - 1):
                i0 = ri * nc + ci
                faces.append((i0, i0 + 1, i0 + nc + 1, i0 + nc))

        # Edge skirts: vertical strips extending downward
        if skirts > 0:
            # Bottom edge (row 0)
            base = len(verts)
            for ci in range(nc):
                x, y, z = verts[ci]
                verts.append((x, y, z - skirts))
            for ci in range(nc - 1):
                t0, t1 = ci, ci + 1
                b0, b1 = base + ci, base + ci + 1
                faces.append((t1, t0, b0, b1))

            # Top edge (last row)
            base = len(verts)
            for ci in range(nc):
                x, y, z = verts[(nr - 1) * nc + ci]
                verts.append((x, y, z - skirts))
            for ci in range(nc - 1):
                t0, t1 = (nr - 1) * nc + ci, (nr - 1) * nc + ci + 1
                b0, b1 = base + ci, base + ci + 1
                faces.append((t0, t1, b1, b0))

            # Left edge (col 0)
            base = len(verts)
            for ri in range(nr):
                x, y, z = verts[ri * nc]
                verts.append((x, y, z - skirts))
            for ri in range(nr - 1):
                t0, t1 = ri * nc, (ri + 1) * nc
                b0, b1 = base + ri, base + ri + 1
                faces.append((t0, t1, b1, b0))

            # Right edge (last col)
            base = len(verts)
            for ri in range(nr):
                x, y, z = verts[ri * nc + nc - 1]
                verts.append((x, y, z - skirts))
            for ri in range(nr - 1):
                t0, t1 = ri * nc + nc - 1, (ri + 1) * nc + nc - 1
                b0, b1 = base + ri, base + ri + 1
                faces.append((t1, t0, b0, b1))

        mesh = bpy.data.meshes.new("heightfield_mesh")
        mesh.from_pydata(verts, [], faces)

        # Generate UV coordinates (planar XY projection, tiled per-meter)
        uv_layer = mesh.uv_layers.new(name="UVMap")
        tile_m = tile_scale if tile_scale is not None else 10.0
        tile_uv = 1.0 / tile_m  # UV units per world metre
        for _face_idx, face in enumerate(mesh.polygons):
            for loop_idx in face.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                vx, vy, _vz = verts[vi] if vi < len(verts) else (0, 0, 0)
                uv_layer.data[loop_idx].uv = (vx * tile_uv, vy * tile_uv)

        mesh.update()

        # Smooth shading for better visual quality
        for poly in mesh.polygons:
            poly.use_smooth = True

        obj = bpy.data.objects.new("heightfield", mesh)

        if material is not None:
            mat = material.get_or_create()
            obj.data.materials.append(mat)

        from blender_cli.scene.entity import Entity

        return Entity(obj)
