"""
Camera — first-class camera abstraction for the render pipeline.

Provides :class:`Camera` with fluent constructors, transforms, and metadata
so cameras can be created, configured, reused, and queried as objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import bpy
import mathutils

from blender_cli.types import Vec3, Vec3Like, as_vec3

# ---------------------------------------------------------------------------
# Preset definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraPreset:
    """Data-driven camera preset configuration."""

    projection: Literal["PERSP", "ORTHO"]
    fov: float = 50.0
    ortho_scale_factor: float = 1.1
    elevation_deg: float = 45.0
    azimuth_deg: float = 45.0
    distance_factor: float = 1.2
    min_distance: float = 0.0


_PRESETS: dict[str, CameraPreset] = {
    "top": CameraPreset(
        projection="ORTHO",
        elevation_deg=90.0,
        azimuth_deg=0.0,
        distance_factor=1.0,
    ),
    "iso": CameraPreset(
        projection="PERSP",
        fov=50.0,
        elevation_deg=45.0,
        azimuth_deg=45.0,
        distance_factor=1.5,
    ),
    "iso_close": CameraPreset(
        projection="PERSP",
        fov=50.0,
        elevation_deg=55.0,
        azimuth_deg=45.0,
        distance_factor=1.5,
        min_distance=20.0,
    ),
}


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


class Camera:
    """
    Fluent wrapper around ``bpy.types.Camera`` + ``bpy.types.Object``.

    Create via factory methods:

    - :meth:`perspective` — perspective projection
    - :meth:`orthographic` — orthographic projection
    - :meth:`from_preset` — named preset positioned relative to a bbox
    """

    __slots__ = (
        "_bpy_data",
        "_bpy_object",
        "_name",
        "_props",
        "_tags",
    )

    def __init__(
        self,
        bpy_data: bpy.types.Camera,
        bpy_object: bpy.types.Object,
    ) -> None:
        self._bpy_data = bpy_data
        self._bpy_object = bpy_object
        self._name: str = bpy_object.name
        self._tags: set[str] = {"camera"}
        self._props: dict[str, object] = {}

    # -- Factory constructors -------------------------------------------------

    @classmethod
    def perspective(
        cls,
        fov: float = 50.0,
        clip_start: float = 0.1,
        clip_end: float = 1000.0,
        name: str = "camera",
    ) -> Camera:
        """Create a perspective camera."""
        cam_data = bpy.data.cameras.new(name)
        cam_data.type = "PERSP"
        cam_data.lens = _fov_to_lens(fov)
        cam_data.clip_start = clip_start
        cam_data.clip_end = clip_end
        cam_obj = bpy.data.objects.new(name, cam_data)
        return cls(cam_data, cam_obj)

    @classmethod
    def orthographic(
        cls,
        ortho_scale: float = 10.0,
        clip_start: float = 0.1,
        clip_end: float = 1000.0,
        name: str = "camera",
    ) -> Camera:
        """Create an orthographic camera."""
        cam_data = bpy.data.cameras.new(name)
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = ortho_scale
        cam_data.clip_start = clip_start
        cam_data.clip_end = clip_end
        cam_obj = bpy.data.objects.new(name, cam_data)
        return cls(cam_data, cam_obj)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        bbox: tuple[Vec3, Vec3],
        name: str = "camera",
    ) -> Camera:
        """
        Create a camera from a named preset positioned relative to *bbox*.

        Supported presets: ``"top"``, ``"iso"``, ``"iso_close"``.
        """
        if preset not in _PRESETS:
            msg = f"Unknown camera preset: {preset!r}. Choose from {sorted(_PRESETS)}"
            raise ValueError(msg)

        cfg = _PRESETS[preset]
        lo, hi = bbox
        center = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))

        cam_data = bpy.data.cameras.new(name)
        cam_data.clip_start = 0.1
        cam_data.clip_end = diag * 10

        if cfg.projection == "ORTHO":
            cam_data.type = "ORTHO"
            cam_data.ortho_scale = max(size.x, size.y, 1.0) * cfg.ortho_scale_factor
            cam_obj = bpy.data.objects.new(name, cam_data)
            cam_obj.location = (center.x, center.y, center.z + diag + 10)
            cam_obj.rotation_euler = (0, 0, 0)
        else:
            cam_data.type = "PERSP"
            dist = max(diag * cfg.distance_factor, cfg.min_distance)
            elev = math.radians(cfg.elevation_deg)
            azim = math.radians(cfg.azimuth_deg)
            cam_obj = bpy.data.objects.new(name, cam_data)
            cam_obj.location = (
                center.x + dist * math.cos(elev) * math.cos(azim),
                center.y - dist * math.cos(elev) * math.sin(azim),
                center.z + dist * math.sin(elev),
            )
            _look_at_bpy(cam_obj, center)

        cam = cls(cam_data, cam_obj)
        cam._props["preset"] = preset
        return cam

    # -- Fluent transforms ----------------------------------------------------

    def at(
        self,
        x: float | Vec3Like,
        y: float | None = None,
        z: float | None = None,
    ) -> Camera:
        """Set absolute position. Accepts ``(x, y, z)`` or a single Vec3/tuple."""
        if y is None and z is None:
            pos = as_vec3(x)  # type: ignore[arg-type]
        elif y is not None and z is not None:
            pos = Vec3(float(x), float(y), float(z))  # type: ignore[arg-type]
        else:
            msg = "at() requires either (x, y, z) or a single Vec3/tuple"
            raise ValueError(msg)
        self._bpy_object.location = (pos.x, pos.y, pos.z)
        return self

    def look_at(
        self,
        x: float | Vec3Like,
        y: float | None = None,
        z: float | None = None,
    ) -> Camera:
        """Point the camera at a world-space target. Accepts ``(x, y, z)`` or Vec3/tuple."""
        if y is None and z is None:
            target = as_vec3(x)  # type: ignore[arg-type]
        elif y is not None and z is not None:
            target = Vec3(float(x), float(y), float(z))  # type: ignore[arg-type]
        else:
            msg = "look_at() requires either (x, y, z) or a single Vec3/tuple"
            raise ValueError(msg)
        _look_at_bpy(self._bpy_object, target)
        return self

    def translate(self, dx: float, dy: float, dz: float) -> Camera:
        """Offset position relative to current location."""
        loc = self._bpy_object.location
        self._bpy_object.location = (loc.x + dx, loc.y + dy, loc.z + dz)
        return self

    def set_fov(self, degrees: float) -> Camera:
        """Set field of view in degrees (perspective only)."""
        if self._bpy_data.type != "PERSP":
            msg = "set_fov() only applies to perspective cameras"
            raise TypeError(msg)
        self._bpy_data.lens = _fov_to_lens(degrees)
        return self

    def set_clip(self, near: float, far: float) -> Camera:
        """Set near and far clipping planes."""
        self._bpy_data.clip_start = near
        self._bpy_data.clip_end = far
        return self

    def set_ortho_scale(self, scale: float) -> Camera:
        """Set orthographic scale (ortho only)."""
        if self._bpy_data.type != "ORTHO":
            msg = "set_ortho_scale() only applies to orthographic cameras"
            raise TypeError(msg)
        self._bpy_data.ortho_scale = scale
        return self

    # -- Fluent metadata ------------------------------------------------------

    def named(self, value: str) -> Camera:
        """Set the camera name."""
        self._name = value
        self._bpy_object.name = value
        return self

    def tag(self, *values: str) -> Camera:
        """Add tags to this camera."""
        for v in values:
            if v:
                self._tags.add(v)
        return self

    def props(self, **values: object) -> Camera:
        """Set metadata properties."""
        self._props.update(values)
        return self

    # -- Properties -----------------------------------------------------------

    @property
    def position(self) -> Vec3:
        """Current world-space position."""
        loc = self._bpy_object.location
        return Vec3(float(loc.x), float(loc.y), float(loc.z))

    @property
    def fov(self) -> float:
        """Field of view in degrees (perspective) or 0 (ortho)."""
        if self._bpy_data.type == "PERSP":
            return _lens_to_fov(self._bpy_data.lens)
        return 0.0

    @property
    def projection(self) -> Literal["ORTHO", "PERSP"]:
        """Projection type."""
        return self._bpy_data.type  # type: ignore[return-value]

    @property
    def name(self) -> str:
        """Camera name."""
        return self._name

    @property
    def tags(self) -> set[str]:
        """Camera tags (always includes ``'camera'``)."""
        return set(self._tags)

    @property
    def camera_props(self) -> dict[str, object]:
        """Read-only copy of metadata properties."""
        return dict(self._props)

    @property
    def bpy_object(self) -> bpy.types.Object:
        """Underlying Blender object."""
        return self._bpy_object

    @property
    def bpy_data(self) -> bpy.types.Camera:
        """Underlying Blender camera data."""
        return self._bpy_data

    # -- Metric methods -------------------------------------------------------

    def visible_extent(self, bbox: tuple[Vec3, Vec3]) -> tuple[float, float]:
        """
        Estimated visible extent ``(width, height)`` in metres for *bbox*.

        - **ORTHO**: ``(ortho_scale, ortho_scale)``
        - **PERSP**: ``2 * dist * tan(fov/2)`` for each axis (square sensor assumed)
        """
        if self._bpy_data.type == "ORTHO":
            s = float(self._bpy_data.ortho_scale)
            return (s, s)
        # Perspective — distance from camera to bbox centre
        lo, hi = bbox
        center = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        pos = self.position
        dist = math.sqrt(
            (pos.x - center.x) ** 2 + (pos.y - center.y) ** 2 + (pos.z - center.z) ** 2
        )
        half_fov = math.radians(self.fov) / 2.0
        side = 2.0 * dist * math.tan(half_fov)
        return (side, side)

    def ground_footprint(self, z: float = 0.0) -> tuple[float, float]:
        """
        Estimated ground-plane footprint ``(width, height)`` at elevation *z*.

        - **ORTHO**: ``(ortho_scale, ortho_scale)``
        - **PERSP**: ``2 * (cam_z - z) * tan(fov/2)`` for each axis
        """
        if self._bpy_data.type == "ORTHO":
            s = float(self._bpy_data.ortho_scale)
            return (s, s)
        cam_z = float(self._bpy_object.location.z)
        height_above = abs(cam_z - z)
        half_fov = math.radians(self.fov) / 2.0
        side = 2.0 * height_above * math.tan(half_fov)
        return (side, side)

    # -- Internal -------------------------------------------------------------

    def _activate(self, bs: bpy.types.Scene) -> None:
        """Link to scene and set as active camera."""
        col = bs.collection
        if col is None:
            msg = "Scene has no root collection"
            raise RuntimeError(msg)
        try:
            col.objects.link(self._bpy_object)
        except RuntimeError:
            pass  # Already linked
        bs.camera = self._bpy_object

    def __repr__(self) -> str:
        proj = self.projection
        pos = self.position
        return (
            f"Camera({self._name!r}, {proj}, "
            f"pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f}))"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fov_to_lens(fov_deg: float) -> float:
    """Convert horizontal FOV in degrees to focal length (mm) for 36mm sensor."""
    return 36.0 / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


def _lens_to_fov(lens_mm: float) -> float:
    """Convert focal length (mm) to horizontal FOV in degrees for 36mm sensor."""
    if lens_mm <= 0:
        return 0.0
    return math.degrees(2.0 * math.atan(36.0 / (2.0 * lens_mm)))


def _look_at_bpy(obj: bpy.types.Object, target: Vec3) -> None:
    """Point a Blender object at a world-space position."""
    d = mathutils.Vector((
        target.x - obj.location.x,
        target.y - obj.location.y,
        target.z - obj.location.z,
    ))
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()
