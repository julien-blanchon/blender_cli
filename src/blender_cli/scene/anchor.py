"""Anchor — named Empty markers for control points."""

from __future__ import annotations

from typing import TYPE_CHECKING

from blender_cli.types import Vec3, Vec3OpsMixin

if TYPE_CHECKING:
    import bpy


class Anchor(Vec3OpsMixin):
    """
    Wrapper around a Blender Empty representing a named control point.

    Anchors are used to define positions for roads, rivers, POIs, etc.
    without hardcoding coordinates.
    """

    __slots__ = ("_obj",)

    def __init__(self, obj: bpy.types.Object) -> None:
        self._obj = obj

    @property
    def bpy_object(self) -> bpy.types.Object:
        """Underlying Blender object."""
        return self._obj

    @property
    def name(self) -> str:
        return self._obj.name

    @property
    def annotation(self) -> str:
        return str(self._obj.get("_annotation", ""))

    def __repr__(self) -> str:
        loc = self.location()
        return f"Anchor({self.name!r}, annotation={self.annotation!r}, pos=({loc.x:.2f}, {loc.y:.2f}, {loc.z:.2f}))"

    def to_vec3(self) -> Vec3:
        """Vec3 adapter used by :class:`Vec3OpsMixin`."""
        return self.location()

    def location(
        self,
        *,
        x: float | None = None,
        y: float | None = None,
        z: float | None = None,
    ) -> Vec3:
        """
        Current position as Vec3, optionally with overridden coordinates.

        With no arguments, returns the anchor's location.
        With keyword arguments, returns a modified copy (useful for offsets).
        """
        loc = self._obj.location
        return Vec3(
            x if x is not None else float(loc.x),
            y if y is not None else float(loc.y),
            z if z is not None else float(loc.z),
        )
