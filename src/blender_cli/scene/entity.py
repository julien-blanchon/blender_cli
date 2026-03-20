"""Fluent addable wrapper for intent-first scene scripting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

import bpy
import mathutils

from blender_cli.blenvy import BevyComponentValue
from blender_cli.types import EntityMetadata, PropValue, Vec3, Vec3Like, as_vec3

if TYPE_CHECKING:
    from blender_cli.scene.anchor import Anchor
    from blender_cli.scene.scene import Scene
    from blender_cli.snap import FilteredScene, SnapPolicy


@dataclass(slots=True, frozen=True)
class SnapSpec:
    """Per-object snap configuration for fluent placement."""

    axis: str = "-Z"
    policy: "SnapPolicy | None" = None
    to: "Scene | FilteredScene | None" = None
    to_where: str | None = None


@runtime_checkable
class _SupportsAt(Protocol):
    def at(self, pos: Vec3Like) -> object: ...


@runtime_checkable
class _SupportsYaw(Protocol):
    def yaw(self, angle_deg: float) -> object: ...


@runtime_checkable
class _SupportsRot(Protocol):
    def rot(
        self,
        x: float,
        y: float,
        z: float,
        *,
        degrees: bool = True,
    ) -> object: ...


@runtime_checkable
class _SupportsApplyScale(Protocol):
    def apply_scale(self, sx: float, sy: float, sz: float) -> object: ...


@runtime_checkable
class _SupportsUniformScale(Protocol):
    def scale(self, value: float) -> object: ...


@runtime_checkable
class _SupportsCustomProps(Protocol):
    def __getitem__(self, key: str) -> object: ...

    def __setitem__(self, key: str, value: object) -> None: ...

    def get(self, key: str, default: object | None = None) -> object | None: ...


def _coerce_position(
    x: float | Vec3Like | Anchor,
    y: float | None = None,
    z: float | None = None,
) -> Vec3:
    if y is None and z is None:
        return as_vec3(cast("Vec3Like", x))
    if y is None or z is None:
        msg = "at()/rot() require either (x,y,z) or a single Vec3/tuple"
        raise ValueError(msg)
    if not isinstance(x, (int, float)):
        msg = "Expected numeric x when y/z are provided"
        raise TypeError(msg)
    return Vec3(float(x), float(y), float(z))


class Entity:
    """Thin wrapper that adds fluent transform + metadata APIs to addables."""

    __slots__ = (
        "_annotations",
        "_asset_id",
        "_asset_path",
        "_bevy_components",
        "_material_ids",
        "_name",
        "_props",
        "_snap_spec",
        "_tags",
        "_target",
    )

    def __init__(self, target: object) -> None:
        self._target = target
        self._name: str | None = None
        self._tags: set[str] = set()
        self._annotations: set[str] = set()
        self._props: dict[str, PropValue] = {}
        self._asset_id: str | None = None
        self._asset_path: str | None = None
        self._material_ids: set[str] = set()
        self._snap_spec: SnapSpec | None = None
        self._bevy_components: dict[str, BevyComponentValue] = {}

    def __repr__(self) -> str:
        name = self._name or type(self._target).__name__
        tags = ", ".join(sorted(self._tags)) if self._tags else ""
        return f"Entity({name!r}{', tags={' + tags + '}' if tags else ''})"

    @property
    def target(self) -> object:
        return self._target

    @property
    def metadata(self) -> EntityMetadata:
        return EntityMetadata(
            name=self._name,
            tags=set(self._tags),
            annotations=set(self._annotations),
            props=dict(self._props),
            asset_id=self._asset_id,
            asset_path=self._asset_path,
            material_ids=sorted(self._material_ids),
            bevy_components=dict(self._bevy_components),
        )

    @property
    def bevy_components(self) -> dict[str, BevyComponentValue]:
        """Bevy/Blenvy component dict (component_name → value)."""
        return dict(self._bevy_components)

    def named(self, value: str) -> Entity:
        self._name = value
        return self

    def component(
        self,
        name: str,
        value: BevyComponentValue = None,
    ) -> Entity:
        """Attach a Bevy/Blenvy component to this entity.

        The component will be written as a GLTF node extra in RON format
        when the scene is exported, allowing Blenvy to inject it as a
        Bevy ECS component at runtime.

        If a Bevy registry is loaded (via :func:`blender_cli.blenvy.set_registry`),
        unknown component names and value issues are reported as warnings.

        Args:
            name: Bevy component type name (e.g. ``"RigidBody"``).
            value: Component data. Use ``None`` for unit/flag components,
                a string for enum variants (``"Dynamic"``), or a dict
                for struct components (``{"max": 100}``).

        Examples::

            entity.component("SpawnBlueprint")          # unit
            entity.component("RigidBody", "Dynamic")    # enum
            entity.component("Health", {"max": 100})    # struct
        """
        import warnings

        from blender_cli.blenvy import get_registry

        reg = get_registry()
        if reg is not None:
            info = reg.find(name)
            if info is None:
                suggestions = reg.suggest(name)
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                warnings.warn(
                    f"Bevy component {name!r} not found in registry.{hint}",
                    stacklevel=2,
                )
            else:
                val_warnings = reg.validate_value(name, value)
                for w in val_warnings:
                    warnings.warn(w, stacklevel=2)
        self._bevy_components[name] = value
        return self

    def at(
        self,
        x: float | Vec3Like | Anchor,
        y: float | None = None,
        z: float | None = None,
    ) -> Entity:
        pos = _coerce_position(x, y, z)
        if isinstance(self._target, bpy.types.Object):
            self._target.location = (pos.x, pos.y, pos.z)
            return self
        if isinstance(self._target, _SupportsAt):
            self._target = self._target.at(pos)
            return self
        msg = f"Target {type(self._target).__name__} does not support at()"
        raise TypeError(msg)

    def rot(
        self,
        x: float | Vec3Like,
        y: float | None = None,
        z: float | None = None,
        *,
        degrees: bool = True,
    ) -> Entity:
        euler = _coerce_position(x, y, z)
        if degrees:
            rx, ry, rz = (
                math.radians(euler.x),
                math.radians(euler.y),
                math.radians(euler.z),
            )
        else:
            rx, ry, rz = euler.x, euler.y, euler.z
        if isinstance(self._target, bpy.types.Object):
            self._target.rotation_euler = (rx, ry, rz)
            return self
        if isinstance(self._target, _SupportsRot):
            self._target = self._target.rot(rx, ry, rz, degrees=False)
            return self
        msg = f"Target {type(self._target).__name__} does not support rot()"
        raise TypeError(msg)

    def yaw(self, angle_deg: float) -> Entity:
        if isinstance(self._target, bpy.types.Object):
            self._target.rotation_euler.z = math.radians(angle_deg)
            return self
        if isinstance(self._target, _SupportsYaw):
            self._target = self._target.yaw(angle_deg)
            return self
        msg = f"Target {type(self._target).__name__} does not support yaw()"
        raise TypeError(msg)

    def scale(
        self,
        x: float,
        y: float | None = None,
        z: float | None = None,
    ) -> Entity:
        sy = x if y is None else y
        sz = x if z is None else z
        if isinstance(self._target, bpy.types.Object):
            self._target.scale = (x, sy, sz)
            return self
        if isinstance(self._target, _SupportsApplyScale):
            self._target = self._target.apply_scale(x, sy, sz)
            return self
        if isinstance(self._target, _SupportsUniformScale):
            if abs(x - sy) > 1e-9 or abs(x - sz) > 1e-9:
                msg = f"Target {type(self._target).__name__} only supports uniform scale()"
                raise ValueError(msg)
            self._target = self._target.scale(x)
            return self
        msg = f"Target {type(self._target).__name__} does not support scale()"
        raise TypeError(msg)

    def translate(
        self,
        dx: float | Vec3Like,
        dy: float | None = None,
        dz: float | None = None,
    ) -> Entity:
        """
        Offset position by *(dx, dy, dz)* relative to current location.

        Accepts the same argument forms as :meth:`at` (3 floats or a Vec3Like).
        Only works on ``bpy.types.Object`` targets.
        """
        delta = _coerce_position(dx, dy, dz)
        if isinstance(self._target, bpy.types.Object):
            loc = self._target.location
            self._target.location = (
                loc.x + delta.x,
                loc.y + delta.y,
                loc.z + delta.z,
            )
            return self
        msg = f"Target {type(self._target).__name__} does not support translate()"
        raise TypeError(msg)

    def rotate_z(self, degrees: float) -> Entity:
        """Add *degrees* to the current Z rotation."""
        if isinstance(self._target, bpy.types.Object):
            self._target.rotation_euler.z += math.radians(degrees)
            return self
        if isinstance(self._target, _SupportsYaw):
            self._target = self._target.yaw(degrees)
            return self
        msg = f"Target {type(self._target).__name__} does not support rotate_z()"
        raise TypeError(msg)

    def scale_by(self, factor: float) -> Entity:
        """Multiply current scale uniformly by *factor*."""
        if isinstance(self._target, bpy.types.Object):
            s = self._target.scale
            self._target.scale = (s.x * factor, s.y * factor, s.z * factor)
            return self
        if isinstance(self._target, _SupportsUniformScale):
            self._target = self._target.scale(factor)
            return self
        msg = f"Target {type(self._target).__name__} does not support scale_by()"
        raise TypeError(msg)

    @property
    def location(self) -> Vec3:
        """World-space location of the object origin."""
        if isinstance(self._target, bpy.types.Object):
            loc = self._target.location
            return Vec3(float(loc.x), float(loc.y), float(loc.z))
        msg = f"Target {type(self._target).__name__} does not have a location"
        raise TypeError(msg)

    @property
    def rotation(self) -> Vec3:
        """Euler XYZ rotation in radians."""
        if isinstance(self._target, bpy.types.Object):
            r = self._target.rotation_euler
            return Vec3(float(r.x), float(r.y), float(r.z))
        msg = f"Target {type(self._target).__name__} does not have rotation"
        raise TypeError(msg)

    def world_bounds(self) -> tuple[Vec3, Vec3]:
        """
        Axis-aligned world-space bounding box ``(min, max)`` of mesh vertices.

        Accounts for location, rotation, and scale. Useful for verifying
        that snapped objects sit above terrain::

            entity.snap(scene, policy=SnapPolicy.FIRST)
            bb_min, bb_max = entity.world_bounds()
            print(f"Bottom Z: {bb_min.z}")
        """
        if not isinstance(self._target, bpy.types.Object):
            msg = (
                f"Target {type(self._target).__name__} does not support world_bounds()"
            )
            raise TypeError(msg)
        corners = [
            self._target.matrix_world @ mathutils.Vector(c)
            for c in self._target.bound_box
        ]
        xs = [float(c.x) for c in corners]
        ys = [float(c.y) for c in corners]
        zs = [float(c.z) for c in corners]
        return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))

    def size(self) -> Vec3:
        """World-space dimensions ``(dx, dy, dz)`` from :meth:`world_bounds`."""
        lo, hi = self.world_bounds()
        return Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)

    def longest_axis(self) -> float:
        """Length of the longest bounding-box axis in metres."""
        s = self.size()
        return max(s.x, s.y, s.z)

    def footprint(self) -> float:
        """Ground-plane area (size.x * size.y) in square metres."""
        s = self.size()
        return s.x * s.y

    def shade_smooth(self) -> Entity:
        """Set all polygons to smooth shading. No-op for non-mesh targets."""
        if isinstance(self._target, bpy.types.Object):
            mesh = self._target.data
            if isinstance(mesh, bpy.types.Mesh):
                for poly in mesh.polygons:
                    poly.use_smooth = True
        return self

    def shade_flat(self) -> Entity:
        """Set all polygons to flat shading. No-op for non-mesh targets."""
        if isinstance(self._target, bpy.types.Object):
            mesh = self._target.data
            if isinstance(mesh, bpy.types.Mesh):
                for poly in mesh.polygons:
                    poly.use_smooth = False
        return self

    def rescale_fit(self, target_size: float = 1.0) -> Entity:
        """Uniformly scale so the longest bounding-box axis equals *target_size* metres."""
        if not isinstance(self._target, bpy.types.Object) or not self._target.bound_box:
            return self
        corners = [
            self._target.matrix_world @ mathutils.Vector(c)
            for c in self._target.bound_box
        ]
        xs = [float(c.x) for c in corners]
        ys = [float(c.y) for c in corners]
        zs = [float(c.z) for c in corners]
        max_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if max_dim <= 1e-9:
            return self
        factor = target_size / max_dim
        return self.scale(factor)

    def tag(self, *values: str) -> Entity:
        for v in values:
            if v:
                self._tags.add(v)
        return self

    def annotate(self, *values: str) -> Entity:
        for v in values:
            if v:
                self._annotations.add(v)
        return self

    def props(self, **values: PropValue) -> Entity:
        self._props.update(values)
        return self

    def material_ids(self, *ids: str) -> Entity:
        for mid in ids:
            if mid:
                self._material_ids.add(mid)
        return self

    def asset(self, asset_id: str, path: str | Path | None = None) -> Entity:
        self._asset_id = asset_id
        if path is not None:
            self._asset_path = str(Path(path).resolve())
        return self

    def snap(
        self,
        scene: "Scene",
        *,
        axis: str = "-Z",
        to: "Scene | FilteredScene | None" = None,
        to_where: str | None = None,
        policy: "SnapPolicy | None" = None,
        spec: SnapSpec | None = None,
    ) -> Entity:
        import warnings

        from blender_cli.snap import SnapPolicy, snap_object

        if not isinstance(self._target, bpy.types.Object):
            msg = f"snap() requires a Blender object, got {type(self._target).__name__}"
            raise TypeError(msg)

        if spec is not None:
            axis = spec.axis
            if spec.policy is not None:
                policy = spec.policy
            if spec.to is not None:
                to = spec.to
            if spec.to_where is not None:
                to_where = spec.to_where

        snap_policy = policy if policy is not None else SnapPolicy.FIRST
        snap_scene = to if to is not None else scene
        if to_where is not None:
            snap_scene = scene.snap_targets(to_where)

        loc = self._target.location
        current = Vec3(float(loc.x), float(loc.y), float(loc.z))
        result = snap_object(
            obj=self._target,
            position=current,
            scene=snap_scene,
            policy=snap_policy,
            axis=axis,
        )
        self._target.location = (
            result.position.x,
            result.position.y,
            result.position.z,
        )
        if result.rotation is not None:
            self._target.rotation_euler = result.rotation
        self._snap_spec = SnapSpec(
            axis=axis, policy=snap_policy, to=to, to_where=to_where
        )

        # Warn about terrain penetration with entity context.
        if result.penetration_depth > 0.1:
            obj_name = self._name or self._target.name
            warnings.warn(
                f"Entity '{obj_name}' penetrates terrain by {result.penetration_depth:.2f}m "
                f"({result.penetrating_vertices} vertices, policy={snap_policy.value}). "
                f"Consider using SnapPolicy.FIRST or adding "
                f".translate(0, 0, {result.penetration_depth:.1f}) after .snap().",
                stacklevel=2,
            )

        return self

    def __getattr__(self, name: str) -> object:
        return getattr(self._target, name)

    def __setattr__(self, name: str, value: object) -> None:
        if name in self.__slots__:
            object.__setattr__(self, name, value)
            return
        setattr(self._target, name, value)

    def __getitem__(self, key: str) -> object:
        if isinstance(self._target, _SupportsCustomProps):
            return self._target[key]
        msg = f"Target {type(self._target).__name__} does not support custom properties"
        raise TypeError(msg)

    def __setitem__(self, key: str, value: object) -> None:
        if isinstance(self._target, _SupportsCustomProps):
            self._target[key] = value
            return
        msg = f"Target {type(self._target).__name__} does not support custom properties"
        raise TypeError(msg)

    def get(self, key: str, default: object = None) -> object:
        if isinstance(self._target, _SupportsCustomProps):
            return self._target.get(key, default)
        return default


def as_entity(target: object) -> Entity:
    """Wrap *target* as an :class:`Entity` if needed."""
    if isinstance(target, Entity):
        return target
    return Entity(target)


def unwrap_entity(target: object) -> object:
    """Return the raw wrapped target if *target* is an :class:`Entity`."""
    if isinstance(target, Entity):
        return target.target
    return target
