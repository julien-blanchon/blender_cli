"""Core types for the maps-creation SDK."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from blender_cli.render.camera import Camera

# JSON-like scalar and prop value types (also exported from build.build_types).
type JsonScalar = str | int | float | bool | None
type PropValue = JsonScalar | list[JsonScalar]
type PropDict = dict[str, PropValue]


@runtime_checkable
class SupportsVec3(Protocol):
    """Protocol for objects that can expose a world-space Vec3."""

    def to_vec3(self) -> Vec3:
        """Return the object's position as a :class:`Vec3`."""
        ...


type Vec3Like = Vec3 | tuple[float, float, float] | list[float] | SupportsVec3


def as_vec3(value: Vec3Like) -> "Vec3":
    """Coerce tuples/lists/Vec3 into a :class:`Vec3`."""
    if isinstance(value, Vec3):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) != 3:
            msg = f"Expected 3 values for Vec3, got {len(value)}"
            raise ValueError(msg)
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    if isinstance(value, SupportsVec3):
        return value.to_vec3()
    msg = f"Expected Vec3 or 3-tuple/list, got {type(value).__name__}"
    raise TypeError(msg)


class Vec3OpsMixin(ABC):
    """Reusable XYZ arithmetic for any object that can expose a Vec3."""

    @abstractmethod
    def to_vec3(self) -> Vec3:
        """Return this object's world-space position."""
        raise NotImplementedError

    def __add__(self, other: Vec3Like) -> Vec3:
        s = self.to_vec3()
        o = as_vec3(other)
        return Vec3(s.x + o.x, s.y + o.y, s.z + o.z)

    def __radd__(self, other: Vec3Like) -> Vec3:
        s = self.to_vec3()
        o = as_vec3(other)
        return Vec3(o.x + s.x, o.y + s.y, o.z + s.z)

    def __sub__(self, other: Vec3Like) -> Vec3:
        s = self.to_vec3()
        o = as_vec3(other)
        return Vec3(s.x - o.x, s.y - o.y, s.z - o.z)

    def __rsub__(self, other: Vec3Like) -> Vec3:
        s = self.to_vec3()
        o = as_vec3(other)
        return Vec3(o.x - s.x, o.y - s.y, o.z - s.z)

    def distance(self, other: Vec3Like) -> float:
        """Euclidean distance to *other*."""
        s = self.to_vec3()
        o = as_vec3(other)
        dx = s.x - o.x
        dy = s.y - o.y
        dz = s.z - o.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def distance_to(self, other: Vec3Like) -> float:
        """Alias for :meth:`distance` used by non-Vec3 wrappers (e.g. Anchor)."""
        return self.distance(other)

    def direction_to(self, other: Vec3Like) -> Vec3:
        """Unit direction vector toward *other*."""
        return (as_vec3(other) - self.to_vec3()).normalized()

    def midpoint(self, other: Vec3Like) -> Vec3:
        """Midpoint between this position and *other*."""
        return self.to_vec3().lerp(other, 0.5)

    def offset(
        self,
        dx: float | Vec3Like,
        dy: float | None = None,
        dz: float | None = None,
    ) -> Vec3:
        """Offset this position by either a vector or explicit deltas."""
        if dy is None and dz is None:
            if isinstance(dx, (int, float)):
                msg = (
                    "offset() needs either (dx,dy,dz) or a single Vec3/tuple-like value"
                )
                raise ValueError(msg)
            return self.to_vec3() + as_vec3(dx)
        if dy is None or dz is None:
            msg = "offset() needs either (dx,dy,dz) or a single Vec3/tuple-like value"
            raise ValueError(msg)
        if not isinstance(dx, (int, float)):
            msg = "offset(dx,dy,dz) expects numeric dx when dy/dz are provided"
            raise TypeError(msg)
        return self.to_vec3() + Vec3(float(dx), float(dy), float(dz))


@dataclass(frozen=True, slots=True)
class Vec3(Vec3OpsMixin):
    """Immutable 3D vector with basic arithmetic, distance, and interpolation."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_vec3(self) -> Vec3:
        return self

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> Vec3:
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vec3:
        return Vec3(self.x / scalar, self.y / scalar, self.z / scalar)

    def __neg__(self) -> Vec3:
        return Vec3(-self.x, -self.y, -self.z)

    # -- geometry --

    def length(self) -> float:
        """Magnitude of the vector."""
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def lerp(self, other: Vec3Like, t: float) -> Vec3:
        """Linear interpolation towards *other* by factor *t* (0→self, 1→other)."""
        o = as_vec3(other)
        return Vec3(
            self.x + (o.x - self.x) * t,
            self.y + (o.y - self.y) * t,
            self.z + (o.z - self.z) * t,
        )

    def dot(self, other: Vec3Like) -> float:
        """Dot product."""
        o = as_vec3(other)
        return self.x * o.x + self.y * o.y + self.z * o.z

    def cross(self, other: Vec3Like) -> Vec3:
        """Cross product."""
        o = as_vec3(other)
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def normalized(self) -> Vec3:
        """Unit vector in the same direction. Returns zero vector if length is 0."""
        ln = self.length()
        if math.isclose(ln, 0.0):
            return Vec3()
        return self / ln

    def component(self, idx: int) -> float:
        """Get component by index (0=x, 1=y, 2=z)."""
        if idx == 0:
            return self.x
        if idx == 1:
            return self.y
        if idx == 2:
            return self.z
        msg = f"Vec3 component index must be 0, 1, or 2, got {idx}"
        raise IndexError(msg)

    def with_component(self, idx: int, value: float) -> Vec3:
        """Return a new Vec3 with one component replaced."""
        if idx == 0:
            return Vec3(value, self.y, self.z)
        if idx == 1:
            return Vec3(self.x, value, self.z)
        if idx == 2:
            return Vec3(self.x, self.y, value)
        msg = f"Vec3 component index must be 0, 1, or 2, got {idx}"
        raise IndexError(msg)


class NearbyEntry(TypedDict):
    """Single nearby-object entry returned in :class:`AddResult`."""

    uid: str
    name: str
    distance: float
    tags: list[str]


class EntityMetadata(TypedDict, total=False):
    """Metadata dict returned by Entity.metadata / Instances.metadata."""

    name: str | None
    tags: set[str]
    annotations: set[str]
    props: dict[str, PropValue]
    asset_id: str | None
    asset_path: str | None
    material_ids: list[str]
    bevy_components: dict[str, object]


class RenderSpec(TypedDict, total=False):
    """Spec dict passed to :meth:`RenderContext.batch`."""

    type: str
    out: str
    preset: str
    where: str
    hide_tags: set[str]
    show_tags: set[str]
    hide_where: str
    highlight_where: str
    ghost_opacity: float
    camera: Camera | None


class RegistryUsage(TypedDict):
    """Usage tracking stored in Blender scene custom properties."""

    prefabs: list[dict[str, str]]
    materials: list[str]


class ValidationIssue(TypedDict, total=False):
    """
    Issue dict returned by :meth:`Scene.validate`.

    Known issue types:
    - ``"zero_dimensions"`` — object has zero bounding-box diagonal.
    - ``"too_small"`` / ``"too_large"`` — object diag vs scene diag out of range.
    - ``"no_uv_layer"`` — textured mesh has no UV layer.
    - ``"uv_stretch"`` — worst-face UV stretch ratio exceeds threshold.
    """

    object: str
    issue: str
    dims: tuple[float, float, float]
    diag: float
    scene_diag: float
    ratio: float
    max_stretch: float
    face_count: int


class RandomChoiceSpec(TypedDict, total=False):
    """Spec for weighted random variant selection in PointSet.randomize()."""

    variant: list[str]
    weights: list[float]


@dataclass(frozen=True, slots=True)
class AddResult:
    """
    Structured feedback returned by :meth:`Scene.add`.

    Gives the agent full context about what happened: where the object
    ended up, what's nearby, and any placement warnings.
    """

    uid: str
    """Unique identifier assigned to the added object."""
    name: str
    """Object name in the scene."""
    position: Vec3
    """World-space position of the object origin."""
    bbox: tuple[Vec3, Vec3]
    """Axis-aligned bounding box ``(min, max)`` in world space."""
    nearby: list[NearbyEntry] = field(default_factory=list)
    """Up to 5 closest objects within 20 m."""
    warnings: list[str] = field(default_factory=list)
    """Placement warnings (overlap, out-of-bounds, scale)."""
