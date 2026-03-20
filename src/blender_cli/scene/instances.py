"""Instances — GPU-instanced geometry via EXT_mesh_gpu_instancing."""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, cast

import bpy

from blender_cli.types import EntityMetadata, PropValue, Vec3, Vec3Like, as_vec3

if TYPE_CHECKING:
    from blender_cli.assets.prefab import Prefab
    from blender_cli.geometry.pointset import PointSet

_UNSET = object()


class Instances:
    """
    GPU-instanced geometry from a prefab and point set.

    Creates linked-duplicate Blender objects sharing mesh data,
    parented under an Empty.  Exports as ``EXT_mesh_gpu_instancing``
    in GLB when :meth:`Scene.save` is called.
    """

    __slots__ = (
        "_align",
        "_annotations",
        "_asset_id",
        "_asset_path",
        "_attrs",
        "_cells",
        "_material_ids",
        "_name",
        "_points",
        "_prefab",
        "_props",
        "_rotations",
        "_scales",
        "_tags",
    )

    # Align aliases: human-readable → axis face
    _ALIGN_ALIASES: dict[str, str] = {
        "center": "center",
        "bottom": "-z",
        "top": "+z",
    }

    def __init__(
        self,
        prefab: Prefab,
        points: list[Vec3],
        rotations: list[float],
        scales: list[float],
        attrs: dict[str, list[float | int | str]],
        cells: dict[str, list[int]] | None = None,
        align: str = "center",
        *,
        name: str | None = None,
        tags: set[str] | None = None,
        annotations: set[str] | None = None,
        props: dict[str, PropValue] | None = None,
        asset_id: str | None = None,
        asset_path: str | None = None,
        material_ids: set[str] | None = None,
    ) -> None:
        self._prefab = prefab
        self._points = list(points)
        self._rotations = list(rotations)
        self._scales = list(scales)
        self._attrs: dict[str, list[float | int | str]] = dict(attrs)
        self._cells = cells
        self._align = self._ALIGN_ALIASES.get(align, align)
        self._name = name
        self._tags = set(tags or set())
        self._annotations = set(annotations or set())
        self._props = dict(props or {})
        self._asset_id = asset_id
        self._asset_path = asset_path
        self._material_ids = set(material_ids or set())

    # -- factories ---------------------------------------------------------

    @classmethod
    def from_points(
        cls,
        prefab: Prefab,
        pointset: PointSet,
        attrs: tuple[str, ...] = (),
        align: str = "center",
    ) -> Instances:
        """
        Create an instanced set from a prefab and point positions/attributes.

        TRS (translation/rotation/scale) is always extracted from the
        PointSet's ``yaw`` and ``scale`` attributes when present.
        *attrs* lists additional per-point attributes to carry (e.g.
        ``('TRS', 'variant')``).  ``'TRS'`` is accepted but is a no-op
        since TRS is always extracted.

        *align* controls which face of the prefab's bounding box should be
        placed at the snapped surface.  This prevents meshes from clipping
        through floors, walls, or ceilings:

        - ``"center"`` (default): origin at snap point, no offset.
        - ``"bottom"`` / ``"-z"``: Z-min face at surface (furniture on floor).
        - ``"top"`` / ``"+z"``: Z-max face at surface (lamp hanging from ceiling).
        - ``"+y"`` / ``"-y"`` / ``"+x"`` / ``"-x"``: named face at surface
          (e.g. ``"+y"`` for poster back against a north wall).
        """
        points = pointset.points
        n = pointset.count

        # TRS from PointSet attrs (fall back to identity)
        rotations = _safe_attr(pointset, "yaw", [0.0] * n)
        scales = _safe_attr(pointset, "scale", [1.0] * n)

        # Carry all PointSet attrs except TRS components
        extra: dict[str, list[float | int | str]] = {}
        for name in pointset._attrs:
            if name in {"yaw", "scale"}:
                continue
            extra[name] = pointset.attr(name)

        # Carry only explicitly requested attrs that weren't already pulled
        for name in attrs:
            if name == "TRS" or name in extra:
                continue
            with contextlib.suppress(KeyError):
                extra[name] = pointset.attr(name)

        # Snap provenance
        if pointset.snap_results:
            extra["hit_uid"] = [sr.hit_uid or "" for sr in pointset.snap_results]
            extra["snap_axis"] = [
                sr.snap_axis if isinstance(sr.snap_axis, str) else str(sr.snap_axis)
                for sr in pointset.snap_results
            ]

        return cls(
            prefab,
            points,
            rotations,
            scales,
            extra,
            align=align,
            asset_id=prefab.path.name,
            asset_path=str(prefab.path.resolve()),
        )

    # -- properties --------------------------------------------------------

    @property
    def count(self) -> int:
        """Total instance count."""
        return len(self._points)

    @property
    def points(self) -> list[Vec3]:
        """Copy of instance positions."""
        return list(self._points)

    @property
    def metadata(self) -> EntityMetadata:
        """Fluent metadata pending scene.add()."""
        return EntityMetadata(
            name=self._name,
            tags=set(self._tags),
            annotations=set(self._annotations),
            props=dict(self._props),
            asset_id=self._asset_id,
            asset_path=self._asset_path,
            material_ids=sorted(self._material_ids),
        )

    def attr(self, name: str) -> list[float | int | str]:
        """Return per-instance attribute values."""
        if name not in self._attrs:
            msg = f"Attribute '{name}' not set"
            raise KeyError(msg)
        return list(self._attrs[name])

    def _copy(
        self,
        *,
        prefab: object = _UNSET,
        points: object = _UNSET,
        rotations: object = _UNSET,
        scales: object = _UNSET,
        attrs: object = _UNSET,
        cells: object = _UNSET,
        align: object = _UNSET,
        name: object = _UNSET,
        tags: object = _UNSET,
        annotations: object = _UNSET,
        props: object = _UNSET,
        asset_id: object = _UNSET,
        asset_path: object = _UNSET,
        material_ids: object = _UNSET,
    ) -> Instances:
        """Clone with selected fields replaced."""
        return Instances(
            cast("Prefab", self._prefab if prefab is _UNSET else prefab),
            cast("list[Vec3]", self._points if points is _UNSET else points),
            cast("list[float]", self._rotations if rotations is _UNSET else rotations),
            cast("list[float]", self._scales if scales is _UNSET else scales),
            cast(
                "dict[str, list[float | int | str]]",
                self._attrs if attrs is _UNSET else attrs,
            ),
            cast(
                "dict[str, list[int]] | None", self._cells if cells is _UNSET else cells
            ),
            align=cast("str", self._align if align is _UNSET else align),
            name=cast("str | None", self._name if name is _UNSET else name),
            tags=cast("set[str]", self._tags if tags is _UNSET else tags),
            annotations=cast(
                "set[str]", self._annotations if annotations is _UNSET else annotations
            ),
            props=cast(
                "dict[str, PropValue]", self._props if props is _UNSET else props
            ),
            asset_id=cast(
                "str | None", self._asset_id if asset_id is _UNSET else asset_id
            ),
            asset_path=cast(
                "str | None", self._asset_path if asset_path is _UNSET else asset_path
            ),
            material_ids=cast(
                "set[str]",
                self._material_ids if material_ids is _UNSET else material_ids,
            ),
        )

    # -- spatial partitioning ----------------------------------------------

    def partition(self, cell_size: float) -> Instances:
        """
        Split instances into spatial cells for culling/batching.

        Each cell becomes a separate instanced node in the exported GLB.
        """
        cells: dict[str, list[int]] = {}
        for i, p in enumerate(self._points):
            cx = math.floor(p.x / cell_size)
            cy = math.floor(p.y / cell_size)
            cells.setdefault(f"{cx}_{cy}", []).append(i)
        return self._copy(cells=cells)

    # -- fluent transforms / metadata ---------------------------------------

    def at(
        self,
        x: float | Vec3Like,
        y: float | None = None,
        z: float | None = None,
    ) -> Instances:
        """Translate instance set so its centroid is at *pos*. Accepts ``(x, y, z)`` or Vec3/tuple."""
        if y is None and z is None:
            target = as_vec3(x)  # type: ignore[arg-type]
        elif y is not None and z is not None:
            target = Vec3(float(x), float(y), float(z))  # type: ignore[arg-type]
        else:
            msg = "at() requires either (x, y, z) or a single Vec3/tuple"
            raise ValueError(msg)
        if not self._points:
            return self
        cx = sum(p.x for p in self._points) / len(self._points)
        cy = sum(p.y for p in self._points) / len(self._points)
        cz = sum(p.z for p in self._points) / len(self._points)
        delta = Vec3(target.x - cx, target.y - cy, target.z - cz)
        shifted = [p + delta for p in self._points]
        return self._copy(points=shifted)

    def yaw(self, yaw_deg: float) -> Instances:
        """Rotate instance positions around centroid and add yaw to each instance."""
        if not self._points:
            return self
        rad = math.radians(yaw_deg)
        c = math.cos(rad)
        s = math.sin(rad)
        cx = sum(p.x for p in self._points) / len(self._points)
        cy = sum(p.y for p in self._points) / len(self._points)
        rotated: list[Vec3] = []
        for p in self._points:
            lx, ly = p.x - cx, p.y - cy
            rx = lx * c - ly * s
            ry = lx * s + ly * c
            rotated.append(Vec3(cx + rx, cy + ry, p.z))
        rots = [r + yaw_deg for r in self._rotations]
        return self._copy(points=rotated, rotations=rots)

    def rot(
        self,
        x: float | Vec3 | tuple[float, float, float],
        y: float | None = None,
        z: float | None = None,
        *,
        degrees: bool = True,
    ) -> Instances:
        """
        Euler rotation helper (currently yaw-only for instance clouds).

        ``Instances`` stores per-point yaw, so X/Y rotations are not representable.
        This helper accepts the fluent ``rot(...)`` API and forwards Z to :meth:`yaw`.
        """
        if y is None and z is None:
            if isinstance(x, Vec3):
                rx, ry, rz = x.x, x.y, x.z
            elif isinstance(x, tuple):
                if len(x) != 3:
                    msg = "rot() tuple must have 3 values"
                    raise ValueError(msg)
                rx, ry, rz = float(x[0]), float(x[1]), float(x[2])
            else:
                msg = "rot() needs (x,y,z) or a Vec3/3-tuple"
                raise ValueError(msg)
        else:
            if y is None or z is None:
                msg = "rot() needs either (x,y,z) or a single Vec3/tuple"
                raise ValueError(msg)
            if not isinstance(x, (int, float)):
                msg = "rot(x,y,z) expects numeric x when y/z are provided"
                raise TypeError(msg)
            rx, ry, rz = float(x), float(y), float(z)

        if not degrees:
            rx = math.degrees(rx)
            ry = math.degrees(ry)
            rz = math.degrees(rz)

        if abs(rx) > 1e-9 or abs(ry) > 1e-9:
            msg = "Instances.rot() only supports Z-axis rotation (yaw) for now"
            raise ValueError(msg)
        return self.yaw(rz)

    def apply_scale(self, sx: float, sy: float, sz: float) -> Instances:
        """Scale instance cloud around centroid and multiply per-instance scales."""
        if not self._points:
            return self
        cx = sum(p.x for p in self._points) / len(self._points)
        cy = sum(p.y for p in self._points) / len(self._points)
        cz = sum(p.z for p in self._points) / len(self._points)
        scaled_pts = [
            Vec3(
                cx + (p.x - cx) * sx,
                cy + (p.y - cy) * sy,
                cz + (p.z - cz) * sz,
            )
            for p in self._points
        ]
        # Instances store scalar uniform scale; use mean factor for compatibility.
        f = (sx + sy + sz) / 3.0
        scaled = [s * f for s in self._scales]
        return self._copy(points=scaled_pts, scales=scaled)

    def scale(self, value: float) -> Instances:
        """Uniform scale around centroid."""
        return self.apply_scale(value, value, value)

    def tag(self, *values: str) -> Instances:
        tags = set(self._tags)
        tags.update(v for v in values if v)
        return self._copy(tags=tags)

    def annotate(self, *values: str) -> Instances:
        anns = set(self._annotations)
        anns.update(v for v in values if v)
        return self._copy(annotations=anns)

    def props(self, **values: PropValue) -> Instances:
        merged = dict(self._props)
        merged.update(values)
        return self._copy(props=merged)

    def name(self, value: str) -> Instances:
        return self._copy(name=value)

    def named(self, value: str) -> Instances:
        """Alias for :meth:`name` — matches Entity.named() convention."""
        return self.name(value)

    def asset(self, asset_id: str, path: str | Path | None = None) -> Instances:
        return self._copy(asset_id=asset_id, asset_path=str(path) if path else None)

    # -- Blender build -----------------------------------------------------

    def build(self, collection: bpy.types.Collection) -> list[bpy.types.Object]:
        """
        Create the Blender instancing hierarchy. Returns parent Empty(s).

        Each parent Empty holds linked duplicates sharing one mesh datablock.
        Blender's glTF exporter recognises this pattern and writes
        ``EXT_mesh_gpu_instancing``.
        """
        # Import prefab to get mesh data
        prefab_col = self._prefab.load("_inst_prefab")
        mesh_data: bpy.types.Mesh | None = None
        for obj in prefab_col.objects:
            if obj.type == "MESH" and obj.data:
                mesh_data = obj.data
                break

        if mesh_data is None:
            msg = "Prefab contains no mesh objects"
            raise ValueError(msg)

        # Clean up the prefab import (we only need the mesh datablock)
        for obj in list(prefab_col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        bpy.data.collections.remove(prefab_col)

        groups: list[tuple[str, list[int]]]
        if self._cells:
            groups = list(self._cells.items())
        else:
            groups = [("instances", list(range(len(self._points))))]

        parents: list[bpy.types.Object] = []
        for group_name, indices in groups:
            parent = self._build_group(mesh_data, collection, indices, group_name)
            parents.append(parent)

        return parents

    @staticmethod
    def _compute_align_offset(
        mesh_data: bpy.types.Mesh,
        align: str,
    ) -> tuple[float, float, float]:
        """
        Return (dx, dy, dz) offset per unit-scale for the given align mode.

        The offset is computed so that the named bounding-box face sits at
        the snapped position instead of the mesh center.
        """
        if align == "center":
            return (0.0, 0.0, 0.0)

        # bound_box is on Object, not Mesh — compute from vertices
        verts = mesh_data.vertices
        if not verts:
            return (0.0, 0.0, 0.0)
        xs = [v.co.x for v in verts]
        ys = [v.co.y for v in verts]
        zs = [v.co.z for v in verts]
        bb_min = (min(xs), min(ys), min(zs))
        bb_max = (max(xs), max(ys), max(zs))

        # Map align string to the bbox extreme that should be at the surface.
        # Offset = -(extreme) so that  origin + extreme * scale == surface.
        if align == "-z":
            return (0.0, 0.0, -bb_min[2])
        if align == "+z":
            return (0.0, 0.0, -bb_max[2])
        if align == "-y":
            return (0.0, -bb_min[1], 0.0)
        if align == "+y":
            return (0.0, -bb_max[1], 0.0)
        if align == "-x":
            return (-bb_min[0], 0.0, 0.0)
        if align == "+x":
            return (-bb_max[0], 0.0, 0.0)

        msg = (
            f"Unknown align mode {align!r}. "
            f"Expected: center, bottom, top, -z, +z, -y, +y, -x, +x"
        )
        raise ValueError(msg)

    def _build_group(
        self,
        mesh_data: bpy.types.Mesh,
        collection: bpy.types.Collection,
        indices: list[int],
        name: str,
    ) -> bpy.types.Object:
        """Build one parent Empty with linked-duplicate children."""
        # Create parent Empty
        parent = bpy.data.objects.new(name, None)
        parent.empty_display_type = "PLAIN_AXES"
        parent.empty_display_size = 0.5
        collection.objects.link(parent)

        # Compute surface alignment offset (per unit scale)
        odx, ody, odz = self._compute_align_offset(mesh_data, self._align)

        # Create linked duplicates
        for i in indices:
            p = self._points[i]
            yaw_deg = self._rotations[i]
            s = self._scales[i]

            child = bpy.data.objects.new(f"{name}_{i}", mesh_data)
            child.location = (p.x + odx * s, p.y + ody * s, p.z + odz * s)
            child.rotation_euler = (0.0, 0.0, math.radians(yaw_deg))
            child.scale = (s, s, s)
            child.parent = parent
            collection.objects.link(child)

        # Store per-instance columnar attrs in extras on the parent
        if self._attrs:
            # Only store attrs for the indices in this group
            group_attrs: dict[str, list[float | int | str]] = {}
            for attr_name, values in self._attrs.items():
                group_attrs[attr_name] = [values[i] for i in indices]
            parent["_instance_attrs"] = json.dumps(group_attrs)

        parent["_instance_count"] = len(indices)
        return parent


def _safe_attr(pointset: PointSet, name: str, default: list[float]) -> list[float]:
    """Get a PointSet attr, returning *default* if missing."""
    try:
        return [float(v) for v in pointset.attr(name)]
    except KeyError:
        return default
