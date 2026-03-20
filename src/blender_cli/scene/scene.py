"""Scene — core CRUD and glTF serialization."""

from __future__ import annotations

import json
import math
import operator
from datetime import UTC, datetime
from pathlib import Path

# TYPE_CHECKING avoids circular import (instances imports Prefab)
from typing import TYPE_CHECKING, TypeVar, cast, overload
from uuid import uuid4

import bpy
import mathutils

from blender_cli.core.diagnostics import logger
from blender_cli.core.metadata import (
    KEY_ANNOTATIONS,
    KEY_ASSET_ID,
    KEY_ASSET_PATH,
    KEY_MATERIAL_IDS,
    KEY_PROPS,
    KEY_TAGS,
    KEY_UID,
    decode_dict,
    decode_json,
    decode_list,
    decode_set,
    encode_dict,
    encode_set,
)
from blender_cli.scene.anchor import Anchor
from blender_cli.scene.entity import Entity, unwrap_entity
from blender_cli.scene.selection import Selection, Transform, parse_query
from blender_cli.snap import FilteredScene
from blender_cli.types import (
    AddResult,
    EntityMetadata,
    NearbyEntry,
    PropValue,
    ValidationIssue,
    Vec3,
    Vec3Like,
    as_vec3,
)

K = TypeVar("K", bound=str)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from blender_cli.assets.registry import AssetRegistry, MaterialRegistry
    from blender_cli.render.camera import Camera
    from blender_cli.scene.entity import SnapSpec
    from blender_cli.scene.instances import Instances


class Scene:
    """
    Wraps a Blender scene with UID tracking, metadata, and GLB serialization.

    Every object added via :meth:`add` gets a stable UUID stored in the
    ``_uid`` custom property, along with optional tags, annotations, and props.
    All metadata survives GLB save/load round-trips.
    """

    __slots__ = ("_assets", "_col", "_materials", "_scene")

    def __init__(self, bpy_scene: bpy.types.Scene | None = None) -> None:
        scene = bpy_scene if bpy_scene is not None else bpy.context.scene
        if scene is None:
            msg = "No active Blender scene"
            raise RuntimeError(msg)
        self._scene: bpy.types.Scene = scene
        self._materials = None
        self._assets = None
        col = scene.collection
        if col is None:
            msg = "Scene has no root collection"
            raise RuntimeError(msg)
        self._col: bpy.types.Collection = col

    @classmethod
    def new(cls) -> "Scene":
        """Create a fresh empty scene (resets Blender to factory defaults)."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return cls()

    @property
    def bpy_scene(self) -> bpy.types.Scene:
        """Underlying Blender scene."""
        return self._scene

    @property
    def materials(self) -> MaterialRegistry:
        """Material registry + cache bound to this scene."""
        if self._materials is None:
            from blender_cli.assets.registry import MaterialRegistry

            self._materials = MaterialRegistry(self)
        return self._materials

    @property
    def assets(self) -> AssetRegistry:
        """Asset registry + prefab cache bound to this scene."""
        if self._assets is None:
            from blender_cli.assets.registry import AssetRegistry

            self._assets = AssetRegistry(self)
        return self._assets

    @property
    def a(self) -> "_AnchorAccessor":
        """Attribute-style anchor lookup: ``scene.a.gate``."""
        return _AnchorAccessor(self)

    # -- Dimension helpers --

    @staticmethod
    def _object_dimensions(obj: bpy.types.Object) -> Vec3:
        """World-space bounding box dimensions including scale."""
        if obj.type == "MESH" and obj.bound_box:
            corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            zs = [c.z for c in corners]
            return Vec3(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        if obj.children:
            all_xs: list[float] = []
            all_ys: list[float] = []
            all_zs: list[float] = []
            for child in obj.children:
                d = Scene._object_dimensions(child)
                loc = child.matrix_world.translation
                all_xs.extend([loc.x - d.x / 2, loc.x + d.x / 2])
                all_ys.extend([loc.y - d.y / 2, loc.y + d.y / 2])
                all_zs.extend([loc.z - d.z / 2, loc.z + d.z / 2])
            if all_xs:
                return Vec3(
                    max(all_xs) - min(all_xs),
                    max(all_ys) - min(all_ys),
                    max(all_zs) - min(all_zs),
                )
        return Vec3(0, 0, 0)

    # -- Feedback helpers --

    @staticmethod
    def _object_bbox(obj: bpy.types.Object) -> tuple[Vec3, Vec3]:
        """World-space AABB for *obj*. Returns ``(min, max)``."""
        if obj.type == "MESH" and obj.bound_box:
            corners = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
            xs = [c.x for c in corners]
            ys = [c.y for c in corners]
            zs = [c.z for c in corners]
            return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))
        loc = obj.matrix_world.translation
        return Vec3(loc.x, loc.y, loc.z), Vec3(loc.x, loc.y, loc.z)

    def _find_nearby(
        self,
        target: bpy.types.Object,
        radius: float = 20.0,
        limit: int = 5,
    ) -> list[NearbyEntry]:
        """Return the closest tracked objects within *radius* metres."""
        t_loc = target.matrix_world.translation
        t_pos = Vec3(t_loc.x, t_loc.y, t_loc.z)
        entries: list[tuple[float, NearbyEntry]] = []
        for obj in self.objects():
            if obj is target:
                continue
            o_loc = obj.matrix_world.translation
            dist = t_pos.distance(Vec3(o_loc.x, o_loc.y, o_loc.z))
            if dist <= radius:
                entries.append((
                    dist,
                    NearbyEntry(
                        uid=self.uid(obj) or "",
                        name=obj.name,
                        distance=round(dist, 2),
                        tags=sorted(self.tags(obj)),
                    ),
                ))
        entries.sort(key=operator.itemgetter(0))
        return [e[1] for e in entries[:limit]]

    @staticmethod
    def _bbox_overlaps(a: tuple[Vec3, Vec3], b: tuple[Vec3, Vec3]) -> bool:
        """True if two AABBs intersect (strict overlap, not just touching)."""
        a_min, a_max = a
        b_min, b_max = b
        return (
            a_min.x < b_max.x
            and a_max.x > b_min.x
            and a_min.y < b_max.y
            and a_max.y > b_min.y
            and a_min.z < b_max.z
            and a_max.z > b_min.z
        )

    def _check_placement(self, obj: bpy.types.Object) -> list[str]:
        """Return placement warnings for *obj* (overlap, out-of-bounds)."""
        warn: list[str] = []
        obj_bb = self._object_bbox(obj)

        # Overlap detection
        for other in self.objects():
            if other is obj:
                continue
            other_bb = self._object_bbox(other)
            if self._bbox_overlaps(obj_bb, other_bb):
                warn.append(f"Overlap: '{obj.name}' bbox intersects '{other.name}'")

        # Out-of-bounds check (scene bbox)
        scene_bb = self.bbox()
        if scene_bb is not None:
            s_min, s_max = scene_bb
            o_min, o_max = obj_bb
            if (
                o_min.x < s_min.x
                or o_min.y < s_min.y
                or o_max.x > s_max.x
                or o_max.y > s_max.y
            ):
                warn.append(
                    f"Out-of-bounds: '{obj.name}' extends beyond scene XY extents"
                )

        return warn

    # -- CRUD --

    def add(
        self,
        obj: bpy.types.Object | Instances | Entity | Camera,
        name: str | None = None,
        tags: set[str] | list[str] | None = None,
        annotations: set[str] | list[str] | None = None,
        props: dict[str, PropValue] | None = None,
    ) -> AddResult:
        """
        Add *obj* (or an :class:`Instances` set) to the scene with metadata.

        Returns an :class:`AddResult` with uid, position, bbox, nearby
        objects, and placement warnings.
        """
        from blender_cli.render.camera import Camera
        from blender_cli.scene.instances import Instances

        # Handle Camera objects — unwrap to bpy_object and merge metadata.
        if isinstance(obj, Camera):
            cam_tags = obj.tags
            if tags is not None:
                cam_tags = cam_tags | set(tags)
            cam_name = name or obj.name
            cam_props = obj.camera_props
            if props:
                cam_props.update(props)
            obj = obj.bpy_object
            name = cam_name  # type: ignore[assignment]  # narrowed from Camera
            tags = cam_tags
            props = cast("dict[str, PropValue] | None", cam_props or None)

        entity_meta: EntityMetadata | None = None
        source = obj
        if isinstance(obj, Entity):
            entity_meta = obj.metadata
            source = unwrap_entity(obj)

        if entity_meta is not None:
            if name is None:
                name = entity_meta.get("name")
            if tags is None:
                tags = entity_meta.get("tags")
            if annotations is None:
                annotations = entity_meta.get("annotations")
            if props is None:
                props = entity_meta.get("props")

        if isinstance(source, Instances):
            inst_meta = source.metadata if hasattr(source, "metadata") else {}
            if name is None:
                name = inst_meta.get("name")
            if tags is None:
                tags = inst_meta.get("tags")
            if annotations is None:
                annotations = inst_meta.get("annotations")
            if props is None:
                props = inst_meta.get("props")
            return self._add_instances(
                source,
                name,
                tags,
                annotations,
                props,
                asset_id=(entity_meta or {}).get("asset_id")
                or inst_meta.get("asset_id"),
                asset_path=(entity_meta or {}).get("asset_path")
                or inst_meta.get("asset_path"),
                material_ids=list((entity_meta or {}).get("material_ids", []))
                or list(inst_meta.get("material_ids", [])),
            )

        if not isinstance(source, bpy.types.Object):
            msg = f"Scene.add() expected bpy object/Instances/Entity, got {type(source).__name__}"
            raise TypeError(msg)

        uid = str(uuid4())
        if name is not None:
            source.name = name

        source[KEY_UID] = uid
        if tags:
            source[KEY_TAGS] = encode_set(tags)
        if annotations:
            source[KEY_ANNOTATIONS] = encode_set(annotations)
        if props:
            source[KEY_PROPS] = encode_dict(props, sort_keys=True)
        if entity_meta is not None:
            asset_id = entity_meta.get("asset_id")
            asset_path = entity_meta.get("asset_path")
            material_ids = entity_meta.get("material_ids", [])
            if asset_id:
                source[KEY_ASSET_ID] = str(asset_id)
            if asset_path:
                source[KEY_ASSET_PATH] = str(asset_path)
            if material_ids:
                source[KEY_MATERIAL_IDS] = encode_set(list(material_ids))
            # Write Blenvy/Bevy components as custom properties (→ GLTF extras).
            bevy_components = entity_meta.get("bevy_components", {})
            if bevy_components:
                from blender_cli.blenvy import apply_bevy_components

                apply_bevy_components(source, bevy_components)

        # Ensure the object is linked to the scene.
        try:
            self._col.objects.link(source)
        except RuntimeError:
            pass  # Already linked

        # Flush transform changes before deriving bbox/position diagnostics.
        view_layer = bpy.context.view_layer
        if view_layer is not None:
            view_layer.update()

        # -- Scale diagnostics --
        dims = self._object_dimensions(source)
        obj_diag = dims.length()
        scene_bb = self.bbox()
        scene_diag = (scene_bb[1] - scene_bb[0]).length() if scene_bb else 0.0
        ratio = obj_diag / scene_diag if scene_diag > 1e-9 else 0.0

        add_warnings: list[str] = []
        if scene_diag > 1e-9:
            dim_str = f"dims={dims.x:.1f}x{dims.y:.1f}x{dims.z:.1f}m"
            if ratio < 0.005:
                add_warnings.append(
                    f"Scale: '{obj.name}' may be too small "
                    f"({obj_diag:.2f}m vs scene {scene_diag:.0f}m, {dim_str}, ratio={ratio:.4f})"
                )
            elif ratio > 0.5:
                add_warnings.append(
                    f"Scale: '{obj.name}' may be too large "
                    f"({obj_diag:.2f}m vs scene {scene_diag:.0f}m, {dim_str}, ratio={ratio:.2f})"
                )

        # Placement checks
        add_warnings.extend(self._check_placement(source))

        # Build result
        loc = source.matrix_world.translation
        position = Vec3(float(loc.x), float(loc.y), float(loc.z))
        obj_bbox = self._object_bbox(source)
        nearby = self._find_nearby(source)

        result = AddResult(
            uid=uid,
            name=source.name,
            position=position,
            bbox=obj_bbox,
            nearby=nearby,
            warnings=add_warnings,
        )

        logger.debug(
            "[add] '%s' pos=(%.1f, %.1f, %.1f) dims=(%.1f, %.1f, %.1f)m nearby=%d warnings=%d",
            source.name,
            position.x,
            position.y,
            position.z,
            dims.x,
            dims.y,
            dims.z,
            len(nearby),
            len(add_warnings),
        )

        return result

    def add_batch(
        self,
        entities: list[Entity],
        *,
        tags: set[str] | list[str] | None = None,
        snap: "SnapSpec | None" = None,
    ) -> list[AddResult]:
        """
        Add multiple entities to the scene, optionally applying shared *tags* and *snap*.

        Each entity's own tags are merged with the shared *tags*.
        If *snap* is provided, each entity is snapped before adding.
        """
        results: list[AddResult] = []
        for ent in entities:
            if tags:
                existing = ent.metadata.get("tags") or set()
                merged = set(existing) | set(tags)
                ent.tag(*sorted(merged))
            if snap is not None:
                ent.snap(self, spec=snap)
            results.append(self.add(ent))
        return results

    def record_rng(self, stream: str, seed: int) -> None:
        """Record an RNG seed for deterministic replay in the scene manifest."""
        from blender_cli.core.metadata import decode_json, encode_json

        raw = self.bpy_scene.get("_mc_rng_streams")
        entries: list[dict[str, object]]
        payload = decode_json(raw, default=[])
        entries = list(payload) if isinstance(payload, list) else []
        entry: dict[str, object] = {"stream": stream, "seed": int(seed)}
        if entry not in entries:
            entries.append(entry)
        self.bpy_scene["_mc_rng_streams"] = encode_json(entries, sort_keys=True)

    def _add_instances(
        self,
        inst: Instances,
        name: str | None,
        tags: set[str] | list[str] | None,
        annotations: set[str] | list[str] | None,
        props: dict[str, PropValue] | None,
        asset_id: str | None = None,
        asset_path: str | None = None,
        material_ids: list[str] | None = None,
    ) -> AddResult:
        """Build Instances into Blender objects and attach metadata."""
        parents = inst.build(self._col)
        first_uid = ""
        first_parent: bpy.types.Object | None = None
        n_instances = 0
        for i, parent in enumerate(parents):
            obj_name = (
                name
                if name and len(parents) == 1
                else (f"{name}_{i}" if name else parent.name)
            )
            uid = str(uuid4())
            parent.name = obj_name
            parent[KEY_UID] = uid
            parent["_type"] = "instances"
            if tags:
                parent[KEY_TAGS] = encode_set(tags)
            if annotations:
                parent[KEY_ANNOTATIONS] = encode_set(annotations)
            if props:
                parent[KEY_PROPS] = encode_dict(props, sort_keys=True)
            if asset_id:
                parent[KEY_ASSET_ID] = str(asset_id)
            if asset_path:
                parent[KEY_ASSET_PATH] = str(asset_path)
            if material_ids:
                parent[KEY_MATERIAL_IDS] = encode_set(material_ids)
            if not first_uid:
                first_uid = uid
                first_parent = parent
            n_instances += len(parent.children)

        label = name or "instances"
        logger.debug(
            "[add] '%s' instances=%d parents=%d", label, n_instances, len(parents)
        )

        # Build AddResult from first parent
        if first_parent is not None:
            loc = first_parent.matrix_world.translation
            position = Vec3(float(loc.x), float(loc.y), float(loc.z))
            obj_bbox = self._object_bbox(first_parent)
        else:
            position = Vec3()
            obj_bbox = (Vec3(), Vec3())

        return AddResult(
            uid=first_uid,
            name=label,
            position=position,
            bbox=obj_bbox,
        )

    def objects(self) -> list[bpy.types.Object]:
        """All scene objects that have a ``_uid``."""
        return [obj for obj in self._col.all_objects if KEY_UID in obj]

    # -- Anchors --

    def ensure_anchor(
        self,
        name: str,
        annotation: str,
        location: Vec3Like = Vec3(),
    ) -> Anchor:
        """
        Create an anchor Empty if it doesn't exist, or return existing one.

        Idempotent — safe to call repeatedly in re-runnable build steps.
        Matches by name or annotation.
        """
        for obj in self._col.all_objects:
            if obj.get("_type") == "anchor" and (
                obj.name == name or obj.get("_annotation") == annotation
            ):
                return Anchor(obj)

        loc = as_vec3(location)
        bpy.ops.object.empty_add(location=(loc.x, loc.y, loc.z))
        obj = bpy.context.active_object
        if obj is None:
            msg = f"Blender failed to create empty for anchor {name!r}"
            raise RuntimeError(msg)
        self.add(obj, name=name, annotations={annotation})
        obj["_type"] = "anchor"
        obj["_annotation"] = annotation
        return Anchor(obj)

    def anchor(self, name_or_annotation: str) -> Anchor | None:
        """Retrieve an existing anchor by name or annotation."""
        for obj in self._col.all_objects:
            if obj.get("_type") == "anchor" and (
                obj.name == name_or_annotation
                or obj.get("_annotation") == name_or_annotation
            ):
                return Anchor(obj)
        return None

    @overload
    def anchors(self, specs: None = None) -> list[Anchor]: ...
    @overload
    def anchors(self, specs: Mapping[K, tuple[str, Vec3Like]]) -> dict[K, Anchor]: ...

    def anchors(
        self,
        specs: Mapping[K, tuple[str, Vec3Like]] | None = None,
    ) -> list[Anchor] | dict[K, Anchor]:
        if specs is None:
            return [
                Anchor(obj)
                for obj in self._col.all_objects
                if obj.get("_type") == "anchor"
            ]

        created: dict[K, Anchor] = {}
        for key, (annotation, location) in specs.items():
            created[key] = self.ensure_anchor(str(key), annotation, location)
        return created

    # -- Camera queries -------------------------------------------------------

    def camera(self, name: str | None = None) -> "Camera | None":
        """
        Get a camera by name, or the first camera if *name* is ``None``.

        Returns ``None`` if no matching camera is found.
        """
        from blender_cli.render.camera import Camera

        for obj in self._col.all_objects:
            if obj.type == "CAMERA" and (name is None or obj.name == name):
                cam_data = obj.data
                if isinstance(cam_data, bpy.types.Camera):
                    return Camera(cam_data, obj)
        return None

    def cameras(self) -> "list[Camera]":
        """Return all cameras in the scene."""
        from blender_cli.render.camera import Camera

        result: list[Camera] = []
        for obj in self._col.all_objects:
            if obj.type == "CAMERA":
                cam_data = obj.data
                if isinstance(cam_data, bpy.types.Camera):
                    result.append(Camera(cam_data, obj))
        return result

    # -- Metadata access (static — work on any bpy object) --

    @staticmethod
    def uid(obj: bpy.types.Object) -> str | None:
        """UID of *obj*, or ``None``."""
        v = obj.get(KEY_UID)
        return str(v) if v is not None else None

    @staticmethod
    def tags(obj: bpy.types.Object) -> set[str]:
        """Tags stored on *obj*."""
        return decode_set(obj.get(KEY_TAGS))

    @staticmethod
    def annotations(obj: bpy.types.Object) -> set[str]:
        """Annotations stored on *obj*."""
        return decode_set(obj.get(KEY_ANNOTATIONS))

    @staticmethod
    def props(obj: bpy.types.Object) -> dict[str, object]:
        """Props dict stored on *obj*."""
        return decode_dict(obj.get(KEY_PROPS))

    # -- Visibility profiles --

    def _get_visibility_profiles(self) -> dict[str, object]:
        return decode_dict(self._scene.get("_visibility_profiles"))

    def create_visibility_profile(
        self,
        name: str,
        hide_tags: set[str] | list[str] | None = None,
        hide_annotations: set[str] | list[str] | None = None,
        hide_names: set[str] | list[str] | None = None,
    ) -> None:
        """
        Create a named visibility profile for toggling object visibility.

        Objects matching any of *hide_tags*, *hide_annotations*, or *hide_names*
        will be hidden when the profile is applied.  Profiles are stored as
        scene metadata and survive GLB round-trips.
        """
        profiles = self._get_visibility_profiles()
        profiles[name] = {
            "hide_tags": sorted(hide_tags or []),
            "hide_annotations": sorted(hide_annotations or []),
            "hide_names": sorted(hide_names or []),
        }
        self._scene["_visibility_profiles"] = encode_dict(profiles, sort_keys=True)

    def apply_visibility_profile(self, name: str) -> None:
        """
        Hide objects matching the named profile.

        Saves current visibility state so :meth:`clear_visibility_profile`
        can restore it exactly.
        """
        profiles = self._get_visibility_profiles()
        if name not in profiles:
            msg = f"Visibility profile {name!r} not found"
            raise KeyError(msg)

        profile = cast("dict[str, object]", profiles[name])
        tag_set = set(profile.get("hide_tags", []))  # type: ignore[arg-type]
        ann_set = set(profile.get("hide_annotations", []))  # type: ignore[arg-type]
        name_set = set(profile.get("hide_names", []))  # type: ignore[arg-type]

        # Save current state before modifying
        saved: dict[str, dict[str, bool]] = {}
        for obj in self.objects():
            uid = self.uid(obj)
            if uid:
                saved[uid] = {
                    "hr": bool(obj.hide_render),
                    "hv": bool(obj.hide_viewport),
                }
        self._scene["_visibility_saved"] = encode_dict(saved, sort_keys=True)

        # Apply visibility
        for obj in self.objects():
            should_hide = bool(
                (tag_set and self.tags(obj) & tag_set)
                or (ann_set and self.annotations(obj) & ann_set)
                or (name_set and obj.name in name_set)
            )
            if should_hide:
                obj.hide_render = True
                obj.hide_viewport = True

    def clear_visibility_profile(self) -> None:
        """Restore object visibility from saved state."""
        raw = self._scene.get("_visibility_saved")
        if raw is None:
            for obj in self.objects():
                obj.hide_render = False
                obj.hide_viewport = False
            return

        saved = decode_dict(raw)
        uid_map = {self.uid(o): o for o in self.objects()}
        for uid, state in saved.items():
            obj = uid_map.get(uid)
            if obj is not None and isinstance(state, dict):
                obj.hide_render = bool(state.get("hr", False))
                obj.hide_viewport = bool(state.get("hv", False))

        if "_visibility_saved" in self._scene:
            del self._scene["_visibility_saved"]

    # -- World / Environment --

    def set_world_hdri(
        self,
        path: str | Path,
        rotation: float = 0.0,
        strength: float = 1.0,
    ) -> None:
        """
        Set up HDRI environment texture on the world.

        The HDRI sky will be visible when rendered (``film_transparent`` is
        automatically disabled by :class:`RenderContext`).
        """
        from blender_cli.render import setup_hdri_world

        setup_hdri_world(self._scene, path, rotation, strength)

    # -- Serialization --

    def _build_manifest(self, path: Path) -> dict[str, object]:
        """
        Build a manifest dict describing the current scene state.

        The manifest captures everything an agent needs to understand the
        scene without loading the GLB: anchors, object counts, spatial
        extent, and generation history.
        """
        # Anchors
        anchor_list = []
        for a in self.anchors():
            pos = a.location()
            anchor_list.append({
                "name": a.name,
                "annotation": a.annotation,
                "position": [round(pos.x, 4), round(pos.y, 4), round(pos.z, 4)],
            })

        # Per-object records and summary
        objs = self.objects()
        object_records: list[dict[str, object]] = []
        by_tag: dict[str, int] = {}
        by_role: dict[str, int] = {}
        for obj in objs:
            obj_tags = sorted(self.tags(obj))
            obj_annotations = sorted(self.annotations(obj))
            obj_props = self.props(obj)
            for tag in obj_tags:
                by_tag[tag] = by_tag.get(tag, 0) + 1
            obj_type = obj.get("_type")
            role = str(obj_type) if obj_type else "object"
            by_role[role] = by_role.get(role, 0) + 1

            loc = obj.matrix_world.translation
            rot = obj.rotation_euler
            scl = obj.scale
            record: dict[str, object] = {
                "uid": self.uid(obj) or "",
                "name": obj.name,
                "type": obj.type,
                "tags": obj_tags,
                "annotations": obj_annotations,
                "props": obj_props,
                "location": [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
                "rotation": [round(rot.x, 4), round(rot.y, 4), round(rot.z, 4)],
                "scale": [round(scl.x, 4), round(scl.y, 4), round(scl.z, 4)],
            }
            parent = obj.parent
            if parent is not None:
                record["parent_uid"] = self.uid(parent) or ""
            asset_id = obj.get(KEY_ASSET_ID)
            asset_path = obj.get(KEY_ASSET_PATH)
            if asset_id is not None:
                record["asset_id"] = str(asset_id)
            if asset_path is not None:
                record["asset_path"] = str(asset_path)
            object_records.append(record)

        # Spatial extent
        bb = self.bbox()
        spatial_extent: dict[str, list[float]] | None = None
        map_resolution: dict[str, float | list[float]] | None = None
        if bb is not None:
            spatial_extent = {
                "min": [round(bb[0].x, 4), round(bb[0].y, 4), round(bb[0].z, 4)],
                "max": [round(bb[1].x, 4), round(bb[1].y, 4), round(bb[1].z, 4)],
            }
            map_resolution = {
                "width": round(bb[1].x - bb[0].x, 4),
                "height": round(bb[1].y - bb[0].y, 4),
                "z_range": [round(bb[0].z, 4), round(bb[1].z, 4)],
            }

        # Generation step from filename stem
        generation_step = path.stem

        # Load existing manifest from scene extras to preserve generation_history
        history: list[dict[str, str]] = []
        existing_str = self.bpy_scene.get("_scene_manifest")
        if existing_str:
            try:
                existing = json.loads(existing_str)
                history = existing.get("generation_history", [])
            except (json.JSONDecodeError, TypeError):
                pass
        history.append({
            "step": generation_step,
            "timestamp": datetime.now(UTC).isoformat(),
        })

        # Asset usage (prefabs/material ids) and deterministic RNG streams.
        prefabs: dict[str, dict[str, str]] = {}
        material_ids: set[str] = set()
        for obj in objs:
            asset_id = obj.get(KEY_ASSET_ID)
            asset_path = obj.get(KEY_ASSET_PATH)
            if asset_id is not None or asset_path is not None:
                key = str(asset_id) if asset_id is not None else str(asset_path)
                prefabs[key] = {
                    "id": str(asset_id) if asset_id is not None else "",
                    "path": str(asset_path) if asset_path is not None else "",
                }

            raw_mids = obj.get(KEY_MATERIAL_IDS)
            material_ids.update(decode_list(raw_mids))

        usage_raw = self._scene.get("_mc_registry_usage")
        usage = decode_dict(usage_raw)
        raw_prefabs = usage.get("prefabs", [])
        if isinstance(raw_prefabs, list):
            for p in raw_prefabs:
                if not isinstance(p, dict):
                    continue
                pid = str(p.get("id", ""))
                ppath = str(p.get("path", ""))
                prefabs[pid or ppath] = {"id": pid, "path": ppath}
        raw_materials = usage.get("materials", [])
        if isinstance(raw_materials, list):
            material_ids.update(str(mid) for mid in raw_materials)

        rng_streams: list[dict[str, object]] = []
        rng_raw = self._scene.get("_mc_rng_streams")
        payload = decode_json(rng_raw, default=[])
        if isinstance(payload, list):
            rng_streams = [dict(item) for item in payload if isinstance(item, dict)]

        return {
            "version": "1.1",
            "generation_step": generation_step,
            "timestamp": datetime.now(UTC).isoformat(),
            "map_resolution": map_resolution,
            "anchors": anchor_list,
            "objects": object_records,
            "object_summary": {
                "total": len(objs),
                "by_tag": by_tag,
                "by_role": by_role,
            },
            "spatial_extent": spatial_extent,
            "assets": {
                "prefabs": sorted(
                    prefabs.values(), key=operator.itemgetter("id", "path")
                ),
                "materials": sorted(material_ids),
            },
            "rng_streams": rng_streams,
            "generation_history": history,
        }

    def set_component(
        self,
        obj: bpy.types.Object,
        name: str,
        value: object = None,
    ) -> None:
        """Set a Bevy/Blenvy component on an existing scene object.

        This writes the component as a Blender custom property (RON string)
        so it appears in the GLTF node extras on export.

        Args:
            obj: The Blender object already in the scene.
            name: Bevy component type name (e.g. ``"RigidBody"``).
            value: Component data — same rules as :meth:`Entity.component`.
        """
        from blender_cli.blenvy import apply_bevy_components

        apply_bevy_components(obj, {name: value})

    def save(self, path: str | Path, *, blenvy_meta: bool = False) -> None:
        """
        Export the scene as GLB with all extras preserved.

        The scene manifest is embedded as a scene-level ``_scene_manifest``
        extra inside the GLB — no sidecar file needed.

        Args:
            blenvy_meta: If ``True``, also write a ``.meta.ron`` sidecar file
                next to the GLB for Blenvy's blueprint asset preloading.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Build manifest and embed in scene-level extras inside the GLB
        manifest = self._build_manifest(p)
        self.bpy_scene["_scene_manifest"] = json.dumps(manifest)

        # Strip SDK-internal metadata properties (prefixed with _) from all
        # objects before GLTF export.  Blenvy tries to parse every custom
        # property as a Bevy component and spams warnings for _uid, _tags, etc.
        for obj in self._col.all_objects:
            to_del = [k for k in obj.keys() if k.startswith("_")]
            for k in to_del:
                del obj[k]
        # Also strip from the scene itself (keeps _scene_manifest above)
        scene_del = [
            k for k in self.bpy_scene.keys()
            if k.startswith("_") and k != "_scene_manifest"
        ]
        for k in scene_del:
            del self.bpy_scene[k]

        bpy.ops.export_scene.gltf(
            filepath=str(p),
            export_format="GLB",
            export_extras=True,
            export_cameras=True,
            export_lights=True,
            export_gpu_instances=True,
        )

        if blenvy_meta:
            self._write_blenvy_meta(p)

    def _write_blenvy_meta(self, glb_path: Path) -> None:
        """Write a Blenvy ``.meta.ron`` sidecar listing blueprint dependencies."""
        meta_path = glb_path.with_suffix(".meta.ron")
        # Collect BlueprintInfo references from object extras.
        assets: list[tuple[str, str]] = []
        for obj in self._col.objects:
            bp = obj.get("BlueprintInfo")
            if bp and isinstance(bp, str):
                # Parse RON: (name: "X", path: "Y")
                import re

                m = re.search(r'path:\s*"([^"]+)"', bp)
                n = re.search(r'name:\s*"([^"]+)"', bp)
                if m:
                    name = n.group(1) if n else m.group(1)
                    assets.append((name, m.group(1)))
        lines = ["(", "  assets:", "   ["]
        for name, path in assets:
            lines.append(f'    ("{name}", File ( path: "{path}" )),')
        lines.append("   ]")
        lines.append(")")
        meta_path.write_text("\n".join(lines) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        """Import a GLB and reconstruct the Scene wrapper."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(Path(path)))
        return cls(bpy.context.scene)

    # -- Filtering (for snap) --

    def ignore(
        self,
        tags: set[str] | list[str] | None = None,
        annotations: set[str] | list[str] | None = None,
        selection: Selection | None = None,
        where: str | None = None,
    ) -> FilteredScene:
        """
        Return a filtered scene view that excludes matching geometry from raycasts.

        Objects matching *any* of the given tags, annotations, or included
        in *selection* are excluded.
        """
        tag_set = set(tags) if tags else set()
        ann_set = set(annotations) if annotations else set()
        excluded: list[bpy.types.Object] = []
        seen: set[int] = set()

        for obj in self.objects():
            obj_id = id(obj)
            if obj_id in seen:
                continue
            if (tag_set and self.tags(obj) & tag_set) or (
                ann_set and self.annotations(obj) & ann_set
            ):
                excluded.append(obj)
                seen.add(obj_id)

        if selection is not None:
            for obj in selection:
                obj_id = id(obj)
                if obj_id not in seen:
                    excluded.append(obj)
                    seen.add(obj_id)

        if where is not None:
            for obj in self.select(where):
                obj_id = id(obj)
                if obj_id not in seen:
                    excluded.append(obj)
                    seen.add(obj_id)

        return FilteredScene(self, excluded)

    def snap_targets(
        self,
        where: str | None = None,
        *,
        tags: set[str] | list[str] | None = None,
        annotations: set[str] | list[str] | None = None,
        selection: Selection | None = None,
    ) -> FilteredScene:
        """
        Return a filtered view that raycasts only against matching objects.

        Objects matching *any* criterion are **included**; everything else is
        excluded from raycasts.  Accepts the same filter params as
        :meth:`ignore` but with inverted semantics.

        Examples::

            scene.snap_targets("tags.has('terrain')")     # DSL query
            scene.snap_targets(tags={"terrain"})           # keyword shorthand
            scene.snap_targets(tags={"terrain", "rock"})   # union
        """
        included_ids: set[int] = set()

        if where is not None:
            included_ids.update(id(o) for o in self.select(where))

        if tags is not None:
            tag_set = set(tags)
            for o in self.objects():
                if self.tags(o) & tag_set:
                    included_ids.add(id(o))

        if annotations is not None:
            ann_set = set(annotations)
            for o in self.objects():
                if self.annotations(o) & ann_set:
                    included_ids.add(id(o))

        if selection is not None:
            included_ids.update(id(o) for o in selection)

        if (
            not included_ids
            and where is None
            and tags is None
            and annotations is None
            and selection is None
        ):
            return FilteredScene(self, [])

        excluded = [o for o in self.objects() if id(o) not in included_ids]
        return FilteredScene(self, excluded)

    # -- Queries --

    def stats(self) -> dict[str, object]:
        """Counts of nodes, meshes, materials, instances, vertices, triangles, and per-prefab."""
        all_objs = list(self._col.all_objects)
        mesh_users: dict[str, int] = {}
        total_verts = 0
        total_tris = 0
        per_prefab: dict[str, int] = {}
        for obj in all_objs:
            if obj.type == "MESH" and obj.data:
                mesh_users[obj.data.name] = mesh_users.get(obj.data.name, 0) + 1
                mesh = obj.data
                total_verts += len(mesh.vertices)
                total_tris += sum(len(p.vertices) - 2 for p in mesh.polygons)
            asset_id = obj.get("_asset_id")
            if asset_id:
                per_prefab[asset_id] = per_prefab.get(asset_id, 0) + 1
        return {
            "nodes": len(all_objs),
            "meshes": sum(1 for o in all_objs if o.type == "MESH"),
            "materials": len(bpy.data.materials),
            "instances": sum(v - 1 for v in mesh_users.values() if v > 1),
            "total_vertices": total_verts,
            "total_triangles": total_tris,
            "per_prefab": per_prefab,
        }

    def gc(self) -> dict[str, int]:
        """Remove orphaned datablocks (0 users). Returns removal counts."""
        counts: dict[str, int] = {"meshes": 0, "materials": 0, "images": 0}
        for mesh in list(bpy.data.meshes):
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
                counts["meshes"] += 1
        for mat in list(bpy.data.materials):
            if mat.users == 0:
                bpy.data.materials.remove(mat)
                counts["materials"] += 1
        for img in list(bpy.data.images):
            if img.users == 0:
                bpy.data.images.remove(img)
                counts["images"] += 1
        return counts

    @staticmethod
    def _check_uv_quality(
        obj: bpy.types.Object,
        stretch_threshold: float,
        max_sample_faces: int = 100,
    ) -> list[ValidationIssue]:
        """
        Check UV quality for a single mesh object.

        Returns issues for:
        - ``"no_uv_layer"`` — textured material but no UV layer.
        - ``"uv_stretch"``  — worst-face stretch ratio exceeds *stretch_threshold*.
        """
        mesh = obj.data
        if not isinstance(mesh, bpy.types.Mesh):
            return []

        # Determine whether any assigned material uses image textures.
        # Check mesh.materials directly — works even before object is linked
        # to a scene (material_slots require scene linkage).
        has_textures = False
        for mat in mesh.materials:
            if mat and mat.node_tree:
                if any(n.type == "TEX_IMAGE" for n in mat.node_tree.nodes):
                    has_textures = True
                    break

        if not has_textures:
            return []

        # --- no UV layer --------------------------------------------------
        if not mesh.uv_layers:
            return [
                ValidationIssue(
                    object=obj.name,
                    issue="no_uv_layer",
                    face_count=len(mesh.polygons),
                )
            ]

        # --- stretch detection --------------------------------------------
        uv_layer = mesh.uv_layers.active
        if uv_layer is None:
            return []

        uv_data = uv_layer.data
        world_mat = obj.matrix_world

        worst_stretch = 1.0
        sampled = 0

        # Iterate faces; cap at *max_sample_faces* to keep cost bounded.
        for poly in mesh.polygons:
            if sampled >= max_sample_faces:
                break

            loop_count = poly.loop_total
            if loop_count < 3:
                continue

            # Pick two edges of the face (edge 0 and edge 1).
            li0 = poly.loop_start
            li1 = li0 + 1
            li2 = li0 + 2

            # World-space edge vectors
            v0 = world_mat @ mesh.vertices[mesh.loops[li0].vertex_index].co
            v1 = world_mat @ mesh.vertices[mesh.loops[li1].vertex_index].co
            v2 = world_mat @ mesh.vertices[mesh.loops[li2].vertex_index].co

            edge_a_world = (v1 - v0).length
            edge_b_world = (v2 - v1).length

            # Skip degenerate faces.
            if edge_a_world < 1e-9 or edge_b_world < 1e-9:
                continue

            # UV-space edge lengths
            uv0 = uv_data[li0].uv
            uv1 = uv_data[li1].uv
            uv2 = uv_data[li2].uv

            edge_a_uv = math.hypot(uv1[0] - uv0[0], uv1[1] - uv0[1])
            edge_b_uv = math.hypot(uv2[0] - uv1[0], uv2[1] - uv1[1])

            if edge_a_uv < 1e-9 or edge_b_uv < 1e-9:
                continue

            density_a = edge_a_world / edge_a_uv
            density_b = edge_b_world / edge_b_uv

            lo = min(density_a, density_b)
            hi = max(density_a, density_b)
            face_stretch = hi / lo if lo > 1e-9 else hi

            worst_stretch = max(worst_stretch, face_stretch)

            sampled += 1

        if worst_stretch > stretch_threshold:
            return [
                ValidationIssue(
                    object=obj.name,
                    issue="uv_stretch",
                    max_stretch=worst_stretch,
                    face_count=sampled,
                )
            ]

        return []

    def validate(
        self,
        *,
        min_ratio: float = 0.005,
        max_ratio: float = 0.5,
        stretch_threshold: float = 4.0,
    ) -> list[ValidationIssue]:
        """
        Full scene health check. Returns list of issue dicts.

        Checks: objects too small/large relative to scene, zero dimensions,
        missing UV layers on textured meshes, and UV stretch quality.
        """
        scene_bb = self.bbox()
        scene_diag = (scene_bb[1] - scene_bb[0]).length() if scene_bb else 0.0
        issues: list[ValidationIssue] = []

        for obj in self.objects():
            dims = self._object_dimensions(obj)
            obj_diag = dims.length()

            if obj_diag < 1e-9:
                issues.append(
                    ValidationIssue(
                        object=obj.name,
                        issue="zero_dimensions",
                        dims=(dims.x, dims.y, dims.z),
                    )
                )
                continue

            if scene_diag > 1e-9:
                ratio = obj_diag / scene_diag
                if ratio < min_ratio:
                    issues.append(
                        ValidationIssue(
                            object=obj.name,
                            issue="too_small",
                            diag=obj_diag,
                            scene_diag=scene_diag,
                            ratio=ratio,
                        )
                    )
                elif ratio > max_ratio:
                    issues.append(
                        ValidationIssue(
                            object=obj.name,
                            issue="too_large",
                            diag=obj_diag,
                            scene_diag=scene_diag,
                            ratio=ratio,
                        )
                    )

            # UV quality checks (only for mesh objects with textured materials).
            if obj.type == "MESH":
                issues.extend(self._check_uv_quality(obj, stretch_threshold))

        if issues:
            logger.warning("[validate] %d issue(s) found", len(issues))
            for iss in issues:
                logger.warning(
                    "  - %s: %s", iss.get("object", "?"), iss.get("issue", "?")
                )
        else:
            logger.debug("[validate] No issues found.")

        return issues

    def bbox(
        self,
        tags: set[str] | None = None,
        exclude: list[bpy.types.Object] | None = None,
    ) -> tuple[Vec3, Vec3] | None:
        """
        Axis-aligned bounding box, optionally filtered by *tags* and/or *exclude*.

        Returns ``(min, max)`` Vec3 pair, or ``None`` if no objects match.
        """
        objs: list[bpy.types.Object] = list(self._col.all_objects)
        if tags is not None:
            objs = [o for o in objs if KEY_UID in o and self.tags(o) & tags]
        if exclude:
            exclude_ids = {id(o) for o in exclude}
            objs = [o for o in objs if id(o) not in exclude_ids]
        if not objs:
            return None

        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for obj in objs:
            if obj.type == "MESH":
                for corner in obj.bound_box:
                    w = obj.matrix_world @ mathutils.Vector(corner)
                    xs.append(w.x)
                    ys.append(w.y)
                    zs.append(w.z)
            else:
                loc = obj.matrix_world.translation
                xs.append(loc.x)
                ys.append(loc.y)
                zs.append(loc.z)

        if not xs:
            return None
        return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))

    # -- World-metric convenience methods --

    def world_extent(self) -> Vec3 | None:
        """Axis-aligned size of the scene ``(dx, dy, dz)`` in metres, or *None* if empty."""
        bb = self.bbox()
        if bb is None:
            return None
        lo, hi = bb
        return Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)

    def world_diagonal(self) -> float:
        """Euclidean diagonal of the scene bounding box in metres (0 if empty)."""
        ext = self.world_extent()
        if ext is None:
            return 0.0
        return ext.length()

    def world_center(self) -> Vec3 | None:
        """Centre of the scene bounding box, or *None* if empty."""
        bb = self.bbox()
        if bb is None:
            return None
        lo, hi = bb
        return Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)

    def world_area(self) -> float:
        """Ground-plane area (extent.x * extent.y) in square metres (0 if empty)."""
        ext = self.world_extent()
        if ext is None:
            return 0.0
        return ext.x * ext.y

    def normalize(self, target_size: float = 1.0) -> None:
        """Scale and center scene so its largest bbox dimension equals *target_size*."""
        # Force depsgraph update so matrix_world is current
        dg = bpy.context.evaluated_depsgraph_get()
        dg.update()

        bb = self.bbox()
        if bb is None:
            return
        lo, hi = bb
        dims = (hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        max_dim = max(dims)
        if max_dim < 1e-9:
            return
        factor = target_size / max_dim
        center_x = (lo.x + hi.x) / 2
        center_y = (lo.y + hi.y) / 2
        center_z = (lo.z + hi.z) / 2
        for obj in self._col.all_objects:
            loc = obj.location
            obj.location = (
                (loc.x - center_x) * factor,
                (loc.y - center_y) * factor,
                (loc.z - center_z) * factor,
            )
            obj.scale = (
                obj.scale.x * factor,
                obj.scale.y * factor,
                obj.scale.z * factor,
            )
        # Update depsgraph after mutations
        dg.update()

    # -- Selection & Query --

    def find(self, name: str) -> Entity:
        """
        Find a tracked object by its display name.

        Returns an :class:`Entity` wrapping the object, or raises
        ``KeyError`` if no object with that name exists in the scene.

        Example::

            tent = scene.find("tent_0")
            print(tent.world_bounds())
        """
        for obj in self._col.all_objects:
            if KEY_UID in obj and obj.name == name:
                return Entity(obj)
        msg = f"No tracked object named {name!r} in scene"
        raise KeyError(msg)

    def select(self, expr: str) -> Selection:
        """
        Select tracked objects matching a query DSL expression.

        Examples::

            scene.select("tags.has('veg')")
            scene.select("props.biome == 'pine' & !tags.has('road')")
        """
        pred = parse_query(expr)
        return Selection([obj for obj in self.objects() if pred(obj)])

    def delete(self, sel: Selection) -> int:
        """Remove all objects in *sel* from the scene. Returns removed count."""
        count = 0
        for obj in sel:
            bpy.data.objects.remove(obj, do_unlink=True)
            count += 1
        return count

    def transform(self, sel: Selection) -> Transform:
        """Return a :class:`Transform` helper for all objects in *sel*."""
        return Transform(list(sel))

    # -- Spatial clustering --

    def _compute_clusters(self, radius: float = 30.0) -> list[dict[str, object]]:
        """
        Single-linkage clustering of scene objects within *radius* metres.

        Returns a list of cluster dicts, each with ``center`` (Vec3-like),
        ``radius`` (float), and ``objects`` (list of names).
        """
        objs = [o for o in self.objects() if o.get("_type") != "anchor"]
        if not objs:
            return []

        # Extract world positions
        positions: list[Vec3] = []
        for obj in objs:
            loc = obj.matrix_world.translation
            positions.append(Vec3(float(loc.x), float(loc.y), float(loc.z)))

        # Union-find
        parent = list(range(len(objs)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                if positions[i].distance(positions[j]) <= radius:
                    union(i, j)

        # Group by root
        groups: dict[int, list[int]] = {}
        for i in range(len(objs)):
            root = find(i)
            groups.setdefault(root, []).append(i)

        clusters: list[dict[str, object]] = []
        for members in groups.values():
            pts = [positions[i] for i in members]
            cx = sum(p.x for p in pts) / len(pts)
            cy = sum(p.y for p in pts) / len(pts)
            cz = sum(p.z for p in pts) / len(pts)
            center = Vec3(round(cx, 4), round(cy, 4), round(cz, 4))
            max_dist = max(center.distance(p) for p in pts)
            clusters.append({
                "center": {"x": center.x, "y": center.y, "z": center.z},
                "radius": round(max_dist, 4),
                "objects": [objs[i].name for i in members],
            })
        return clusters

    def describe(self, sel: Selection | None = None) -> str:
        """
        Structured scene description with objects, anchors, clusters, and stats.

        Returns a JSON string with full spatial context for agent consumption.
        """
        objs = list(sel) if sel is not None else self.objects()
        entries = []
        for obj in objs:
            loc = obj.matrix_world.translation
            entries.append({
                "uid": self.uid(obj),
                "name": obj.name,
                "tags": sorted(self.tags(obj)),
                "annotations": sorted(self.annotations(obj)),
                "props": self.props(obj),
                "position": {
                    "x": round(float(loc.x), 4),
                    "y": round(float(loc.y), 4),
                    "z": round(float(loc.z), 4),
                },
            })

        # Anchors
        anchor_list = []
        for a in self.anchors():
            pos = a.location()
            anchor_list.append({
                "name": a.name,
                "annotation": a.annotation,
                "position": {
                    "x": round(pos.x, 4),
                    "y": round(pos.y, 4),
                    "z": round(pos.z, 4),
                },
            })

        # Bounding box
        bb = self.bbox()
        bbox_dict = None
        if bb:
            bbox_dict = {
                "min": {
                    "x": round(bb[0].x, 4),
                    "y": round(bb[0].y, 4),
                    "z": round(bb[0].z, 4),
                },
                "max": {
                    "x": round(bb[1].x, 4),
                    "y": round(bb[1].y, 4),
                    "z": round(bb[1].z, 4),
                },
            }

        # Stats
        by_tag: dict[str, int] = {}
        for obj in objs:
            for tag in self.tags(obj):
                by_tag[tag] = by_tag.get(tag, 0) + 1
        stats = {"total_objects": len(objs), "by_tag": by_tag}

        # Spatial clusters
        clusters = self._compute_clusters()

        # World metrics
        world_metrics: dict[str, object] | None = None
        if bb:
            ext = Vec3(bb[1].x - bb[0].x, bb[1].y - bb[0].y, bb[1].z - bb[0].z)
            world_metrics = {
                "extent": {
                    "x": round(ext.x, 4),
                    "y": round(ext.y, 4),
                    "z": round(ext.z, 4),
                },
                "diagonal": round(ext.length(), 4),
                "center": {
                    "x": round((bb[0].x + bb[1].x) / 2, 4),
                    "y": round((bb[0].y + bb[1].y) / 2, 4),
                    "z": round((bb[0].z + bb[1].z) / 2, 4),
                },
                "ground_area": round(ext.x * ext.y, 4),
                "unit": "metres",
            }

        result = {
            "objects": entries,
            "anchors": anchor_list,
            "spatial_clusters": clusters,
            "bounding_box": bbox_dict,
            "world_metrics": world_metrics,
            "stats": stats,
        }
        return json.dumps(result, indent=2)


class _AnchorAccessor:
    """Attribute-style anchor accessor used by ``Scene.a``."""

    __slots__ = ("_scene",)

    def __init__(self, scene: Scene) -> None:
        self._scene = scene

    def __getattr__(self, name: str) -> Anchor:
        anchor = self._scene.anchor(name)
        if anchor is None:
            msg = f"Anchor {name!r} not found"
            raise AttributeError(msg)
        return anchor
