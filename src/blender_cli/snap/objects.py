"""Snap operations — FilteredScene, snap(), snap_object() and private helpers."""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import bpy
import mathutils

from blender_cli.snap.axis import (
    _FAR,
    AXIS_DIR,
    active_snap_axes,
    resolve_direction,
)
from blender_cli.snap.results import (
    SnapObjectResult,
    SnapPolicy,
    SnapResult,
    SnapResults,
    SnapSummary,
)
from blender_cli.types import Vec3

if TYPE_CHECKING:
    from blender_cli.scene.scene import Scene


# -- FilteredScene -----------------------------------------------------


class FilteredScene:
    """A scene view that excludes certain objects from raycasts."""

    __slots__ = ("_excluded", "_scene")

    def __init__(self, scene: Scene, excluded: list[bpy.types.Object]) -> None:
        self._scene = scene
        self._excluded = excluded

    @property
    def bpy_scene(self) -> bpy.types.Scene:
        return self._scene.bpy_scene

    @property
    def scene(self) -> Scene:
        return self._scene

    def bbox(self) -> tuple[Vec3, Vec3] | None:
        """Bounding box of included objects only (excluded objects are omitted)."""
        return self._scene.bbox(exclude=self._excluded)


# -- Internal helpers --------------------------------------------------


def _ray_cast_through(
    bpy_scene: bpy.types.Scene,
    depsgraph: bpy.types.Depsgraph,
    origin: tuple[float, float, float],
    direction: tuple[float, float, float],
    max_hits: int = 64,
    epsilon: float = 0.001,
) -> list[tuple[Vec3, Vec3, bpy.types.Object | None]]:
    """Cast a ray and return all *(position, normal, object)* hits along the path."""
    hits: list[tuple[Vec3, Vec3, bpy.types.Object | None]] = []
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz = direction
    for _ in range(max_hits):
        ok, loc, normal, _idx, obj, _mat = bpy_scene.ray_cast(
            depsgraph, (ox, oy, oz), (dx, dy, dz)
        )
        if not ok:
            break
        hits.append((
            Vec3(float(loc.x), float(loc.y), float(loc.z)),
            Vec3(float(normal.x), float(normal.y), float(normal.z)),
            obj,
        ))
        # Advance past the hit point.
        ox = float(loc.x) + dx * epsilon
        oy = float(loc.y) + dy * epsilon
        oz = float(loc.z) + dz * epsilon
    return hits


def _normal_to_euler(nx: float, ny: float, nz: float) -> tuple[float, float, float]:
    """Convert a surface normal to Euler XYZ rotation *(pitch, roll, 0)*."""
    pitch = math.atan2(-ny, nz)
    roll = math.atan2(nx, nz)
    return (pitch, roll, 0.0)


def _build_world_matrix(obj: bpy.types.Object) -> mathutils.Matrix:
    """
    Build the object's world matrix from its TRS properties.

    This avoids relying on ``obj.matrix_world`` which may be stale for
    objects not yet linked to a scene collection (e.g. during the fluent
    ``box().at().snap().tag()`` chain before ``scene.add()``).
    """
    loc = mathutils.Matrix.Translation((obj.location.x, obj.location.y, obj.location.z))
    rot = obj.rotation_euler.to_matrix().to_4x4()
    scl = mathutils.Matrix.Diagonal((*obj.scale, 1.0))
    return loc @ rot @ scl


def _get_mesh_vertices_world(obj: bpy.types.Object) -> list[Vec3]:
    """
    Extract world-space vertex positions from a mesh object.

    Uses ``evaluated_get(depsgraph)`` to apply modifiers, then transforms
    each vertex by a TRS matrix built from ``obj.location/rotation/scale``
    (not ``matrix_world``, which may be stale for unlinked objects).
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    if mesh is None:
        return []
    mat = _build_world_matrix(obj)
    verts: list[Vec3] = []
    for v in mesh.vertices:
        co = mat @ v.co
        verts.append(Vec3(float(co.x), float(co.y), float(co.z)))
    eval_obj.to_mesh_clear()
    return verts


def _snap_axis_index(axis: str) -> int:
    """Return the component index (0=x, 1=y, 2=z) for a snap axis."""
    d = AXIS_DIR[axis]
    if d[0] != 0:
        return 0
    if d[1] != 0:
        return 1
    return 2


# -- snap function (low-level, point-list) -----------------------------


_AXIS_NAMES = {0: "x", 1: "y", 2: "z"}


def snap(
    points: list[Vec3],
    scene: Scene | FilteredScene,
    axis: str | tuple[float, float, float] = "-Z",
    *,
    _wildcards: list[frozenset[int]] | None = None,
    _resolved_axes: frozenset[int] = frozenset(),
) -> SnapResults:
    """
    Snap *points* onto scene geometry by raycasting along *axis*.

    *axis* may be a string (``'-Z'``, ``'+X'``, …) or an arbitrary
    direction tuple ``(dx, dy, dz)`` for diagonal/custom raycasting.

    When *_wildcards* is provided (from :meth:`PointSet.from_coords`):

    - Wildcard on a **lateral axis** (direction component = 0) → ``ValueError``.
    - Unresolved wildcard on a **snap axis** (direction component ≠ 0) →
      ray origin at ``±_FAR`` + warning.
    - Snap axis in *_resolved_axes* (set by ``.with_x()`` etc.) → the
      point's actual coordinate is used as ray origin.

    Returns a :class:`SnapResults` (behaves like ``list[SnapResult]``)
    with an attached ``.summary`` (:class:`SnapSummary`).
    Points that miss are flagged with ``hit=False`` (never silently dropped).
    """
    from blender_cli.scene.scene import Scene as SceneCls

    # -- Direction resolution ------------------------------------------
    dx, dy, dz = resolve_direction(axis)
    active = active_snap_axes((dx, dy, dz))

    # Resolve scene and excluded objects.
    if isinstance(scene, FilteredScene):
        excluded = scene._excluded
        bpy_scene = scene.bpy_scene
    else:
        excluded = []
        bpy_scene = scene.bpy_scene

    # -- Validate & warn on wildcards ----------------------------------
    if _wildcards is not None:
        for i, wc in enumerate(_wildcards):
            bad = wc - active  # Wildcard on lateral axis → ERROR
            if bad:
                bad_names = ", ".join(sorted(_AXIS_NAMES[a] for a in bad))
                msg = (
                    f"Point {i} has wildcard ('*') on lateral "
                    f"axis {bad_names} (direction component is zero). "
                    f"Only axes with non-zero direction may be wildcard."
                )
                raise ValueError(msg)

        # Unresolved wildcards on snap axes → WARNING
        unresolved_snap: set[int] = set()
        for wc in _wildcards:
            unresolved_snap |= wc - _resolved_axes
        if unresolved_snap & active:
            names = ", ".join(
                sorted(_AXIS_NAMES[a] for a in (unresolved_snap & active))
            )
            hints = ", ".join(
                f".with_{_AXIS_NAMES[a]}(value)"
                for a in sorted(unresolved_snap & active)
            )
            warnings.warn(
                f"Snap axis coordinate(s) {names} are wildcard ('*'): "
                f"defaulting ray origin to ±{_FAR:.0f}. "
                f"Use {hints} to set specific ray start(s).",
                stacklevel=3,
            )

    # -- Build ray origins (per-component) -----------------------------
    direction = (dx, dy, dz)
    ray_points: list[Vec3] = []
    for pt in points:
        ox = (
            pt.x
            if (abs(dx) < 1e-9 or 0 in _resolved_axes)
            else -math.copysign(_FAR, dx)
        )
        oy = (
            pt.y
            if (abs(dy) < 1e-9 or 1 in _resolved_axes)
            else -math.copysign(_FAR, dy)
        )
        oz = (
            pt.z
            if (abs(dz) < 1e-9 or 2 in _resolved_axes)
            else -math.copysign(_FAR, dz)
        )
        ray_points.append(Vec3(ox, oy, oz))

    # Temporarily hide excluded objects so ray_cast ignores them.
    saved: list[tuple[bpy.types.Object, bool]] = []
    for obj in excluded:
        saved.append((obj, obj.hide_viewport))
        obj.hide_viewport = True

    try:
        # view_layer.update() is called once per snap() invocation to
        # flush visibility changes (hidden excluded objects) into the
        # depsgraph so that ray_cast ignores them.  This is the minimum
        # required — one call per batch, not per point.
        view_layer = bpy.context.view_layer
        if view_layer is None:
            msg = "No active view layer"
            raise RuntimeError(msg)
        view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        results: list[SnapResult] = []
        for orig_pt, ray_pt in zip(points, ray_points, strict=False):
            origin_tuple = (ray_pt.x, ray_pt.y, ray_pt.z)
            ray_origin = Vec3(*origin_tuple)

            ok, loc, normal, _idx, obj, _mat = bpy_scene.ray_cast(
                depsgraph, origin_tuple, direction
            )

            if ok:
                hit_pos = Vec3(float(loc.x), float(loc.y), float(loc.z))
                hit_normal = Vec3(float(normal.x), float(normal.y), float(normal.z))
                hit_uid = SceneCls.uid(obj) if obj is not None else None
                hit_distance = ray_origin.distance(hit_pos)
                results.append(
                    SnapResult(
                        point=orig_pt,
                        hit=True,
                        hit_pos=hit_pos,
                        hit_normal=hit_normal,
                        hit_uid=hit_uid,
                        hit_distance=hit_distance,
                        snap_axis=axis,
                        ray_origin=ray_origin,
                    )
                )
            else:
                results.append(
                    SnapResult(
                        point=orig_pt,
                        hit=False,
                        hit_pos=orig_pt,
                        hit_normal=None,
                        hit_uid=None,
                        hit_distance=-1.0,
                        snap_axis=axis,
                        ray_origin=ray_origin,
                    )
                )

        summary = SnapSummary.from_results(results)
        return SnapResults(results, summary)

    finally:
        # Restore visibility of excluded objects.
        for obj, was_visible in saved:
            obj.hide_viewport = was_visible


# -- snap_object (mesh-accurate, per-vertex) ---------------------------


def snap_object(
    obj: bpy.types.Object,
    position: Vec3,
    scene: Scene | FilteredScene,
    policy: SnapPolicy = SnapPolicy.FIRST,
    axis: str = "-Z",
) -> SnapObjectResult:
    """
    Snap a mesh object to scene geometry using its actual vertices.

    Casts rays from every vertex of *obj* (at the proposed *position*)
    along *axis* and resolves the final placement according to *policy*.
    The object itself is excluded from raycasting.

    Args:
        obj: Mesh object to snap.
        position: Proposed world-space position for the object origin.
        scene: Scene (or FilteredScene) to raycast against.
        policy: How to resolve the final position from vertex hits.
        axis: Snap axis (default ``"-Z"`` for floor snap).

    Returns:
        :class:`SnapObjectResult` with the resolved position, optional
        rotation, and hit statistics.

    """
    if axis not in AXIS_DIR:
        msg = f"Unknown snap axis {axis!r}, expected one of {sorted(AXIS_DIR)}"
        raise ValueError(msg)

    from blender_cli.scene.entity import unwrap_entity

    raw_obj = unwrap_entity(obj)
    if not isinstance(raw_obj, bpy.types.Object):
        msg = f"snap_object() expected bpy.types.Object, got {type(raw_obj).__name__}"
        raise TypeError(msg)
    obj = raw_obj

    # Resolve scene and excluded objects.
    if isinstance(scene, FilteredScene):
        excluded = list(scene._excluded)
        bpy_scn = scene.bpy_scene
    else:
        excluded = []
        bpy_scn = scene.bpy_scene

    # Hide excluded objects + the object itself before updating.
    to_hide = list(excluded)
    if obj not in to_hide:
        to_hide.append(obj)
    saved_vis: list[tuple[bpy.types.Object, bool]] = []
    for o in to_hide:
        saved_vis.append((o, o.hide_viewport))
        o.hide_viewport = True

    try:
        # Single view_layer.update() per snap_object() call.  This
        # flushes the hide_viewport changes into the depsgraph AND
        # ensures evaluated meshes are current for vertex extraction.
        # Hiding an object only affects ray_cast visibility, not
        # evaluated_get(), so vertex extraction still works.
        view_layer = bpy.context.view_layer
        if view_layer is None:
            msg = "No active view layer"
            raise RuntimeError(msg)
        view_layer.update()
        depsgraph = bpy.context.evaluated_depsgraph_get()

        # Get vertex world positions at current location, then offset
        # to the proposed position.  This avoids moving the object or
        # re-evaluating the depsgraph.
        current_verts = _get_mesh_vertices_world(obj)
        if not current_verts:
            return SnapObjectResult(
                position=position,
                rotation=None,
                policy=policy,
                vertex_hits=0,
                vertex_total=0,
                z_min=position.z,
                z_max=position.z,
                z_mean=position.z,
                z_spread=0.0,
                drop_distance=0.0,
            )

        # Offset vertices to proposed position.
        off_x = position.x - float(obj.location.x)
        off_y = position.y - float(obj.location.y)
        off_z = position.z - float(obj.location.z)
        vertices = [Vec3(v.x + off_x, v.y + off_y, v.z + off_z) for v in current_verts]

        dx, dy, dz = AXIS_DIR[axis]
        direction = (dx, dy, dz)

        # Place ray origins far away along the snap axis so rays always
        # start outside the scene.  Simple and correct for all cases.
        ray_verts = [
            Vec3(
                v.x if dx == 0 else (-dx * _FAR),
                v.y if dy == 0 else (-dy * _FAR),
                v.z if dz == 0 else (-dz * _FAR),
            )
            for v in vertices
        ]

        # Cast rays from each vertex.
        # Collect: (original_vertex, hit_pos, hit_normal)
        hit_data: list[tuple[Vec3, Vec3, Vec3]] = []

        if policy == SnapPolicy.LAST:
            # Through-cast: find deepest hit per vertex.
            for orig_v, ray_v in zip(vertices, ray_verts, strict=False):
                origin = (ray_v.x, ray_v.y, ray_v.z)
                hits = _ray_cast_through(bpy_scn, depsgraph, origin, direction)
                if hits:
                    deepest_pos, deepest_normal, _ = hits[-1]
                    hit_data.append((orig_v, deepest_pos, deepest_normal))
        else:
            # Single ray per vertex.
            for orig_v, ray_v in zip(vertices, ray_verts, strict=False):
                origin = (ray_v.x, ray_v.y, ray_v.z)
                ok, loc, normal, _idx, _obj, _mat = bpy_scn.ray_cast(
                    depsgraph, origin, direction
                )
                if ok:
                    hit_pos = Vec3(float(loc.x), float(loc.y), float(loc.z))
                    hit_normal = Vec3(float(normal.x), float(normal.y), float(normal.z))
                    hit_data.append((orig_v, hit_pos, hit_normal))

        vertex_total = len(vertices)
        vertex_hits = len(hit_data)

        if vertex_hits == 0:
            return SnapObjectResult(
                position=position,
                rotation=None,
                policy=policy,
                vertex_hits=0,
                vertex_total=vertex_total,
                z_min=position.z,
                z_max=position.z,
                z_mean=position.z,
                z_spread=0.0,
                drop_distance=0.0,
            )

        # Hit Z statistics (always Z component, regardless of snap axis).
        hit_zs = [h.z for _, h, _ in hit_data]
        z_min_v = min(hit_zs)
        z_max_v = max(hit_zs)
        z_mean_v = sum(hit_zs) / len(hit_zs)
        z_spread_v = z_max_v - z_min_v

        axis_idx = _snap_axis_index(axis)

        # -- Compute per-vertex travel distances -----------------------
        # Travel = how far each vertex must move along the ray to reach
        # its hit point.  Used by all policies.
        travels: list[float] = []
        for orig_v, hit_pos, _ in hit_data:
            travel = (
                dx * (hit_pos.x - orig_v.x)
                + dy * (hit_pos.y - orig_v.y)
                + dz * (hit_pos.z - orig_v.z)
            )
            travels.append(travel)

        # -- Resolve final position by policy --------------------------
        rotation: tuple[float, float, float] | None = None

        if policy in {SnapPolicy.FIRST, SnapPolicy.LAST}:
            # Min travel: the lowest-reaching vertex just touches.
            # FIRST uses first hit (standard ray_cast), LAST uses
            # deepest hit (through-cast).
            chosen_travel = min(travels)

        elif policy == SnapPolicy.ORIENT:
            # ORIENT: compute rotation from mean surface normal, then
            # re-compute vertex positions with that rotation applied,
            # and THEN find min travel so the rotated mesh sits on terrain.
            normals = [n for _, _, n in hit_data]
            mean_nx = sum(n.x for n in normals) / len(normals)
            mean_ny = sum(n.y for n in normals) / len(normals)
            mean_nz = sum(n.z for n in normals) / len(normals)
            rotation = _normal_to_euler(mean_nx, mean_ny, mean_nz)

            # Re-compute vertices as if the object had this rotation at
            # the proposed position — then raycast from those to find
            # the correct travel that accounts for rotation.
            rot_euler = mathutils.Euler(rotation, "XYZ")
            rot_mat = rot_euler.to_matrix().to_4x4()
            scl = mathutils.Matrix.Diagonal((*obj.scale, 1.0))
            local_mat = rot_mat @ scl

            depsgraph2 = bpy.context.evaluated_depsgraph_get()
            eval_obj2 = obj.evaluated_get(depsgraph2)
            mesh2 = eval_obj2.to_mesh()
            if mesh2 is not None:
                rotated_verts: list[Vec3] = []
                for v in mesh2.vertices:
                    co = local_mat @ v.co
                    rotated_verts.append(
                        Vec3(
                            position.x + float(co.x),
                            position.y + float(co.y),
                            position.z + float(co.z),
                        )
                    )
                eval_obj2.to_mesh_clear()

                # Re-cast rays from rotated vertices to get correct travel.
                rot_travels: list[float] = []
                for rv in rotated_verts:
                    origin = (
                        rv.x if dx == 0 else (-dx * _FAR),
                        rv.y if dy == 0 else (-dy * _FAR),
                        rv.z if dz == 0 else (-dz * _FAR),
                    )
                    ok2, loc2, _, _, _, _ = bpy_scn.ray_cast(
                        depsgraph2, origin, direction
                    )
                    if ok2:
                        hit2 = Vec3(float(loc2.x), float(loc2.y), float(loc2.z))
                        t = (
                            dx * (hit2.x - rv.x)
                            + dy * (hit2.y - rv.y)
                            + dz * (hit2.z - rv.z)
                        )
                        rot_travels.append(t)

                chosen_travel = min(rot_travels) if rot_travels else min(travels)
            else:
                chosen_travel = min(travels)

        elif policy == SnapPolicy.HIGHEST:
            hit_axis_vals = [h.component(axis_idx) for _, h, _ in hit_data]
            target = max(hit_axis_vals)
            # direction[axis_idx] is ±1 for unit axis-aligned directions.
            dir_component = (dx, dy, dz)[axis_idx]
            chosen_travel = (target - position.component(axis_idx)) / dir_component

        elif policy == SnapPolicy.LOWEST:
            hit_axis_vals = [h.component(axis_idx) for _, h, _ in hit_data]
            target = min(hit_axis_vals)
            dir_component = (dx, dy, dz)[axis_idx]
            chosen_travel = (target - position.component(axis_idx)) / dir_component

        else:
            # AVERAGE: object origin at mean hit along snap axis.
            hit_axis_vals = [h.component(axis_idx) for _, h, _ in hit_data]
            target = sum(hit_axis_vals) / len(hit_axis_vals)
            dir_component = (dx, dy, dz)[axis_idx]
            chosen_travel = (target - position.component(axis_idx)) / dir_component

        final_pos = Vec3(
            position.x + dx * chosen_travel,
            position.y + dy * chosen_travel,
            position.z + dz * chosen_travel,
        )

        # -- Penetration detection -------------------------------------
        # Check if any vertex ends up past its terrain hit point in the
        # snap direction.  Penetration > 0 means the vertex is below
        # (inside) the terrain surface.
        #
        # For ORIENT, the mesh is rotated — use rotated vertices to
        # avoid false positives from un-rotated positions.
        # For FIRST/LAST, penetration is impossible by construction
        # (min travel), so this is mainly for AVERAGE/HIGHEST/LOWEST.
        pen_depth = 0.0
        pen_count = 0
        if policy != SnapPolicy.ORIENT:
            for orig_v, hit_pos, _ in hit_data:
                new_vx = orig_v.x + dx * chosen_travel
                new_vy = orig_v.y + dy * chosen_travel
                new_vz = orig_v.z + dz * chosen_travel
                pen = (
                    dx * (new_vx - hit_pos.x)
                    + dy * (new_vy - hit_pos.y)
                    + dz * (new_vz - hit_pos.z)
                )
                if pen > 0.001:
                    pen_depth = max(pen_depth, pen)
                    pen_count += 1

        PENETRATION_WARN_THRESHOLD = 0.1  # metres
        if pen_depth > PENETRATION_WARN_THRESHOLD:
            warnings.warn(
                f"snap_object: {pen_count}/{vertex_hits} vertices penetrate terrain "
                f"by up to {pen_depth:.2f}m (policy={policy.value}). "
                f"This may cause visual clipping. "
                f"Suggestions: "
                f"(1) use SnapPolicy.FIRST for ground-resting objects, "
                f"(2) add .translate(0, 0, {pen_depth:.1f}) after snap to lift the object, "
                f"(3) check that the mesh origin is at the base of the model.",
                stacklevel=2,
            )

        return SnapObjectResult(
            position=final_pos,
            rotation=rotation,
            policy=policy,
            vertex_hits=vertex_hits,
            vertex_total=vertex_total,
            z_min=z_min_v,
            z_max=z_max_v,
            z_mean=z_mean_v,
            z_spread=z_spread_v,
            drop_distance=abs(chosen_travel),
            penetration_depth=pen_depth,
            penetrating_vertices=pen_count,
        )

    finally:
        # Restore visibility.
        for o, vis in saved_vis:
            o.hide_viewport = vis
