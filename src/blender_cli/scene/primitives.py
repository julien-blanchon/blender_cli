"""
Primitive shape constructors — box, cylinder, sphere, plane, cone, torus.

Create basic mesh objects via ``bpy.data`` (no operator / view-layer dependency).
Each function returns a ``bpy.types.Object`` wrapped in an :class:`Entity`,
ready to be added to a :class:`Scene`.

All primitives generate UV coordinates using the same convention as
:meth:`Heightfield.to_mesh`: ``UV = world_position / tile_scale``.
The default *tile_scale* is **10 m** (one full texture repeat every 10 metres),
matching the heightfield default.  The material's ``tile`` parameter (which adds
a Mapping node that *multiplies* UVs) stacks on top, so the effective repeat
distance is ``tile_scale / tile``.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING

import bpy

from blender_cli.scene.entity import Entity

if TYPE_CHECKING:
    from blender_cli.assets.material import Material

# Default metres per UV unit — matches Heightfield.to_mesh() default.
_DEFAULT_TILE_M: float = 10.0


def _apply_material(obj: bpy.types.Object, material: Material) -> None:
    mat = material.get_or_create()
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        msg = f"Expected Mesh data on object {obj.name!r}, got {type(mesh).__name__}"
        raise TypeError(msg)
    mesh.materials.append(mat)

    # Warn when a textured material is applied to a mesh without UV data.
    has_textures = any(
        n.type == "TEX_IMAGE" for n in (mat.node_tree.nodes if mat.node_tree else ())
    )
    if has_textures and not mesh.uv_layers:
        warnings.warn(
            f"Mesh '{obj.name}' has no UV layer — textures on material "
            f"'{mat.name}' will appear stretched or flat. "
            f"Add UV coordinates before assigning a textured material.",
            UserWarning,
            stacklevel=2,
        )


# -- UV helpers ---------------------------------------------------------------

# Per-face UV projection axes for box(): (u_axis, v_axis) into the XYZ tuple.
_BOX_FACE_UV_AXES: list[tuple[int, int]] = [
    (0, 1),  # bottom (Z-): X, Y
    (0, 1),  # top    (Z+): X, Y
    (0, 2),  # front  (Y-): X, Z
    (0, 2),  # back   (Y+): X, Z
    (1, 2),  # left   (X-): Y, Z
    (1, 2),  # right  (X+): Y, Z
]


# -- Primitives ---------------------------------------------------------------


def box(
    name: str,
    size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a box mesh with per-face planar UVs.

    *size*: (width, depth, height) dimensions.

    *tile_scale*: metres per UV unit (default 10).  One full texture repeat
    spans this many metres.  The material's ``tile`` multiplier stacks on top.
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
    verts = [
        (-sx, -sy, -sz),
        (sx, -sy, -sz),
        (sx, sy, -sz),
        (-sx, sy, -sz),
        (-sx, -sy, sz),
        (sx, -sy, sz),
        (sx, sy, sz),
        (-sx, sy, sz),
    ]
    faces = [
        (0, 1, 2, 3),  # bottom
        (4, 7, 6, 5),  # top
        (0, 4, 5, 1),  # front
        (2, 6, 7, 3),  # back
        (0, 3, 7, 4),  # left
        (1, 5, 6, 2),  # right
    ]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # Per-face planar UVs: project onto the face's two tangent axes,
    # scaled by tile_uv so 1 UV unit = tile_m world metres.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for face_idx, poly in enumerate(mesh.polygons):
        u_ax, v_ax = _BOX_FACE_UV_AXES[face_idx]
        for loop_idx in poly.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = (
                verts[vi][u_ax] * tile_uv,
                verts[vi][v_ax] * tile_uv,
            )

    obj = bpy.data.objects.new(name, mesh)

    if material is not None:
        _apply_material(obj, material)

    return Entity(obj)


def plane(
    name: str,
    size: tuple[float, float] = (1.0, 1.0),
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a plane mesh with planar XY UVs.

    *size*: (width, depth) dimensions.

    *tile_scale*: metres per UV unit (default 10).
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    sx, sy = size[0] / 2, size[1] / 2
    verts = [
        (-sx, -sy, 0.0),
        (sx, -sy, 0.0),
        (sx, sy, 0.0),
        (-sx, sy, 0.0),
    ]
    faces = [(0, 1, 2, 3)]

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # Planar XY projection, scaled by tile_uv.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            uv_layer.data[loop_idx].uv = (
                verts[vi][0] * tile_uv,
                verts[vi][1] * tile_uv,
            )

    obj = bpy.data.objects.new(name, mesh)

    if material is not None:
        _apply_material(obj, material)

    return Entity(obj)


def cylinder(
    name: str,
    radius: float = 0.5,
    height: float = 1.0,
    segments: int = 32,
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a cylinder mesh with UVs.

    *tile_scale*: metres per UV unit (default 10).
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    half_h = height / 2
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    # Bottom ring (indices 0 .. segments-1)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), -half_h))

    # Top ring (indices segments .. 2*segments-1)
    for i in range(segments):
        angle = 2 * math.pi * i / segments
        verts.append((radius * math.cos(angle), radius * math.sin(angle), half_h))

    # Center bottom (index 2*segments)
    verts.append((0.0, 0.0, -half_h))
    # Center top (index 2*segments + 1)
    verts.append((0.0, 0.0, half_h))

    bot_center = 2 * segments
    top_center = 2 * segments + 1

    # Side quads
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = i + segments, j + segments
        faces.append((b0, b1, t1, t0))

    # Bottom cap (triangles, winding inward)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((bot_center, j, i))

    # Top cap (triangles)
    for i in range(segments):
        j = (i + 1) % segments
        faces.append((top_center, i + segments, j + segments))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # UVs ---
    # Sides: U = arc length, V = height, both divided by tile_m.
    # Caps : planar XY projection, shifted and divided by tile_m.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    circumference = 2 * math.pi * radius
    n_side = segments
    n_bot = segments

    for face_idx, poly in enumerate(mesh.polygons):
        if face_idx < n_side:
            # Side quad — face i spans arc [i, i+1].
            u0 = circumference * face_idx / segments * tile_uv
            u1 = circumference * (face_idx + 1) / segments * tile_uv
            v0 = 0.0
            v1 = height * tile_uv
            # Face vertex order: (b0, b1, t1, t0)
            loop_uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
            for k, loop_idx in enumerate(poly.loop_indices):
                uv_layer.data[loop_idx].uv = loop_uvs[k]
        elif face_idx < n_side + n_bot:
            # Bottom cap triangle — planar XY, shifted by +radius.
            for loop_idx in poly.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                x, y, _ = verts[vi]
                uv_layer.data[loop_idx].uv = (
                    (x + radius) * tile_uv,
                    (y + radius) * tile_uv,
                )
        else:
            # Top cap triangle — same planar XY projection.
            for loop_idx in poly.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                x, y, _ = verts[vi]
                uv_layer.data[loop_idx].uv = (
                    (x + radius) * tile_uv,
                    (y + radius) * tile_uv,
                )

    obj = bpy.data.objects.new(name, mesh)

    if material is not None:
        _apply_material(obj, material)

    return Entity(obj)


def sphere(
    name: str,
    radius: float = 0.5,
    segments: int = 32,
    rings: int = 16,
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a UV sphere mesh with equirectangular UVs.

    *tile_scale*: metres per UV unit (default 10).
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    # Bottom pole (index 0)
    verts.append((0.0, 0.0, -radius))

    # Latitude rings (index 1 .. rings-1 * segments)
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        z = -radius * math.cos(phi)
        r = radius * math.sin(phi)
        for seg in range(segments):
            theta = 2 * math.pi * seg / segments
            verts.append((r * math.cos(theta), r * math.sin(theta), z))

    # Top pole
    top_idx = len(verts)
    verts.append((0.0, 0.0, radius))

    # Bottom triangle fan (pole → first ring)
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        faces.append((0, 1 + next_seg, 1 + seg))

    # Middle quads between rings
    for ring in range(rings - 2):
        base = 1 + ring * segments
        for seg in range(segments):
            next_seg = (seg + 1) % segments
            v0 = base + seg
            v1 = base + next_seg
            v2 = base + segments + next_seg
            v3 = base + segments + seg
            faces.append((v0, v1, v2, v3))

    # Top triangle fan (last ring → pole)
    last_base = 1 + (rings - 2) * segments
    for seg in range(segments):
        next_seg = (seg + 1) % segments
        faces.append((last_base + seg, last_base + next_seg, top_idx))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # Equirectangular UVs — U = longitude arc, V = latitude arc,
    # both divided by tile_m.
    uv_layer = mesh.uv_layers.new(name="UVMap")
    dtheta = 2 * math.pi / segments
    dphi = math.pi / rings

    face_cursor = 0

    # Bottom fan: face vertices (pole, ring1_next, ring1_seg)
    for seg in range(segments):
        theta0 = seg * dtheta * radius * tile_uv
        theta1 = (seg + 1) * dtheta * radius * tile_uv
        theta_mid = (theta0 + theta1) / 2
        phi1 = dphi * radius * tile_uv
        loop_uvs = [
            (theta_mid, 0.0),  # pole
            (theta1, phi1),  # ring-1 next
            (theta0, phi1),  # ring-1 current
        ]
        poly = mesh.polygons[face_cursor]
        for k, loop_idx in enumerate(poly.loop_indices):
            uv_layer.data[loop_idx].uv = loop_uvs[k]
        face_cursor += 1

    # Middle quads: face vertices (v0, v1, v2, v3)
    for ring in range(rings - 2):
        phi_top = (ring + 1) * dphi * radius * tile_uv
        phi_bot = (ring + 2) * dphi * radius * tile_uv
        for seg in range(segments):
            theta0 = seg * dtheta * radius * tile_uv
            theta1 = (seg + 1) * dtheta * radius * tile_uv
            loop_uvs = [
                (theta0, phi_top),
                (theta1, phi_top),
                (theta1, phi_bot),
                (theta0, phi_bot),
            ]
            poly = mesh.polygons[face_cursor]
            for k, loop_idx in enumerate(poly.loop_indices):
                uv_layer.data[loop_idx].uv = loop_uvs[k]
            face_cursor += 1

    # Top fan: face vertices (last_ring_seg, last_ring_next, pole)
    phi_last = (rings - 1) * dphi * radius * tile_uv
    phi_pole = math.pi * radius * tile_uv
    for seg in range(segments):
        theta0 = seg * dtheta * radius * tile_uv
        theta1 = (seg + 1) * dtheta * radius * tile_uv
        theta_mid = (theta0 + theta1) / 2
        loop_uvs = [
            (theta0, phi_last),  # last ring current
            (theta1, phi_last),  # last ring next
            (theta_mid, phi_pole),  # pole
        ]
        poly = mesh.polygons[face_cursor]
        for k, loop_idx in enumerate(poly.loop_indices):
            uv_layer.data[loop_idx].uv = loop_uvs[k]
        face_cursor += 1

    obj = bpy.data.objects.new(name, mesh)

    if material is not None:
        _apply_material(obj, material)

    return Entity(obj)


def cone(
    name: str,
    radius1: float = 1.0,
    radius2: float = 0.0,
    depth: float = 2.0,
    vertices: int = 32,
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a cone mesh with UVs.

    *radius1*: bottom radius. *radius2*: top radius (0 for a point).
    *depth*: total height. *vertices*: number of segments around the axis.

    *tile_scale*: metres per UV unit (default 10).
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    half_d = depth / 2
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    # Bottom ring (indices 0 .. vertices-1)
    for i in range(vertices):
        angle = 2 * math.pi * i / vertices
        verts.append((radius1 * math.cos(angle), radius1 * math.sin(angle), -half_d))

    # Top ring (indices vertices .. 2*vertices-1)
    for i in range(vertices):
        angle = 2 * math.pi * i / vertices
        verts.append((radius2 * math.cos(angle), radius2 * math.sin(angle), half_d))

    # Center bottom (index 2*vertices), center top (index 2*vertices+1)
    verts.extend(((0.0, 0.0, -half_d), (0.0, 0.0, half_d)))
    bot_center = 2 * vertices
    top_center = 2 * vertices + 1

    # Side quads
    for i in range(vertices):
        j = (i + 1) % vertices
        faces.append((i, j, j + vertices, i + vertices))

    # Bottom cap triangles
    for i in range(vertices):
        j = (i + 1) % vertices
        faces.append((bot_center, j, i))

    # Top cap triangles (only if top radius > 0)
    if radius2 > 0:
        for i in range(vertices):
            j = (i + 1) % vertices
            faces.append((top_center, i + vertices, j + vertices))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # UVs: sides use arc-length / height, caps use planar XY
    uv_layer = mesh.uv_layers.new(name="UVMap")
    circumference = 2 * math.pi * max(radius1, radius2)
    slant = math.sqrt(depth**2 + (radius1 - radius2) ** 2)
    n_side = vertices
    n_bot = vertices

    for face_idx, poly in enumerate(mesh.polygons):
        if face_idx < n_side:
            u0 = circumference * face_idx / vertices * tile_uv
            u1 = circumference * (face_idx + 1) / vertices * tile_uv
            loop_uvs = [
                (u0, 0.0),
                (u1, 0.0),
                (u1, slant * tile_uv),
                (u0, slant * tile_uv),
            ]
            for k, loop_idx in enumerate(poly.loop_indices):
                uv_layer.data[loop_idx].uv = loop_uvs[k]
        elif face_idx < n_side + n_bot:
            # Bottom cap — planar XY shifted by radius1
            for loop_idx in poly.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                x, y, _ = verts[vi]
                uv_layer.data[loop_idx].uv = (
                    (x + radius1) * tile_uv,
                    (y + radius1) * tile_uv,
                )
        else:
            # Top cap — planar XY shifted by radius2
            for loop_idx in poly.loop_indices:
                vi = mesh.loops[loop_idx].vertex_index
                x, y, _ = verts[vi]
                uv_layer.data[loop_idx].uv = (
                    (x + radius2) * tile_uv,
                    (y + radius2) * tile_uv,
                )

    obj = bpy.data.objects.new(name, mesh)
    if material is not None:
        _apply_material(obj, material)
    return Entity(obj)


def torus(
    name: str,
    major_radius: float = 1.0,
    minor_radius: float = 0.25,
    major_segments: int = 48,
    minor_segments: int = 12,
    material: Material | None = None,
    tile_scale: float | None = None,
) -> Entity:
    """
    Create a torus mesh with UVs.

    *major_radius*: distance from torus centre to tube centre.
    *minor_radius*: tube radius.

    *tile_scale*: metres per UV unit (default 10).
    """
    tile_m = tile_scale if tile_scale is not None else _DEFAULT_TILE_M
    tile_uv = 1.0 / tile_m

    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, ...]] = []

    for i in range(major_segments):
        theta = 2 * math.pi * i / major_segments
        ct, st = math.cos(theta), math.sin(theta)
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            cp, sp = math.cos(phi), math.sin(phi)
            r = major_radius + minor_radius * cp
            verts.append((r * ct, r * st, minor_radius * sp))

    for i in range(major_segments):
        ni = (i + 1) % major_segments
        for j in range(minor_segments):
            nj = (j + 1) % minor_segments
            v0 = i * minor_segments + j
            v1 = ni * minor_segments + j
            v2 = ni * minor_segments + nj
            v3 = i * minor_segments + nj
            faces.append((v0, v1, v2, v3))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # UVs: U = major arc, V = minor arc, both scaled by tile_uv
    uv_layer = mesh.uv_layers.new(name="UVMap")
    major_circ = 2 * math.pi * major_radius
    minor_circ = 2 * math.pi * minor_radius

    for face_idx, poly in enumerate(mesh.polygons):
        i = face_idx // minor_segments
        j = face_idx % minor_segments
        u0 = major_circ * i / major_segments * tile_uv
        u1 = major_circ * (i + 1) / major_segments * tile_uv
        v0 = minor_circ * j / minor_segments * tile_uv
        v1 = minor_circ * (j + 1) / minor_segments * tile_uv
        loop_uvs = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        for k, loop_idx in enumerate(poly.loop_indices):
            uv_layer.data[loop_idx].uv = loop_uvs[k]

    obj = bpy.data.objects.new(name, mesh)
    if material is not None:
        _apply_material(obj, material)
    return Entity(obj)
