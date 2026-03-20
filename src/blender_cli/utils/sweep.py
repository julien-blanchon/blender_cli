"""
Sweep — extrude a 3D cross-section profile along a spline.

Best suited for geometry that needs vertical thickness or a shaped
cross-section (walls, pipes, rails, bridge spans, stairways).

For flat texture overlays (cobblestone paths, dirt trails, tire tracks)
prefer :func:`blender_cli.utils.spline_strip.spline_strip` which
produces a terrain-conforming strip with proper UVs and vertex-color
alpha falloff.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import bpy

from blender_cli.scene.entity import Entity
from blender_cli.types import Vec3

if TYPE_CHECKING:
    from blender_cli.assets.material import Material
    from blender_cli.geometry.heightfield import Heightfield
    from blender_cli.geometry.spline import Spline


def sweep(
    name: str,
    spline: Spline,
    material: Material | None = None,
    *,
    width: float | None = None,
    profile: list[tuple[float, float]] | None = None,
    conform_to: Heightfield | None = None,
    z_offset: float = 0.0,
) -> Entity:
    """
    Extrude a cross-section profile along a spline to create road/river/path meshes.

    The profile is swept perpendicular to the spline at each control point.
    For smooth results, resample the spline first (``spline.resample(1.0)``).

    Args:
        name: Blender object name.
        spline: Path to follow.
        material: Optional material to apply.
        width: Width for the default flat-strip profile (metres).
        profile: Custom cross-section as ``(lateral_offset, vertical_offset)`` pairs.
            Lateral: positive = left of travel, negative = right.
            Overrides *width* if both are given.
        conform_to: Optional heightfield to drape the mesh onto.  Each vertex
            samples the heightfield at its own XY position so the mesh
            conforms to terrain topology (including laterally displaced
            profile points).
        z_offset: Constant Z offset added to every vertex when *conform_to*
            is used (e.g. 0.3 to float a river above the carved bed).

    Returns:
        The generated Blender mesh object (not yet linked to a scene).

    """
    if profile is None:
        if width is None:
            msg = "Either width or profile must be provided"
            raise ValueError(msg)
        half = width / 2.0
        if conform_to is not None:
            # Subdivide the profile so each vertex independently samples
            # the heightfield — prevents the flat strip from cutting through
            # terrain features (channel walls, shoulders, etc.).
            step = conform_to.meters_per_px
            n_lateral = max(2, int(width / step) + 1)
            profile = [
                (-half + i * width / (n_lateral - 1), 0.0) for i in range(n_lateral)
            ]
        else:
            profile = [(-half, 0.0), (half, 0.0)]

    if len(profile) < 2:
        msg = "Profile must have at least 2 points"
        raise ValueError(msg)

    points = spline.points
    n_rings = len(points)
    n_profile = len(profile)
    n_segs = n_rings if spline.closed else n_rings - 1

    # Build vertices: one ring of profile points per spline control point.
    verts: list[tuple[float, float, float]] = []
    for i, pt in enumerate(points):
        t = i / n_segs if n_segs > 0 else 0.0
        t = min(t, 1.0)
        tang = spline.tangent(t)

        # Left-of-travel normal in the XY plane.
        left = Vec3(-tang.y, tang.x, 0.0)
        left = Vec3(1.0, 0.0, 0.0) if left.length() < 1e-09 else left.normalized()

        for lat, vert in profile:
            vx = pt.x + left.x * lat
            vy = pt.y + left.y * lat
            if conform_to is not None:
                vz = conform_to.sample_at(vx, vy) + vert + z_offset
            else:
                vz = pt.z + vert
            verts.append((vx, vy, vz))

    # Quad faces between adjacent rings (CCW from above → normals face +Z).
    faces: list[tuple[int, int, int, int]] = []
    for i in range(n_rings - 1):
        for j in range(n_profile - 1):
            v0 = i * n_profile + j
            v1 = i * n_profile + j + 1
            v2 = (i + 1) * n_profile + j + 1
            v3 = (i + 1) * n_profile + j
            faces.append((v0, v3, v2, v1))

    # Close the loop for closed splines.
    if spline.closed:
        last = n_rings - 1
        for j in range(n_profile - 1):
            v0 = last * n_profile + j
            v1 = last * n_profile + j + 1
            v2 = j + 1
            v3 = j
            faces.append((v0, v3, v2, v1))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # Generate UVs: U = distance along spline, V = lateral position in profile
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for _face_idx, face in enumerate(mesh.polygons):
        for loop_idx in face.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            ring = vi // n_profile
            prof = vi % n_profile
            u = ring / max(n_rings - 1, 1)
            v = prof / max(n_profile - 1, 1)
            uv_layer.data[loop_idx].uv = (u * (n_rings / 10.0), v)

    mesh.update()

    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)

    if material is not None:
        mat = material.get_or_create()
        obj.data.materials.append(mat)

    return Entity(obj)
