"""SplineStrip — glTF-compatible texture overlay along a spline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import bpy

from blender_cli.scene.entity import Entity
from blender_cli.types import Vec3

if TYPE_CHECKING:
    from blender_cli.assets.material import Material
    from blender_cli.geometry.heightfield import Heightfield
    from blender_cli.geometry.spline import Spline


def spline_strip(
    name: str,
    spline: Spline,
    material: Material,
    width: float,
    conform_to: Heightfield,
    *,
    uv_mode: Literal["tile", "stretch"] = "tile",
    v_density: float = 1.0,
    u_repeats: float = 1.0,
    falloff: float = 0.0,
    falloff_curve: Literal["linear", "smooth"] = "smooth",
    z_offset: float = 0.005,
    alpha_mode: Literal["OPAQUE", "BLEND", "MASK"] = "BLEND",
    alpha_cutoff: float = 0.5,
    opacity: float = 1.0,
    overlay_on: Entity | None = None,
    overlay_tile_scale: float | None = None,
    overlay_channels: tuple[str, ...] = ("base_color", "roughness"),
) -> Entity:
    """
    Build a thin mesh strip conforming to terrain along a spline.

    The strip gets proper UV mapping and vertex-color alpha for edge
    falloff — all glTF 2.0 compatible.

    **Texture overlay mode** (``overlay_on`` is set): instead of creating a
    separate mesh, the path texture is painted directly onto the terrain's
    material using a mask-driven blend.  No separate geometry = no z-fighting.

    Parameters
    ----------
    name:
        Blender object name.
    spline:
        Path to follow.
    material:
        Material to apply.  ``set_alpha_mode`` and ``set_vertex_color_alpha``
        will be called automatically when *falloff* > 0.
    width:
        Strip width in metres.
    conform_to:
        Heightfield the strip drapes onto.
    uv_mode:
        ``"tile"``: V repeats every metre × *v_density*.
        ``"stretch"``: V goes 0→1 over the entire path.
    v_density:
        Tiles per metre along path (for ``"tile"`` mode).
    u_repeats:
        Tiles across width.
    falloff:
        Metres of edge fadeout (0 = hard edge).
    falloff_curve:
        ``"linear"`` or ``"smooth"`` (Hermite).
    z_offset:
        Constant Z lift to prevent z-fighting.
    alpha_mode:
        glTF alpha mode applied to the material.
    alpha_cutoff:
        Threshold for ``"MASK"`` mode.
    opacity:
        Centre-vertex alpha (1.0 = fully opaque).
    overlay_on:
        If set, paint the path texture onto this terrain entity's material
        instead of creating a separate mesh strip.  Returns the same entity.
    overlay_tile_scale:
        UV tiling scale for overlay textures (only used in overlay mode).
    overlay_channels:
        PBR channels to blend in overlay mode (default: base_color, roughness).

    """
    # -- Texture overlay mode --
    if overlay_on is not None:
        from blender_cli.assets.material import Material as Mat
        from blender_cli.geometry.mask import Mask

        # Generate mask aligned to terrain grid
        mask = Mask.from_spline(spline, width, falloff=falloff, reference=conform_to)

        # Get the terrain material from the overlay target
        target_obj = overlay_on.target
        if not isinstance(target_obj, bpy.types.Object):
            msg = "overlay_on target must be a Blender object"
            raise TypeError(msg)
        mesh_data = target_obj.data
        if not isinstance(mesh_data, bpy.types.Mesh) or not mesh_data.materials:
            msg = "overlay_on target has no materials"
            raise ValueError(msg)

        terrain_mat_bpy = mesh_data.materials[0]
        terrain_mat = Mat(terrain_mat_bpy.name)

        # Apply texture overlay
        terrain_mat.apply_texture_overlay(
            overlay=material,
            mask_array=mask.to_numpy(),
            channels=overlay_channels,
            overlay_tile_scale=overlay_tile_scale,
        )

        return overlay_on
    half = width / 2.0
    mpp = conform_to.meters_per_px
    n_lateral = max(2, int(width / mpp) + 1)

    # Lateral offsets (left to right)
    lat_offsets = [-half + i * width / (n_lateral - 1) for i in range(n_lateral)]

    points = spline.points
    n_rings = len(points)
    n_segs = n_rings if spline.closed else n_rings - 1

    # Build arc-length at each control point for UV
    arc_lengths = [0.0]
    for i in range(1, n_rings):
        arc_lengths.append(arc_lengths[-1] + points[i - 1].distance(points[i]))
    total_arc = arc_lengths[-1] if arc_lengths[-1] > 1e-9 else 1.0

    # Build vertices
    verts: list[tuple[float, float, float]] = []
    for i, pt in enumerate(points):
        t = i / n_segs if n_segs > 0 else 0.0
        t = min(t, 1.0)
        tang = spline.tangent(t)

        left = Vec3(-tang.y, tang.x, 0.0)
        left = Vec3(1.0, 0.0, 0.0) if left.length() < 1e-9 else left.normalized()

        for lat in lat_offsets:
            vx = pt.x + left.x * lat
            vy = pt.y + left.y * lat
            vz = conform_to.sample_at(vx, vy) + z_offset
            verts.append((vx, vy, vz))

    # Quad faces
    faces: list[tuple[int, int, int, int]] = []
    for i in range(n_rings - 1):
        for j in range(n_lateral - 1):
            v0 = i * n_lateral + j
            v1 = i * n_lateral + j + 1
            v2 = (i + 1) * n_lateral + j + 1
            v3 = (i + 1) * n_lateral + j
            faces.append((v0, v3, v2, v1))

    if spline.closed:
        last = n_rings - 1
        for j in range(n_lateral - 1):
            v0 = last * n_lateral + j
            v1 = last * n_lateral + j + 1
            v2 = j + 1
            v3 = j
            faces.append((v0, v3, v2, v1))

    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(verts, [], faces)

    # UV layer
    uv_layer = mesh.uv_layers.new(name="UVMap")
    for face in mesh.polygons:
        for loop_idx in face.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            ring = vi // n_lateral
            prof = vi % n_lateral

            # U: lateral position scaled by u_repeats
            u = (prof / max(n_lateral - 1, 1)) * u_repeats

            # V: distance along path
            arc = arc_lengths[min(ring, len(arc_lengths) - 1)]
            if uv_mode == "stretch":
                v = arc / total_arc
            else:  # tile
                v = arc * v_density

            uv_layer.data[loop_idx].uv = (u, v)

    # Vertex colors for alpha falloff (COLOR_0 — glTF standard)
    color_layer = mesh.color_attributes.new(
        name="COLOR_0", type="BYTE_COLOR", domain="CORNER"
    )

    for face in mesh.polygons:
        for loop_idx in face.loop_indices:
            vi = mesh.loops[loop_idx].vertex_index
            prof = vi % n_lateral

            # Compute alpha based on lateral position
            lateral_pos = prof / max(n_lateral - 1, 1)  # 0..1
            dist_from_edge = min(lateral_pos, 1.0 - lateral_pos) * width

            if falloff > 0 and dist_from_edge < falloff:
                t = dist_from_edge / falloff
                if falloff_curve == "smooth":
                    alpha = opacity * t * t * (3.0 - 2.0 * t)
                else:
                    alpha = opacity * t
            else:
                alpha = opacity

            color_layer.data[loop_idx].color = (1.0, 1.0, 1.0, alpha)

    mesh.update()
    for poly in mesh.polygons:
        poly.use_smooth = True

    obj = bpy.data.objects.new(name, mesh)

    # Apply material with alpha settings
    mat = material.get_or_create()
    obj.data.materials.append(mat)

    if falloff > 0 or opacity < 1.0:
        material.set_alpha_mode(alpha_mode, cutoff=alpha_cutoff)
        material.set_vertex_color_alpha("COLOR_0")

    return Entity(obj)
