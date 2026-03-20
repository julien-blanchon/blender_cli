"""Raycast command."""

from __future__ import annotations

import bpy
import click

from blender_cli.cli.common import _output, _quiet, _scene_context
from blender_cli.scene import Scene
from blender_cli.snap import AXIS_DIR
from blender_cli.types import Vec3


@click.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--origin", required=True, nargs=3, type=float, help="Ray origin x y z.")
@click.option(
    "--axis", default="-Z", type=click.Choice(sorted(AXIS_DIR)), show_default=True
)
@click.option(
    "--ignore", default=None, help="Comma-separated tags to exclude from hit."
)
@click.option("--json", "json_out", default=None, type=click.Path())
def raycast(
    glb: str,
    origin: tuple[float, float, float],
    axis: str,
    ignore: str | None,
    json_out: str | None,
) -> None:
    """Perform a single raycast and report hit."""
    with _quiet():
        scene = Scene.load(glb)

        hidden: list[tuple[bpy.types.Object, bool]] = []
        if ignore:
            ignore_tags = {t.strip() for t in ignore.split(",")}
            for obj in scene.objects():
                if Scene.tags(obj) & ignore_tags:
                    hidden.append((obj, obj.hide_viewport))
                    obj.hide_viewport = True

        try:
            vl = bpy.context.view_layer
            if vl is None:
                msg = "No active view layer"
                raise RuntimeError(msg)
            vl.update()
            dg = bpy.context.evaluated_depsgraph_get()
            ok, loc, normal, _idx, hit_obj, _mat = scene.bpy_scene.ray_cast(
                dg,
                origin,
                AXIS_DIR[axis],
            )
        finally:
            for obj, vis in hidden:
                obj.hide_viewport = vis

        ctx = _scene_context(scene)

    if ok:
        hp = Vec3(float(loc.x), float(loc.y), float(loc.z))
        hn = Vec3(float(normal.x), float(normal.y), float(normal.z))
        data: dict[str, object] = {
            "hit": True,
            "hit_pos": {"x": round(hp.x, 4), "y": round(hp.y, 4), "z": round(hp.z, 4)},
            "hit_normal": {
                "x": round(hn.x, 4),
                "y": round(hn.y, 4),
                "z": round(hn.z, 4),
            },
            "hit_uid": Scene.uid(hit_obj) if hit_obj else None,
            "hit_distance": round(Vec3(*origin).distance(hp), 4),
        }
    else:
        data = {
            "hit": False,
            "hit_pos": None,
            "hit_normal": None,
            "hit_uid": None,
            "hit_distance": -1,
        }

    data["context"] = ctx
    _output(data, json_out)
