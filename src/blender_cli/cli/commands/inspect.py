"""Scene inspection commands."""

from __future__ import annotations

import json
import operator
from pathlib import Path

import click

from blender_cli.cli.common import _output, _quiet, _resolve_where
from blender_cli.scene import Scene
from blender_cli.types import Vec3


@click.group()
def inspect() -> None:
    """Inspect commands."""


@inspect.command("scene")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--json", "json_out", default=None, type=click.Path())
def inspect_scene(glb: str, json_out: str | None) -> None:
    """Dump full scene description as JSON."""
    with _quiet():
        scene = Scene.load(glb)
        desc = json.loads(scene.describe())
    _output(desc, json_out)


@inspect.command("selection")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--json", "json_out", default=None, type=click.Path())
def inspect_selection(glb: str, where: str, json_out: str | None) -> None:
    """Dump matching entities as JSON."""
    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        desc = json.loads(scene.describe(sel))
    _output(desc, json_out)


@inspect.command("manifest")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--json", "json_out", default=None, type=click.Path())
def inspect_manifest(glb: str, json_out: str | None) -> None:
    """Dump scene manifest JSON from a GLB (embedded in scene extras)."""
    import bpy

    from blender_cli.cli.common import _quiet
    from blender_cli.scene import Scene

    with _quiet():
        Scene.load(glb)
    manifest_str = bpy.context.scene.get("_scene_manifest")
    if not manifest_str:
        msg = "No manifest found in GLB scene extras"
        raise click.ClickException(msg)
    _output(json.loads(manifest_str), json_out)


@inspect.command("scale")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--json", "json_out", default=None, type=click.Path())
def inspect_scale(glb: str, json_out: str | None) -> None:
    """Dump world-scale metrics for a GLB scene."""
    with _quiet():
        scene = Scene.load(glb)
        bb = scene.bbox()

    if bb is None:
        _output({"error": "Scene is empty — no bounding box"}, json_out)
        return

    lo, hi = bb
    ext = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
    diag = ext.length()
    center = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)

    # Per-object info
    obj_entries: list[dict[str, object]] = []
    for obj in scene.objects():
        dims = Scene._object_dimensions(obj)
        obj_diag = dims.length()
        ratio = obj_diag / diag if diag > 1e-9 else 0.0
        tags = sorted(scene.tags(obj))
        obj_entries.append({
            "name": obj.name,
            "tags": tags,
            "dimensions": {
                "x": round(dims.x, 4),
                "y": round(dims.y, 4),
                "z": round(dims.z, 4),
            },
            "diagonal": round(obj_diag, 4),
            "ratio_to_scene": round(ratio, 6),
        })
    obj_entries.sort(key=operator.itemgetter("diagonal"), reverse=True)  # type: ignore[arg-type]

    result: dict[str, object] = {
        "bounding_box": {
            "min": {"x": round(lo.x, 4), "y": round(lo.y, 4), "z": round(lo.z, 4)},
            "max": {"x": round(hi.x, 4), "y": round(hi.y, 4), "z": round(hi.z, 4)},
        },
        "world_metrics": {
            "extent": {
                "x": round(ext.x, 4),
                "y": round(ext.y, 4),
                "z": round(ext.z, 4),
            },
            "diagonal": round(diag, 4),
            "center": {
                "x": round(center.x, 4),
                "y": round(center.y, 4),
                "z": round(center.z, 4),
            },
            "ground_area": round(ext.x * ext.y, 4),
            "unit": "metres",
        },
        "objects": obj_entries,
    }
    _output(result, json_out)
