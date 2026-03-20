"""Shared CLI helpers used across command modules."""

from __future__ import annotations

import contextlib
import json
import operator
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from blender_cli.scene import Scene, Selection
from blender_cli.types import NearbyEntry, Vec3

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextlib.contextmanager
def _quiet() -> Iterator[None]:
    """Redirect the real stdout fd to devnull (suppresses bpy C-level noise)."""
    stdout_fd = sys.stdout.fileno()
    saved_fd = os.dup(stdout_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stdout_fd)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(saved_fd, stdout_fd)
        os.close(saved_fd)


def _output(data: object, path: str | None) -> None:
    """Write JSON to file or stdout."""
    text = json.dumps(data, indent=2)
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    else:
        click.echo(text)


def _json_ok(
    action: str, result: object = None, warnings: list[str] | None = None
) -> dict:
    """Build a standard JSON success envelope."""
    out: dict[str, object] = {"status": "ok", "action": action}
    if result is not None:
        out["result"] = result
    if warnings:
        out["warnings"] = warnings
    return out


def _json_error(message: str, exc_type: str = "Error") -> dict:
    """Build a standard JSON error envelope."""
    return {"status": "error", "error": message, "type": exc_type}


@contextlib.contextmanager
def _cli_json_errors() -> Iterator[None]:
    """Catch common CLI errors and emit a JSON error envelope to stderr."""
    try:
        yield
    except (ValueError, IndexError, KeyError, click.BadParameter) as exc:
        click.echo(json.dumps(_json_error(str(exc)), indent=2), err=True)
        raise SystemExit(1) from exc


def _resolve_where(scene: Scene, where: str) -> Selection:
    """Resolve ``--where``: ref shorthand (uid:, name:, ann:, tag:) or query DSL."""
    if ":" in where:
        prefix, _, value = where.partition(":")
        if prefix == "uid":
            return Selection([o for o in scene.objects() if Scene.uid(o) == value])
        if prefix == "name":
            return Selection([o for o in scene.objects() if o.name == value])
        if prefix == "ann":
            return scene.select(f"annotations.has('{value}')")
        if prefix == "tag":
            return scene.select(f"tags.has('{value}')")
    return scene.select(where)


def _ref_position(scene: Scene, ref: str) -> Vec3:
    """Resolve a ref to a world-space position (first matched object)."""
    sel = _resolve_where(scene, ref)
    obj = sel.first()
    if obj is None:
        msg = f"No object for ref: {ref}"
        raise click.ClickException(msg)
    loc = obj.location
    return Vec3(float(loc.x), float(loc.y), float(loc.z))


def _scene_context(scene: Scene) -> dict[str, object]:
    """Build scene summary context for CLI output."""
    objs = scene.objects()
    by_tag: dict[str, int] = {}
    for obj in objs:
        for tag in Scene.tags(obj):
            by_tag[tag] = by_tag.get(tag, 0) + 1
    bb = scene.bbox()
    ctx: dict[str, object] = {
        "total_objects": len(objs),
        "by_tag": by_tag,
    }
    if bb:
        ctx["bbox"] = {
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
    return ctx


def _nearby_at_position(
    scene: Scene,
    pos: Vec3,
    radius: float = 20.0,
    limit: int = 5,
) -> list[NearbyEntry]:
    """Find tracked objects near a world-space position."""
    entries: list[tuple[float, NearbyEntry]] = []
    for obj in scene.objects():
        o_loc = obj.matrix_world.translation
        o_pos = Vec3(float(o_loc.x), float(o_loc.y), float(o_loc.z))
        dist = pos.distance(o_pos)
        if dist <= radius:
            entries.append((
                dist,
                NearbyEntry(
                    uid=Scene.uid(obj) or "",
                    name=obj.name,
                    distance=round(dist, 2),
                    tags=sorted(Scene.tags(obj)),
                ),
            ))
    entries.sort(key=operator.itemgetter(0))
    return [e[1] for e in entries[:limit]]


def _check_placement_warnings(
    scene: Scene,
    pos: Vec3,
    *,
    rescale_fit: float | None = None,
    scale_factor: float | None = None,
) -> list[str]:
    """
    Check if a placed object at *pos* would overlap structure entities.

    Uses a conservative half-extent estimate based on *rescale_fit* or
    *scale_factor*.  Returns a list of human-readable warning strings.
    """
    import mathutils as _mu  # noqa: PLC0415

    warnings: list[str] = []

    # Estimate the placed object's half-extent on each axis.
    # rescale_fit sets the longest dim; the object extends ±half in XY,
    # and from 0..rescale_fit in Z (it sits on the floor after snap).
    if rescale_fit is not None:
        half = rescale_fit / 2.0
    elif scale_factor is not None:
        # Generic models are ~2 units max → after scale, half ≈ scale_factor
        half = scale_factor
    else:
        return warnings  # no size info → cannot check

    obj_min_x = pos.x - half
    obj_max_x = pos.x + half
    obj_min_y = pos.y - half
    obj_max_y = pos.y + half

    # Check against every structure entity (walls only — skip floor/ceiling).
    for obj in scene.objects():
        tags = Scene.tags(obj)
        if "structure" not in tags and "wall" not in tags:
            continue
        if obj.type != "MESH":
            continue
        # Skip floor/ceiling (they don't form vertical barriers).
        if "floor" in tags or "ceiling" in tags:
            continue

        # Compute world-space AABB of the structure entity (XY plane only).
        corners = [obj.matrix_world @ _mu.Vector(c) for c in obj.bound_box]
        s_min_x = min(c.x for c in corners)
        s_max_x = max(c.x for c in corners)
        s_min_y = min(c.y for c in corners)
        s_max_y = max(c.y for c in corners)

        # AABB overlap test (XY plane is sufficient for wall collision).
        overlap_x = max(0.0, min(obj_max_x, s_max_x) - max(obj_min_x, s_min_x))
        overlap_y = max(0.0, min(obj_max_y, s_max_y) - max(obj_min_y, s_min_y))

        if overlap_x > 0.001 and overlap_y > 0.001:
            name = obj.name
            warnings.append(
                f"Object may overlap with '{name}' "
                f"(est. overlap ~{max(overlap_x, overlap_y):.3f}m). "
                f"Consider adjusting position."
            )

    # Check against the scene's structure bbox (room boundaries).
    struct_bb = scene.bbox(tags={"structure"})
    if struct_bb:
        lo, hi = struct_bb
        protrusions: list[str] = []
        if obj_min_x < lo.x - 0.001:
            protrusions.append(f"-X by {lo.x - obj_min_x:.3f}m")
        if obj_max_x > hi.x + 0.001:
            protrusions.append(f"+X by {obj_max_x - hi.x:.3f}m")
        if obj_min_y < lo.y - 0.001:
            protrusions.append(f"-Y by {lo.y - obj_min_y:.3f}m")
        if obj_max_y > hi.y + 0.001:
            protrusions.append(f"+Y by {obj_max_y - hi.y:.3f}m")
        if protrusions:
            warnings.append(
                f"Object may extend beyond room boundary: "
                f"{', '.join(protrusions)}. "
                f"Consider shifting position inward."
            )

    return warnings


def _write_script(path: str, code: str) -> None:
    """Write a generated Python script to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(code, encoding="utf-8")


def _codegen_select(where: str) -> tuple[str, str]:
    """Convert ``--where`` to (Python expression, extra import line) for codegen."""
    if ":" in where:
        prefix, _, value = where.partition(":")
        if prefix == "tag":
            return f"scene.select(\"tags.has('{value}')\")", ""
        if prefix == "ann":
            return f"scene.select(\"annotations.has('{value}')\")", ""
        if prefix == "uid":
            return (
                f"Selection([o for o in scene.objects() if Scene.uid(o) == {value!r}])",
                "from blender_cli.scene import Selection\n",
            )
        if prefix == "name":
            return (
                f"Selection([o for o in scene.objects() if o.name == {value!r}])",
                "from blender_cli.scene import Selection\n",
            )
    return f"scene.select({where!r})", ""


# ---------------------------------------------------------------------------
# Shared codegen / parsing helpers for new op & candidate commands
# ---------------------------------------------------------------------------


def _parse_kv_pairs(values: tuple[str, ...]) -> dict[str, str]:
    """Parse repeatable ``key=value`` CLI arguments into a dict."""
    result: dict[str, str] = {}
    for kv in values:
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        result[k.strip()] = v.strip()
    return result


def _parse_point(s: str) -> tuple[float, float, float]:
    """Parse a ``x,y,z`` string into a 3-float tuple."""
    parts = s.split(",")
    if len(parts) != 3:
        msg = f"Expected x,y,z but got {s!r}"
        raise click.BadParameter(msg)
    return float(parts[0]), float(parts[1]), float(parts[2])


def _resolve_anchor_or_pos(
    scene: Scene,
    anchor_name: str | None,
    point: tuple[float, float, float] | None,
) -> Vec3:
    """Resolve an anchor name OR explicit (x,y,z) coords to a Vec3."""
    if anchor_name:
        a = scene.anchor(anchor_name)
        if a is None:
            msg = f"Anchor not found: {anchor_name}"
            raise click.ClickException(msg)
        return a.location()
    if point is not None:
        return Vec3(*point)
    msg = "Provide either --anchor or explicit coordinates"
    raise click.ClickException(msg)


def _material_codegen(
    material_id: str | None,
    tile: float | None,
    color: tuple[float, ...] | None,
    metallic: float | None,
    roughness: float | None,
) -> tuple[str, str]:
    """
    Generate material code snippet and its required import line.

    Returns ``(code_str, import_str)``.  ``code_str`` is a Python expression
    producing a Material, or ``"None"`` when no material is needed.
    """
    if material_id:
        tile_arg = f", tile={tile}" if tile else ""
        code = f"scene.materials.pbr({material_id!r}{tile_arg})"
        imp = ""
        return code, imp

    if color:
        rgba = ", ".join(str(c) for c in color)
        met = metallic if metallic is not None else 0.0
        rough = roughness if roughness is not None else 0.5
        code = f"Material.from_color(({rgba}), metallic={met}, roughness={rough})"
        imp = "from blender_cli.assets import Material\n"
        return code, imp

    return "None", ""
