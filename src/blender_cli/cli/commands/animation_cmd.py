"""Animation CLI commands — keyframe, remove-keyframe, frame-range, fps, list-keyframes."""

from __future__ import annotations

import json
from typing import Any

import click

from blender_cli.animation.keyframes import (
    ANIMATABLE_PROPERTIES,
    INTERPOLATION_MODES,
    Animation,
)
from blender_cli.cli.common import _cli_json_errors
from blender_cli.project.project_file import ProjectFile


def _parse_value(raw: str, prop: str) -> Any:
    """Parse a CLI string value based on the property type."""
    vtype = ANIMATABLE_PROPERTIES.get(prop)
    if vtype is None:
        return raw

    if vtype == "vec3":
        parts = raw.split(",")
        if len(parts) != 3:
            msg = f"Property {prop!r} expects 3 comma-separated numbers (x,y,z)"
            raise click.BadParameter(msg)
        return [float(p.strip()) for p in parts]

    if vtype == "color":
        parts = raw.split(",")
        if len(parts) != 4:
            msg = f"Property {prop!r} expects 4 comma-separated numbers (r,g,b,a)"
            raise click.BadParameter(msg)
        return [float(p.strip()) for p in parts]

    if vtype == "float":
        return float(raw)

    if vtype == "bool":
        if raw.lower() in {"true", "1", "yes"}:
            return True
        if raw.lower() in {"false", "0", "no"}:
            return False
        msg = f"Property {prop!r} expects bool (true/false), got {raw!r}"
        raise click.BadParameter(msg)

    return raw


@click.group("animation")
def animation() -> None:
    """Animation system -- set keyframes, frame range, and FPS."""


@animation.command("keyframe")
@click.argument("object_ref")
@click.argument("frame", type=int)
@click.argument("prop")
@click.argument("value")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option(
    "-i",
    "--interpolation",
    default="BEZIER",
    type=click.Choice(sorted(INTERPOLATION_MODES)),
    help="Interpolation mode.",
)
def keyframe_cmd(
    object_ref: str,
    frame: int,
    prop: str,
    value: str,
    project_path: str,
    interpolation: str,
) -> None:
    """Set a keyframe on an object."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        parsed = _parse_value(value, prop)
        result = Animation.keyframe(
            pf.data, object_ref, frame, prop, parsed, interpolation
        )
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "animation.keyframe", "keyframe": result},
            indent=2,
        )
    )


@animation.command("remove-keyframe")
@click.argument("object_ref")
@click.argument("frame", type=int)
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--prop", default=None, help="Remove only keyframes for this property.")
def remove_keyframe_cmd(
    object_ref: str, frame: int, project_path: str, prop: str | None
) -> None:
    """Remove keyframe(s) at a frame."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        removed = Animation.remove(pf.data, object_ref, frame, prop)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "animation.remove-keyframe", "removed": removed},
            indent=2,
        )
    )


@animation.command("frame-range")
@click.argument("start", type=int)
@click.argument("end", type=int)
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def frame_range_cmd(start: int, end: int, project_path: str) -> None:
    """Set the scene frame range."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        result = Animation.set_frame_range(pf.data, start, end)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "animation.frame-range", "result": result},
            indent=2,
        )
    )


@animation.command("fps")
@click.argument("fps", type=int)
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def fps_cmd(fps: int, project_path: str) -> None:
    """Set the scene FPS."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        result = Animation.set_fps(pf.data, fps)
    pf.save()
    click.echo(
        json.dumps({"status": "ok", "action": "animation.fps", "fps": result}, indent=2)
    )


@animation.command("list-keyframes")
@click.argument("object_ref")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--prop", default=None, help="Filter by property.")
def list_keyframes_cmd(object_ref: str, project_path: str, prop: str | None) -> None:
    """List keyframes on an object."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        kfs = Animation.list(pf.data, object_ref, prop)
    click.echo(json.dumps(kfs, indent=2))
