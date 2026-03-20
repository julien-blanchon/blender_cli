"""Camera CLI commands — DOF configuration."""

from __future__ import annotations

import json
from typing import Any

import click

from blender_cli.cli.common import _cli_json_errors
from blender_cli.project.project_file import ProjectFile
from blender_cli.render.settings import validate_dof

_DOF_PROPS = {"dof_enabled", "dof_focus_distance", "dof_aperture"}


def _parse_camera_value(prop: str, raw: str) -> Any:
    """Parse a string value for a camera property."""
    if prop == "dof_enabled":
        if raw.lower() in {"true", "1", "yes"}:
            return True
        if raw.lower() in {"false", "0", "no"}:
            return False
        msg = f"dof_enabled expects bool, got {raw!r}"
        raise click.BadParameter(msg)
    if prop in {"dof_focus_distance", "dof_aperture"}:
        return float(raw)
    msg = f"Unknown camera property {prop!r}. Valid: {sorted(_DOF_PROPS)}"
    raise click.BadParameter(msg)


@click.group("camera")
def camera_cmd() -> None:
    """Camera commands — DOF configuration."""


@camera_cmd.command("set")
@click.argument("index", type=int)
@click.argument("prop")
@click.argument("value")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def camera_set_cmd(index: int, prop: str, value: str, project_path: str) -> None:
    """Set a camera DOF property (dof_enabled, dof_focus_distance, dof_aperture)."""
    pf = ProjectFile.load(project_path)
    cameras = pf.data["cameras"]

    if index < 0 or index >= len(cameras):
        click.echo(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Camera index {index} out of range (0..{len(cameras) - 1})",
                },
                indent=2,
            ),
            err=True,
        )
        raise SystemExit(1)

    if prop not in _DOF_PROPS:
        click.echo(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Unknown property {prop!r}. Valid: {sorted(_DOF_PROPS)}",
                },
                indent=2,
            ),
            err=True,
        )
        raise SystemExit(1)

    with _cli_json_errors():
        parsed = _parse_camera_value(prop, value)

    cam = cameras[index]
    dof = cam.setdefault(
        "dof", {"dof_enabled": False, "dof_focus_distance": 10.0, "dof_aperture": 2.8}
    )
    dof[prop] = parsed

    errors = validate_dof(dof)
    if errors:
        click.echo(
            json.dumps({"status": "error", "error": "; ".join(errors)}, indent=2),
            err=True,
        )
        raise SystemExit(1)

    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "camera.set", "camera_index": index, "dof": dof},
            indent=2,
        )
    )


@camera_cmd.command("info")
@click.argument("index", type=int)
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def camera_info_cmd(index: int, project_path: str) -> None:
    """Show camera info including DOF settings."""
    pf = ProjectFile.load(project_path)
    cameras = pf.data["cameras"]

    if index < 0 or index >= len(cameras):
        click.echo(
            json.dumps(
                {
                    "status": "error",
                    "error": f"Camera index {index} out of range (0..{len(cameras) - 1})",
                },
                indent=2,
            ),
            err=True,
        )
        raise SystemExit(1)

    click.echo(json.dumps(cameras[index], indent=2))
