"""CLI commands for world/environment configuration."""

from __future__ import annotations

import json

import click

from blender_cli.cli.common import _cli_json_errors
from blender_cli.project.project_file import ProjectFile
from blender_cli.render.world import WorldSettings


@click.group("world")
def world_cmd() -> None:
    """World/environment configuration."""


@world_cmd.command("set")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.option(
    "--background", default=None, help="Background color as R,G,B (0-1 each)."
)
@click.option("--hdri", default=None, type=click.Path(), help="Path to HDRI file.")
@click.option("--strength", default=None, type=float, help="HDRI strength.")
@click.option("--rotation", default=None, type=float, help="HDRI rotation in radians.")
def world_set(
    project_path: str,
    background: str | None,
    hdri: str | None,
    strength: float | None,
    rotation: float | None,
) -> None:
    """Set world/environment settings on a project."""
    pf = ProjectFile.load(project_path)
    w = pf.data["world"]

    if background is not None:
        parts = [p.strip() for p in background.split(",")]
        if len(parts) != 3:
            click.echo(
                json.dumps(
                    {"status": "error", "error": "background must be R,G,B"}, indent=2
                ),
                err=True,
            )
            raise SystemExit(1)
        with _cli_json_errors():
            color = [float(v) for v in parts]
        w["background_color"] = color

    if hdri is not None:
        w["use_hdri"] = True
        w["hdri_path"] = hdri

    if strength is not None:
        w["hdri_strength"] = strength

    if rotation is not None:
        w["hdri_rotation"] = rotation

    # Validate
    with _cli_json_errors():
        WorldSettings.from_dict(w)

    pf.save()
    click.echo(
        json.dumps({"status": "ok", "action": "world.set", "world": w}, indent=2)
    )


@world_cmd.command("info")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def world_info(project_path: str) -> None:
    """Display current world settings."""
    pf = ProjectFile.load(project_path)
    click.echo(json.dumps(pf.data["world"], indent=2))


@world_cmd.command("clear-hdri")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def world_clear_hdri(project_path: str) -> None:
    """Disable HDRI, revert to background color."""
    pf = ProjectFile.load(project_path)
    w = pf.data["world"]
    w["use_hdri"] = False
    w["hdri_path"] = None
    pf.save()
    click.echo(
        json.dumps({"status": "ok", "action": "world.clear-hdri", "world": w}, indent=2)
    )
