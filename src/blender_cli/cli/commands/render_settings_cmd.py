"""CLI commands for render engine settings, presets, and info."""

from __future__ import annotations

import json

import click

from blender_cli.cli.common import _cli_json_errors
from blender_cli.project.project_file import ProjectFile
from blender_cli.render.settings import (
    RENDER_PRESETS,
    VALID_ENGINES,
    VALID_FORMATS,
    RenderSettings,
)


@click.command("settings")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.option("--engine", type=click.Choice(sorted(VALID_ENGINES)), default=None)
@click.option("--samples", type=int, default=None)
@click.option("--denoising/--no-denoising", default=None)
@click.option("--transparent/--no-transparent", default=None)
@click.option(
    "--format", "output_format", type=click.Choice(sorted(VALID_FORMATS)), default=None
)
@click.option("--resolution-pct", type=click.IntRange(1, 100), default=None)
@click.option(
    "--preset",
    type=click.Choice(sorted(RENDER_PRESETS)),
    default=None,
    help="Apply a named render preset.",
)
def render_settings_cmd(
    project_path: str,
    engine: str | None,
    samples: int | None,
    denoising: bool | None,
    transparent: bool | None,
    output_format: str | None,
    resolution_pct: int | None,
    preset: str | None,
) -> None:
    """Configure render engine settings on a project."""
    pf = ProjectFile.load(project_path)
    rnd = pf.data["render"]

    # Apply preset first (individual flags override)
    if preset is not None:
        p = RENDER_PRESETS[preset]
        rnd["engine"] = p["engine"]
        rnd["samples"] = p["samples"]
        rnd["denoising"] = p["denoising"]
        rnd["film_transparent"] = p["film_transparent"]
        rnd["output_format"] = p["output_format"]
        rnd["resolution_pct"] = p["resolution_pct"]

    # Individual overrides
    if engine is not None:
        rnd["engine"] = engine
    if samples is not None:
        rnd["samples"] = samples
    if denoising is not None:
        rnd["denoising"] = denoising
    if transparent is not None:
        rnd["film_transparent"] = transparent
    if output_format is not None:
        rnd["output_format"] = output_format
    if resolution_pct is not None:
        rnd["resolution_pct"] = resolution_pct

    # Validate via RenderSettings
    with _cli_json_errors():
        RenderSettings(
            engine=rnd["engine"],
            samples=rnd["samples"],
            denoising=rnd.get("denoising", False),
            film_transparent=rnd.get("film_transparent", False),
            output_format=rnd["output_format"],
            resolution_pct=rnd.get("resolution_pct", 100),
        )

    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "render.settings", "render": rnd}, indent=2
        )
    )


@click.command("presets")
def render_presets_cmd() -> None:
    """List all available render presets."""
    out = []
    for name, cfg in sorted(RENDER_PRESETS.items()):
        out.append({"name": name, **cfg})
    click.echo(json.dumps(out, indent=2))


@click.command("info")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def render_info_cmd(project_path: str) -> None:
    """Show current render settings including effective resolution."""
    pf = ProjectFile.load(project_path)
    rnd = pf.data["render"]
    res = rnd["resolution"]
    pct = rnd.get("resolution_pct", 100)
    effective = [int(res[0] * pct / 100), int(res[1] * pct / 100)]
    info = {
        **rnd,
        "effective_resolution": effective,
    }
    click.echo(json.dumps(info, indent=2))
