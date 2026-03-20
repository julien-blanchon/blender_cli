"""Material management commands — add, remove, list."""

from __future__ import annotations

import json

import click

from blender_cli.project.project_file import ProjectFile


@click.group("material")
def material_cmd() -> None:
    """Material management — PBR folder refs and flat-color materials."""


@material_cmd.command("add")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--pbr-folder", default=None, help="Path to PBR texture folder.")
@click.option("--tiling", nargs=2, type=float, default=None, help="UV tiling [u, v].")
@click.option("--color", nargs=4, type=float, default=None, help="Base color RGBA (0..1).")
@click.option("--roughness", type=float, default=None, help="Roughness (0..1).")
@click.option("--metallic", type=float, default=None, help="Metallic (0..1).")
def add(
    project_path: str,
    name: str,
    pbr_folder: str | None,
    tiling: tuple[float, float] | None,
    color: tuple[float, float, float, float] | None,
    roughness: float | None,
    metallic: float | None,
) -> None:
    """Add a named material definition."""
    pf = ProjectFile.load(project_path)
    mat = pf.add_material(
        name,
        pbr_folder=pbr_folder,
        tiling=list(tiling) if tiling else None,
        color=list(color) if color else None,
        roughness=roughness,
        metallic=metallic,
    )
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "material.add", "material": mat}, indent=2))


@material_cmd.command("remove")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
def remove(project_path: str, name: str) -> None:
    """Remove a material by name."""
    pf = ProjectFile.load(project_path)
    removed = pf.remove_material(name)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "material.remove", "removed": removed}, indent=2))


@material_cmd.command("list")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def list_materials(project_path: str) -> None:
    """List all materials in the project."""
    pf = ProjectFile.load(project_path)
    materials = pf.data.get("materials", [])
    click.echo(json.dumps({"status": "ok", "count": len(materials), "materials": materials}, indent=2))
