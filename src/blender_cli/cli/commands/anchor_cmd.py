"""Anchor management commands — add, remove, list."""

from __future__ import annotations

import json

import click

from blender_cli.project.project_file import ProjectFile


@click.group("anchor")
def anchor_cmd() -> None:
    """Anchor point management — named spatial control points."""


@anchor_cmd.command("add")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--pos", required=True, nargs=3, type=float, help="Position x y z.")
@click.option("--annotation", default=None, help="Optional annotation string.")
def add(project_path: str, name: str, pos: tuple[float, float, float], annotation: str | None) -> None:
    """Add a named anchor point."""
    pf = ProjectFile.load(project_path)
    a = pf.add_anchor(name, list(pos), annotation=annotation)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "anchor.add", "anchor": a}, indent=2))


@anchor_cmd.command("remove")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
def remove(project_path: str, name: str) -> None:
    """Remove an anchor by name."""
    pf = ProjectFile.load(project_path)
    removed = pf.remove_anchor(name)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "anchor.remove", "removed": removed}, indent=2))


@anchor_cmd.command("list")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def list_anchors(project_path: str) -> None:
    """List all anchors in the project."""
    pf = ProjectFile.load(project_path)
    anchors = pf.data.get("anchors", [])
    click.echo(json.dumps({"status": "ok", "count": len(anchors), "anchors": anchors}, indent=2))
