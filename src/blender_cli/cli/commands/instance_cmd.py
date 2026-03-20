"""Instance management commands — add, remove, list GPU-instanced groups."""

from __future__ import annotations

import json

import click

from blender_cli.project.project_file import ProjectFile


@click.group("instance")
def instance_cmd() -> None:
    """Instance group management — GPU-instanced object placement."""


@instance_cmd.command("add")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--prefab", required=True, help="Path to prefab GLB file.")
@click.option("--points", required=True, help="Points as JSON: [[x,y,z], ...]")
@click.option(
    "--align", default="bottom", type=click.Choice(["center", "bottom", "top"])
)
@click.option("--tags", default=None, help="Comma-separated tags.")
def add(
    project_path: str,
    name: str,
    prefab: str,
    points: str,
    align: str,
    tags: str | None,
) -> None:
    """Add a GPU-instanced object group."""
    pf = ProjectFile.load(project_path)
    pts = json.loads(points)
    tag_list = [t.strip() for t in tags.split(",")] if tags else None
    inst = pf.add_instance(name, prefab, pts, align=align, tags=tag_list)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "instance.add", "instance": inst}, indent=2
        )
    )


@instance_cmd.command("remove")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
def remove(project_path: str, name: str) -> None:
    """Remove an instance group by name."""
    pf = ProjectFile.load(project_path)
    removed = pf.remove_instance(name)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "instance.remove", "removed": removed}, indent=2
        )
    )


@instance_cmd.command("list")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def list_instances(project_path: str) -> None:
    """List all instance groups in the project."""
    pf = ProjectFile.load(project_path)
    instances = pf.data.get("instances", [])
    click.echo(
        json.dumps(
            {"status": "ok", "count": len(instances), "instances": instances}, indent=2
        )
    )
