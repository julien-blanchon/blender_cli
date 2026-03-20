"""Object management CLI commands — duplicate, set-parent, set-visible."""

from __future__ import annotations

import json

import click


@click.group("object")
def object_cmd() -> None:
    """Object management — duplicate, parent, visibility."""


@object_cmd.command("duplicate")
@click.option("--project", required=True, type=click.Path(exists=True))
@click.argument("index_or_name")
def duplicate(project: str, index_or_name: str) -> None:
    """Deep-copy an object with a new UID and auto-incremented name."""
    from blender_cli.project import ProjectFile

    pf = ProjectFile.load(project)
    new_obj = pf.duplicate_object(index_or_name)
    pf.save()
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "duplicate",
                "result": {"name": new_obj["name"], "uid": new_obj["uid"]},
            },
            indent=2,
        )
    )


@object_cmd.command("set-parent")
@click.option("--project", required=True, type=click.Path(exists=True))
@click.argument("child")
@click.argument("parent")
def set_parent(project: str, child: str, parent: str) -> None:
    """Set parent-child relationship between objects."""
    from blender_cli.project import ProjectFile

    pf = ProjectFile.load(project)
    pf.set_parent(child, parent)
    pf.save()
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "set-parent",
                "result": {"child": child, "parent": parent},
            },
            indent=2,
        )
    )


@object_cmd.command("set-visible")
@click.option("--project", required=True, type=click.Path(exists=True))
@click.argument("index_or_name")
@click.argument("visible", type=click.BOOL)
def set_visible(project: str, index_or_name: str, visible: bool) -> None:
    """Toggle persistent visibility on an object."""
    from blender_cli.project import ProjectFile

    pf = ProjectFile.load(project)
    pf.set_visible(index_or_name, visible)
    pf.save()
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "set-visible",
                "result": {"object": index_or_name, "visible": visible},
            },
            indent=2,
        )
    )
