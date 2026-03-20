"""Modifier CLI commands — list-available, info, add, remove, set, list."""

from __future__ import annotations

import json
from typing import Any

import click

from blender_cli.cli.common import _cli_json_errors
from blender_cli.modifiers.modifier import Modifier
from blender_cli.modifiers.registry import ModifierRegistry
from blender_cli.project.project_file import ProjectFile


def _parse_param_value(raw: str) -> int | float | bool | str:
    """Parse a CLI string into a typed Python value."""
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


@click.group("modifier")
def modifier() -> None:
    """Modifier system — add, remove, configure Blender modifiers on objects."""


@modifier.command("list-available")
@click.option(
    "--category",
    type=click.Choice(["generate", "deform"]),
    default=None,
    help="Filter by category.",
)
def list_available(category: str | None) -> None:
    """List all available modifier types."""
    mods = ModifierRegistry.available(category)
    result = [
        {
            "type": m["type"],
            "name": m["name"],
            "category": m["category"],
            "description": m["description"],
        }
        for m in mods
    ]
    click.echo(json.dumps(result, indent=2))


@modifier.command("info")
@click.argument("name")
def info(name: str) -> None:
    """Show full parameter specs for a modifier type."""
    with _cli_json_errors():
        data = ModifierRegistry.info(name)
    click.echo(json.dumps(data, indent=2))


@modifier.command("add")
@click.argument("mod_type")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--object", "object_ref", required=True, help="Object index or name.")
@click.option(
    "-p", "params_raw", multiple=True, help="Parameter as key=value (repeatable)."
)
def add(
    mod_type: str, project_path: str, object_ref: str, params_raw: tuple[str, ...]
) -> None:
    """Add a modifier to an object."""
    params: dict[str, Any] = {}
    for kv in params_raw:
        if "=" not in kv:
            click.echo(
                json.dumps(
                    {
                        "status": "error",
                        "error": f"Invalid param format: {kv!r}. Use key=value",
                    },
                    indent=2,
                ),
                err=True,
            )
            raise SystemExit(1)
        k, v = kv.split("=", 1)
        params[k] = _parse_param_value(v)

    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        result = Modifier.add(pf.data, object_ref, mod_type, params)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "modifier.add", "modifier": result}, indent=2
        )
    )


@modifier.command("remove")
@click.argument("index", type=int)
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--object", "object_ref", required=True, help="Object index or name.")
def remove(index: int, project_path: str, object_ref: str) -> None:
    """Remove a modifier by index."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        removed = Modifier.remove(pf.data, object_ref, index)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "modifier.remove", "removed": removed}, indent=2
        )
    )


@modifier.command("set")
@click.argument("index", type=int)
@click.argument("param")
@click.argument("value")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--object", "object_ref", required=True, help="Object index or name.")
def set_param(
    index: int, param: str, value: str, project_path: str, object_ref: str
) -> None:
    """Update a modifier parameter."""
    pf = ProjectFile.load(project_path)
    typed_value = _parse_param_value(value)
    with _cli_json_errors():
        updated = Modifier.set(pf.data, object_ref, index, param, typed_value)
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "modifier.set", "modifier": updated}, indent=2
        )
    )


@modifier.command("list")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--object", "object_ref", required=True, help="Object index or name.")
def list_modifiers(project_path: str, object_ref: str) -> None:
    """List modifiers on an object."""
    pf = ProjectFile.load(project_path)
    with _cli_json_errors():
        mods = Modifier.list(pf.data, object_ref)
    click.echo(json.dumps(mods, indent=2))
