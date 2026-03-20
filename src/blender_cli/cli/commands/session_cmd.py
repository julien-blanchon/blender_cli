"""Session management CLI commands — undo, redo, history, status."""

from __future__ import annotations

import json

import click

from blender_cli.cli.common import _cli_json_errors
from blender_cli.project.project_file import ProjectFile
from blender_cli.project.session import Session


@click.group()
def session() -> None:
    """Session management — undo, redo, and history tracking."""


@session.command("status")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def status(project_path: str) -> None:
    """Print session status."""
    pf = ProjectFile.load(project_path)
    s = Session(pf)
    click.echo(json.dumps(s.status(), indent=2))


@session.command("undo")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def undo(project_path: str) -> None:
    """Undo the last operation."""
    pf = ProjectFile.load(project_path)
    s = Session(pf)
    with _cli_json_errors():
        desc = s.undo()
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "session.undo", "undone": desc}, indent=2
        )
    )


@session.command("redo")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def redo(project_path: str) -> None:
    """Redo the last undone operation."""
    pf = ProjectFile.load(project_path)
    s = Session(pf)
    with _cli_json_errors():
        desc = s.redo()
    pf.save()
    click.echo(
        json.dumps(
            {"status": "ok", "action": "session.redo", "redone": desc}, indent=2
        )
    )


@session.command("history")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def history(project_path: str) -> None:
    """List undo history."""
    pf = ProjectFile.load(project_path)
    s = Session(pf)
    click.echo(json.dumps(s.list_history(), indent=2))
