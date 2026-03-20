"""Interactive REPL command."""

from __future__ import annotations

import click


@click.command("repl")
@click.option(
    "--project",
    "project_path",
    default=None,
    type=click.Path(exists=True),
    help="Project file to open in REPL.",
)
@click.pass_context
def repl_cmd(ctx: click.Context, project_path: str | None) -> None:
    """Start an interactive REPL session."""
    from blender_cli.cli.repl import ReplSession

    # Prefer explicit --project, fall back to global --project
    proj = project_path or (ctx.obj or {}).get("project")
    root = ctx.find_root().command
    session = ReplSession(root, project_path=proj)
    session.loop()
