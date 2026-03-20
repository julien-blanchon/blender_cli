"""blender_cli entrypoint and command registration."""

from __future__ import annotations

import json
import sys

import click

from blender_cli.cli.commands import register_all


class _JsonErrorGroup(click.Group):
    """Click group that catches exceptions and formats them as JSON when --json is active."""

    def invoke(self, ctx: click.Context) -> None:
        try:
            super().invoke(ctx)
        except (click.ClickException, click.UsageError) as exc:
            ctx.ensure_object(dict)
            if ctx.obj.get("json"):
                msg = (
                    exc.format_message() if hasattr(exc, "format_message") else str(exc)
                )
                click.echo(
                    json.dumps(
                        {"status": "error", "error": msg, "type": type(exc).__name__},
                        indent=2,
                    ),
                    err=True,
                )
                sys.exit(getattr(exc, "exit_code", 1))
            raise
        except Exception as exc:
            ctx.ensure_object(dict)
            if ctx.obj.get("json"):
                click.echo(
                    json.dumps(
                        {
                            "status": "error",
                            "error": str(exc),
                            "type": type(exc).__name__,
                        },
                        indent=2,
                    ),
                    err=True,
                )
                sys.exit(1)
            raise


@click.group(cls=_JsonErrorGroup)
@click.option(
    "--project",
    default=None,
    type=click.Path(),
    help="Project JSON file (alternative to per-command --glb).",
)
@click.option(
    "--json",
    "json_mode",
    is_flag=True,
    default=False,
    help="Output JSON to stdout for all commands.",
)
@click.pass_context
def main(ctx: click.Context, project: str | None, json_mode: bool) -> None:
    """Blender CLI — Map SDK CLI for scene inspection and debug."""
    ctx.ensure_object(dict)
    ctx.obj["project"] = project
    ctx.obj["json"] = json_mode


register_all(main)
