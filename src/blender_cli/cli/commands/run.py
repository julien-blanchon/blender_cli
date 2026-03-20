"""Run command — execute generation scripts via BuildContext."""

from __future__ import annotations

import json

import click

from blender_cli.cli.common import _quiet


@click.command()
@click.argument("script_path", type=click.Path(exists=True))
@click.option("--seed", default=0, type=int, show_default=True, help="RNG seed.")
def run(script_path: str, seed: int) -> None:
    """Run a generation script's run(ctx) function."""
    from pathlib import Path

    with _quiet():
        from blender_cli.build.runner import run_script

        run_script(script_path, seed=seed)

    # Determine output path for feedback
    script = Path(script_path).resolve()
    out_dir = script.parent / "output"
    feedback: dict[str, object] = {
        "status": "ok",
        "action": "run",
        "script": str(script),
        "seed": seed,
        "output_dir": str(out_dir) if out_dir.exists() else None,
    }
    click.echo(json.dumps(feedback, indent=2))
