"""Stats command."""

from __future__ import annotations

import click

from blender_cli.cli.common import _output, _quiet, _scene_context
from blender_cli.scene import Scene


@click.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--json", "json_out", default=None, type=click.Path())
def stats(glb: str, json_out: str | None) -> None:
    """Output node/mesh/material/instance counts."""
    with _quiet():
        scene = Scene.load(glb)
        data: dict[str, object] = scene.stats()
        data["context"] = _scene_context(scene)
    _output(data, json_out)
