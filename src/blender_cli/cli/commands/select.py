"""Selection command."""

from __future__ import annotations

import click

from blender_cli.cli.common import _output, _quiet, _resolve_where, _scene_context
from blender_cli.scene import Scene


@click.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--json", "json_out", default=None, type=click.Path())
def select(glb: str, where: str, json_out: str | None) -> None:
    """Return UIDs of matching entities."""
    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        data: dict[str, object] = {"uids": sel.uids(), "count": sel.count()}
        data["context"] = _scene_context(scene)
    _output(data, json_out)
