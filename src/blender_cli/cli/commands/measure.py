"""Measurement commands."""

from __future__ import annotations

import click

from blender_cli.cli.common import (
    _output,
    _quiet,
    _ref_position,
    _scene_context,
)
from blender_cli.scene import Scene


@click.group()
def measure() -> None:
    """Measurement commands."""


@measure.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--a", "ref_a", required=True, help="Ref for point A (uid:, name:, ann:, tag:)."
)
@click.option("--b", "ref_b", required=True, help="Ref for point B.")
@click.option("--json", "json_out", default=None, type=click.Path())
def distance(glb: str, ref_a: str, ref_b: str, json_out: str | None) -> None:
    """Measure Euclidean distance between two entity refs."""
    with _quiet():
        scene = Scene.load(glb)
        pa = _ref_position(scene, ref_a)
        pb = _ref_position(scene, ref_b)
        ctx = _scene_context(scene)
    d = pa.distance(pb)
    _output(
        {
            "a": {
                "ref": ref_a,
                "position": {
                    "x": round(pa.x, 4),
                    "y": round(pa.y, 4),
                    "z": round(pa.z, 4),
                },
            },
            "b": {
                "ref": ref_b,
                "position": {
                    "x": round(pb.x, 4),
                    "y": round(pb.y, 4),
                    "z": round(pb.z, 4),
                },
            },
            "distance": round(d, 4),
            "context": ctx,
        },
        json_out,
    )
