"""Manifest commands."""

from __future__ import annotations

import json

import click

from blender_cli.cli.common import _output


def _load_manifest(path: str) -> dict:
    """Load manifest from a GLB (embedded extras) or a raw JSON file."""
    from pathlib import Path

    p = Path(path)
    if p.suffix.lower() == ".glb":
        import bpy

        from blender_cli.cli.common import _quiet
        from blender_cli.scene import Scene

        with _quiet():
            Scene.load(str(p))
        manifest_str = bpy.context.scene.get("_scene_manifest")
        if not manifest_str:
            msg = f"No manifest in GLB: {p}"
            raise click.ClickException(msg)
        return json.loads(manifest_str)
    return json.loads(p.read_text())


@click.group()
def manifest() -> None:
    """Manifest inspection and diffs."""


@manifest.command("diff")
@click.option("--a", "path_a", required=True, type=click.Path(exists=True))
@click.option("--b", "path_b", required=True, type=click.Path(exists=True))
@click.option("--json", "json_out", default=None, type=click.Path())
def manifest_diff(path_a: str, path_b: str, json_out: str | None) -> None:
    """Diff two scene manifests (GLBs or JSON files)."""
    a = _load_manifest(path_a)
    b = _load_manifest(path_b)

    a_sum = a.get("object_summary", {})
    b_sum = b.get("object_summary", {})
    a_anchors = {x.get("name"): x for x in a.get("anchors", [])}
    b_anchors = {x.get("name"): x for x in b.get("anchors", [])}
    a_assets = a.get("assets", {})
    b_assets = b.get("assets", {})

    diff = {
        "from": path_a,
        "to": path_b,
        "object_total": {
            "a": a_sum.get("total", 0),
            "b": b_sum.get("total", 0),
            "delta": b_sum.get("total", 0) - a_sum.get("total", 0),
        },
        "tags_delta": {
            k: b_sum.get("by_tag", {}).get(k, 0) - a_sum.get("by_tag", {}).get(k, 0)
            for k in sorted(set(a_sum.get("by_tag", {})) | set(b_sum.get("by_tag", {})))
        },
        "anchors_added": sorted(set(b_anchors) - set(a_anchors)),
        "anchors_removed": sorted(set(a_anchors) - set(b_anchors)),
        "assets": {
            "prefabs_added": sorted(
                {
                    (p.get("id") or p.get("path"))
                    for p in b_assets.get("prefabs", [])
                    if (p.get("id") or p.get("path"))
                }
                - {
                    (p.get("id") or p.get("path"))
                    for p in a_assets.get("prefabs", [])
                    if (p.get("id") or p.get("path"))
                }
            ),
            "materials_added": sorted(
                set(b_assets.get("materials", [])) - set(a_assets.get("materials", []))
            ),
        },
        "rng_streams_added": sorted(
            {s.get("stream") for s in b.get("rng_streams", [])}
            - {s.get("stream") for s in a.get("rng_streams", [])}
        ),
    }
    _output(diff, json_out)
