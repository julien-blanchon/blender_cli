"""Asset index/validation commands."""

from __future__ import annotations

import json
from pathlib import Path

import click


@click.group()
def assets() -> None:
    """Asset indexing/validation/info."""


@assets.command("index")
@click.option("--root", required=True, type=click.Path(exists=True))
@click.option("--out", required=True, type=click.Path())
def assets_index(root: str, out: str) -> None:
    """Index prefab models and PBR material folders under a root."""
    root_p = Path(root).resolve()
    model_files = sorted(
        p
        for p in root_p.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".glb", ".gltf"}
    )
    materials: dict[str, str] = {}
    for d in sorted(p for p in root_p.glob("**/*") if p.is_dir()):
        has_images = any(
            f.suffix.lower()
            in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".exr", ".hdr"}
            for f in d.iterdir()
            if f.is_file()
        )
        if has_images:
            materials[d.name] = str(d.resolve())
    models = {p.name: str(p.resolve()) for p in model_files}
    payload = {"models": models, "materials": materials}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "index": out,
                "models": len(models),
                "materials": len(materials),
            },
            indent=2,
        )
    )


@assets.command("validate")
@click.option("--index", "index_path", required=True, type=click.Path(exists=True))
def assets_validate(index_path: str) -> None:
    """Validate indexed paths exist and are accessible."""
    data = json.loads(Path(index_path).read_text(encoding="utf-8"))
    missing_models = [
        k for k, v in data.get("models", {}).items() if not Path(v).is_file()
    ]
    missing_materials = [
        k for k, v in data.get("materials", {}).items() if not Path(v).is_dir()
    ]
    click.echo(
        json.dumps(
            {
                "status": "ok"
                if not missing_models and not missing_materials
                else "warning",
                "missing_models": missing_models,
                "missing_materials": missing_materials,
            },
            indent=2,
        )
    )


@assets.command("info")
@click.option("--model", required=True, type=click.Path(exists=True))
def assets_info(model: str) -> None:
    """Show basic file info for a model asset."""
    p = Path(model).resolve()
    click.echo(
        json.dumps(
            {
                "path": str(p),
                "name": p.name,
                "size_bytes": p.stat().st_size,
                "suffix": p.suffix.lower(),
            },
            indent=2,
        )
    )
