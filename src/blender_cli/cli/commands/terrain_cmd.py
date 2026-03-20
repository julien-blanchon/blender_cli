"""Terrain management commands — init, add operations, configure mesh, info."""

from __future__ import annotations

import json

import click

from blender_cli.project.project_file import ProjectFile, _TERRAIN_OPS


@click.group("terrain")
def terrain_cmd() -> None:
    """Terrain recipe management — procedural terrain pipeline stored in project.json."""


@terrain_cmd.command("init")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.option("--width", required=True, type=int, help="Terrain width in pixels.")
@click.option("--height", required=True, type=int, help="Terrain height in pixels.")
@click.option("--meters-per-px", default=1.0, type=float, help="World metres per pixel.")
@click.option("--seed", default=None, type=int, help="Global terrain seed.")
def init(project_path: str, width: int, height: int, meters_per_px: float, seed: int | None) -> None:
    """Initialise terrain recipe in the project."""
    pf = ProjectFile.load(project_path)
    t = pf.set_terrain(width, height, meters_per_px, seed=seed)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "terrain.init", "terrain": t}, indent=2))


@terrain_cmd.command("op")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("op_type", type=click.Choice(sorted(_TERRAIN_OPS)))
@click.argument("params", nargs=-1)
def add_op(project_path: str, op_type: str, params: tuple[str, ...]) -> None:
    """Append a terrain operation. Params as key=value pairs.

    Example: blender_cli terrain op --project p.json noise type=fbm amp=15 freq=0.008
    """
    pf = ProjectFile.load(project_path)
    kwargs: dict = {}
    for p in params:
        if "=" not in p:
            msg = f"Invalid param {p!r} — expected key=value"
            raise click.BadParameter(msg)
        k, v = p.split("=", 1)
        # Auto-convert: JSON arrays/objects, then int, then float, then string
        if v.startswith("[") or v.startswith("{"):
            try:
                kwargs[k] = json.loads(v)
                continue
            except json.JSONDecodeError:
                pass
        try:
            kwargs[k] = int(v)
        except ValueError:
            try:
                kwargs[k] = float(v)
            except ValueError:
                kwargs[k] = v
    entry = pf.terrain_op(op_type, **kwargs)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "terrain.op", "operation": entry}, indent=2))


@terrain_cmd.command("mesh")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.option("--lod", type=int, help="Level of detail (decimation factor).")
@click.option("--skirts", type=float, help="Skirt depth around edges.")
@click.option("--tile-scale", type=float, help="UV tile scale in metres.")
def mesh(project_path: str, lod: int | None, skirts: float | None, tile_scale: float | None) -> None:
    """Configure terrain mesh generation parameters."""
    pf = ProjectFile.load(project_path)
    params = {}
    if lod is not None:
        params["lod"] = lod
    if skirts is not None:
        params["skirts"] = skirts
    if tile_scale is not None:
        params["tile_scale"] = tile_scale
    pf.set_terrain_mesh(**params)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "terrain.mesh", "params": params}, indent=2))


@terrain_cmd.command("material")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
def set_material(project_path: str, name: str) -> None:
    """Assign a named material to the terrain."""
    pf = ProjectFile.load(project_path)
    pf.set_terrain_material(name)
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "terrain.material", "material": name}, indent=2))


@terrain_cmd.command("clear")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def clear(project_path: str) -> None:
    """Remove terrain recipe from the project."""
    pf = ProjectFile.load(project_path)
    pf.clear_terrain()
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "terrain.clear"}, indent=2))


@terrain_cmd.command("info")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
def info(project_path: str) -> None:
    """Show terrain recipe details."""
    pf = ProjectFile.load(project_path)
    terrain = pf.data.get("terrain")
    if terrain is None:
        click.echo(json.dumps({"status": "ok", "terrain": None}, indent=2))
    else:
        click.echo(json.dumps({"status": "ok", "terrain": terrain}, indent=2))
