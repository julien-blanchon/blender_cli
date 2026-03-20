"""Project management commands."""

from __future__ import annotations

import json
from typing import Any

import click

from blender_cli.project.project_file import PROFILES, ProjectFile, _PRIMITIVE_TYPES


@click.group()
def project() -> None:
    """Project file management — create, open, save, inspect."""


@project.command("new")
@click.option("--name", required=True, help="Project name.")
@click.option(
    "--profile",
    default="default",
    show_default=True,
    type=click.Choice(sorted(PROFILES)),
    help="Scene profile preset.",
)
@click.option(
    "-o",
    "--out",
    required=True,
    type=click.Path(),
    help="Output path for project JSON.",
)
def new(name: str, profile: str, out: str) -> None:
    """Create a new project file."""
    pf = ProjectFile.new(name, profile=profile)
    saved = pf.save(out)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "project.new",
                "path": str(saved),
                "profile": profile,
            },
            indent=2,
        )
    )


@project.command("open")
@click.argument("path", type=click.Path(exists=True))
def open_project(path: str) -> None:
    """Load a project and print summary."""
    pf = ProjectFile.load(path)
    click.echo(json.dumps(pf.summary(), indent=2))


@project.command("save")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option(
    "-o", "--out", default=None, type=click.Path(), help="Save to a different path."
)
def save(project_path: str, out: str | None) -> None:
    """Save project (optionally to a new path)."""
    pf = ProjectFile.load(project_path)
    saved = pf.save(out or project_path)
    click.echo(
        json.dumps(
            {"status": "ok", "action": "project.save", "path": str(saved)}, indent=2
        )
    )


@project.command("info")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def info(project_path: str) -> None:
    """Print project summary."""
    pf = ProjectFile.load(project_path)
    click.echo(json.dumps(pf.summary(), indent=2))


@project.command("profiles")
def profiles() -> None:
    """List all available scene profiles."""
    result = {}
    for name, cfg in sorted(PROFILES.items()):
        r = cfg["render"]
        result[name] = {
            "engine": r["engine"],
            "resolution": r["resolution"],
            "samples": r["samples"],
            "denoising": r["denoising"],
            "film_transparent": r["film_transparent"],
            "fps": cfg["scene"]["fps"],
        }
    click.echo(json.dumps(result, indent=2))


@project.command("describe")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
def describe(project_path: str) -> None:
    """Full scene description — objects, anchors, spatial extent, tags."""
    pf = ProjectFile.load(project_path)
    click.echo(json.dumps(pf.describe(), indent=2))


@project.command("export")
@click.option(
    "--project",
    "project_path",
    required=True,
    type=click.Path(exists=True),
    help="Project file.",
)
@click.option("--out", required=True, type=click.Path(), help="Output GLB path.")
def export_glb(project_path: str, out: str) -> None:
    """Export project to GLB."""
    pf = ProjectFile.load(project_path)
    result = pf.export_glb(out)
    click.echo(
        json.dumps(
            {"status": "ok", "action": "project.export", "path": str(result)}, indent=2
        )
    )


@project.command("import")
@click.option(
    "--glb", required=True, type=click.Path(exists=True), help="GLB file to import."
)
@click.option("--name", default=None, help="Project name (defaults to GLB filename).")
@click.option(
    "-o", "--out", required=True, type=click.Path(), help="Output project JSON path."
)
def import_glb(glb: str, name: str | None, out: str) -> None:
    """Import a GLB file into a new project."""
    pf = ProjectFile.import_glb(glb, name=name)
    saved = pf.save(out)
    click.echo(
        json.dumps(
            {"status": "ok", "action": "project.import", "path": str(saved)}, indent=2
        )
    )


@project.command("add-object")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--primitive", default=None, type=click.Choice(sorted(_PRIMITIVE_TYPES)), help="Primitive type.")
@click.option("--primitive-params", default=None, help='JSON params for primitive: {"size": [1,1,1]}')
@click.option("--asset-path", default=None, help="Path to GLB prefab.")
@click.option("--at", "location", nargs=3, type=float, default=None, help="Position x y z.")
@click.option("--at-anchor", default=None, help="Position at named anchor.")
@click.option("--offset", nargs=3, type=float, default=None, help="Offset from anchor x y z.")
@click.option("--rotation", nargs=3, type=float, default=None, help="Rotation x y z (radians).")
@click.option("--scale", nargs=3, type=float, default=None, help="Scale x y z.")
@click.option("--material", default=None, help="Material name.")
@click.option("--tags", default=None, help="Comma-separated tags.")
@click.option("--snap", "snap_axis", default=None,
              help="Snap axis at export (-Z, +Z, -X, +X, -Y, +Y). Default: -Z.")
@click.option("--snap-policy", default=None,
              type=click.Choice(["FIRST", "LAST", "HIGHEST", "LOWEST", "AVERAGE", "ORIENT"]),
              help="Snap policy.")
@click.option("--snap-exclude", default=None, help="Comma-separated tags to exclude from snap targets.")
@click.option("--snap-target", default=None, help="Comma-separated tags to snap onto (only these).")
def add_object(
    project_path: str,
    name: str,
    primitive: str | None,
    primitive_params: str | None,
    asset_path: str | None,
    location: tuple[float, float, float] | None,
    at_anchor: str | None,
    offset: tuple[float, float, float] | None,
    rotation: tuple[float, float, float] | None,
    scale: tuple[float, float, float] | None,
    material: str | None,
    tags: str | None,
    snap_axis: str | None,
    snap_policy: str | None,
    snap_exclude: str | None,
    snap_target: str | None,
) -> None:
    """Add an object to the project."""
    pf = ProjectFile.load(project_path)
    prim_dict = None
    if primitive:
        prim_dict = {"type": primitive}
        if primitive_params:
            prim_dict.update(json.loads(primitive_params))
    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    # Resolve position: --at takes precedence, then --at-anchor + --offset
    loc: list[float] | None = None
    if location:
        loc = list(location)
    elif at_anchor:
        off = list(offset) if offset else None
        loc = pf.anchor_pos(at_anchor, off)

    scl = list(scale) if scale else None
    # Build snap spec
    snap_spec: dict[str, Any] | None = None
    if snap_axis:
        snap_spec = {"axis": snap_axis}
        if snap_policy:
            snap_spec["policy"] = snap_policy
        if snap_exclude:
            snap_spec["exclude_tags"] = [t.strip() for t in snap_exclude.split(",")]
        if snap_target:
            snap_spec["target_tags"] = [t.strip() for t in snap_target.split(",")]

    # Compute warnings before add (for JSON response)
    warnings = pf.placement_warnings(name, loc or [0, 0, 0], prim_dict, scl)
    nearby = pf.nearby_objects(loc or [0, 0, 0], radius=20.0, limit=5)
    obj = pf.add_object(
        name,
        primitive=prim_dict,
        asset_path=asset_path,
        location=loc,
        rotation=list(rotation) if rotation else None,
        scale=scl,
        material=material,
        tags=tag_list,
        snap=snap_spec,
    )
    pf.save()
    output: dict[str, Any] = {
        "status": "ok",
        "action": "project.add-object",
        "object": obj,
        "nearby": nearby,
    }
    if warnings:
        output["warnings"] = warnings
    click.echo(json.dumps(output, indent=2))


@project.command("add-camera")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--at", "location", nargs=3, type=float, default=None, help="Position x y z.")
@click.option("--at-anchor", default=None, help="Position at named anchor.")
@click.option("--offset", nargs=3, type=float, default=None, help="Offset from anchor x y z.")
@click.option("--look-at", nargs=3, type=float, default=None, help="Point camera at x y z.")
@click.option("--look-at-anchor", default=None, help="Point camera at named anchor.")
@click.option("--rotation", nargs=3, type=float, default=None, help="Rotation x y z (radians).")
@click.option("--lens", type=float, default=50.0, help="Focal length.")
@click.option("--path-def", default=None, help='Camera path JSON: {"type": "orbit", ...}')
@click.option("--ghost", is_flag=True, default=False, help="Ghost camera (not exported to GLB).")
def add_camera(
    project_path: str,
    name: str,
    location: tuple[float, float, float] | None,
    at_anchor: str | None,
    offset: tuple[float, float, float] | None,
    look_at: tuple[float, float, float] | None,
    look_at_anchor: str | None,
    rotation: tuple[float, float, float] | None,
    lens: float,
    path_def: str | None,
    ghost: bool,
) -> None:
    """Add a camera to the project."""
    pf = ProjectFile.load(project_path)

    # Resolve position
    loc: list[float] | None = None
    if location:
        loc = list(location)
    elif at_anchor:
        off = list(offset) if offset else None
        loc = pf.anchor_pos(at_anchor, off)

    # Resolve look_at
    look: list[float] | None = None
    if look_at:
        look = list(look_at)
    elif look_at_anchor:
        look = pf.anchor_pos(look_at_anchor)

    path = json.loads(path_def) if path_def else None
    cam = pf.add_camera(
        name,
        location=loc,
        rotation=list(rotation) if rotation else None,
        look_at=look,
        lens=lens,
        path=path,
        ghost=ghost,
    )
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "project.add-camera", "camera": cam}, indent=2))


@project.command("add-light")
@click.option("--project", "project_path", required=True, type=click.Path(exists=True))
@click.argument("name")
@click.option("--type", "light_type", default="SUN", type=click.Choice(["SUN", "POINT", "SPOT", "AREA"]))
@click.option("--at", "location", nargs=3, type=float, default=None, help="Position x y z.")
@click.option("--rotation", nargs=3, type=float, default=None, help="Rotation x y z (radians).")
@click.option("--energy", type=float, default=1.0, help="Light intensity.")
@click.option("--color", nargs=3, type=float, default=None, help="Light color RGB (0..1).")
def add_light(
    project_path: str,
    name: str,
    light_type: str,
    location: tuple[float, float, float] | None,
    rotation: tuple[float, float, float] | None,
    energy: float,
    color: tuple[float, float, float] | None,
) -> None:
    """Add a light to the project."""
    pf = ProjectFile.load(project_path)
    light = pf.add_light(
        name,
        light_type,
        location=list(location) if location else None,
        rotation=list(rotation) if rotation else None,
        energy=energy,
        color=list(color) if color else None,
    )
    pf.save()
    click.echo(json.dumps({"status": "ok", "action": "project.add-light", "light": light}, indent=2))
