"""Reproducible edit operations (codegen)."""

from __future__ import annotations

import json
from pathlib import Path

import click

from blender_cli.cli.common import (
    _check_placement_warnings,
    _codegen_select,
    _material_codegen,
    _nearby_at_position,
    _parse_kv_pairs,
    _parse_point,
    _quiet,
    _resolve_anchor_or_pos,
    _resolve_where,
    _scene_context,
    _write_script,
)
from blender_cli.scene import Scene
from blender_cli.snap import AXIS_DIR
from blender_cli.types import Vec3

# ---------------------------------------------------------------------------
# Shared codegen helper for spline + heightfield loading
# ---------------------------------------------------------------------------


def _spline_hf_codegen(
    heightmap_abs: str,
    meters_per_px: float,
    remap_min: float,
    remap_max: float,
    control_points: list[tuple[float, float, float]],
    resample_step: float,
    closed: bool,
    snap_spline: bool,
) -> str:
    """Return common code block for heightfield + spline setup."""
    pts_code = ", ".join(f"Vec3({p[0]}, {p[1]}, {p[2]})" for p in control_points)
    closed_arg = ", closed=True" if closed else ""
    snap_code = "    spline = spline.snap(scene)\n" if snap_spline else ""
    return (
        f"    field = Field2D.from_image({heightmap_abs!r}, meters_per_px={meters_per_px})\n"
        f"    field = field.remap({remap_min}, {remap_max})\n"
        f"    hf = Heightfield(field)\n"
        f"    spline = Spline.catmull([{pts_code}]{closed_arg})\n"
        f"    spline = spline.resample({resample_step})\n"
        f"{snap_code}"
    )


@click.group()
def op() -> None:
    """Reproducible edit operations — generate Python build steps."""


@op.command("add_prefab")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--prefab", required=True, type=click.Path(exists=True))
@click.option("--at", default=None, nargs=3, type=float, help="Position x y z.")
@click.option("--anchor", default=None, help="Anchor name (alternative to --at).")
@click.option(
    "--scale", "scale_factor", default=None, type=float, help="Uniform scale factor."
)
@click.option(
    "--rescale-fit",
    "rescale_fit",
    default=None,
    type=float,
    help="Target metric size (metres) for the longest bbox dimension. "
    "Auto-scales the prefab so its longest axis equals this value. "
    "Mutually exclusive with --scale.",
)
@click.option(
    "--yaw", "yaw_deg", default=None, type=float, help="Z-rotation in degrees."
)
@click.option(
    "--name", "entity_name", default=None, type=str, help="Entity display name."
)
@click.option("--snap", "snap_axis", default=None, type=click.Choice(sorted(AXIS_DIR)))
@click.option(
    "--snap-policy",
    default="FIRST",
    type=click.Choice(["FIRST", "LAST", "HIGHEST", "LOWEST", "AVERAGE", "ORIENT"]),
    show_default=True,
)
@click.option(
    "--ignore", "ignore_tags", multiple=True, help="Tags to ignore during snap."
)
@click.option(
    "--snap-to-where", default=None, help="DSL expression for snap target selection."
)
@click.option("--tag", "tags", multiple=True, help="Tag(s) to attach to new object.")
@click.option("--prop", "props_kv", multiple=True, help="Prop key=value (repeatable).")
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def add_prefab(
    glb: str,
    prefab: str,
    at: tuple[float, float, float] | None,
    anchor: str | None,
    scale_factor: float | None,
    rescale_fit: float | None,
    yaw_deg: float | None,
    entity_name: str | None,
    snap_axis: str | None,
    snap_policy: str,
    ignore_tags: tuple[str, ...],
    snap_to_where: str | None,
    tags: tuple[str, ...],
    props_kv: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that adds a prefab at a position."""
    if at is None and anchor is None:
        msg = "Provide either --at x y z or --anchor NAME."
        raise click.UsageError(msg)
    if at is not None and anchor is not None:
        msg = "--at and --anchor are mutually exclusive."
        raise click.UsageError(msg)
    if scale_factor is not None and rescale_fit is not None:
        msg = "--scale and --rescale-fit are mutually exclusive."
        raise click.UsageError(msg)

    prefab_abs = str(Path(prefab).resolve())

    snapped = False
    snap_surface: str | None = None
    warn_list: list[str] = []
    props: dict[str, str] = {}
    for kv in props_kv:
        if "=" not in kv:
            warn_list.append(f"Ignored invalid --prop value: {kv!r}")
            continue
        k, v = kv.split("=", 1)
        props[k.strip()] = v.strip()

    with _quiet():
        scene = Scene.load(glb)
        pos = _resolve_anchor_or_pos(scene, anchor, at)
        x, y, z = pos.x, pos.y, pos.z
        final_pos = pos
        if snap_axis:
            from blender_cli.snap import snap as snap_fn

            if snap_to_where:
                snap_scene = scene.snap_targets(snap_to_where)
            elif ignore_tags:
                snap_scene = scene.ignore(tags=set(ignore_tags))
            else:
                snap_scene = scene

            results = snap_fn([Vec3(x, y, z)], snap_scene, snap_axis)
            hit = results[0]
            if hit.hit:
                final_pos = hit.hit_pos
                snapped = True
                snap_surface = hit.hit_uid
            else:
                warn_list.append(f"Snap miss at ({x}, {y}, {z}) axis={snap_axis}")
            warn_list.extend(results.summary.warnings)
        nearby = _nearby_at_position(scene, final_pos)
        warn_list.extend(
            _check_placement_warnings(
                scene,
                final_pos,
                rescale_fit=rescale_fit,
                scale_factor=scale_factor,
            )
        )

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "add_prefab",
        "result": {
            "uid": None,
            "position": {
                "x": round(final_pos.x, 4),
                "y": round(final_pos.y, 4),
                "z": round(final_pos.z, 4),
            },
            "snapped": snapped,
            "snap_surface": snap_surface,
            "nearby": nearby,
            "tags": sorted(tags),
            "props": props,
            "scale": scale_factor,
            "rescale_fit": rescale_fit,
            "yaw": yaw_deg,
            "name": entity_name,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    stem = Path(prefab).stem
    display_name = entity_name or stem

    # Build codegen chain: .at().rescale_fit().scale().yaw().snap().named().tag().props()
    normalize_code = f".rescale_fit({rescale_fit})" if rescale_fit is not None else ""
    scale_code = f".scale({scale_factor})" if scale_factor is not None else ""
    yaw_code = f".yaw({yaw_deg})" if yaw_deg is not None else ""
    named_code = f".named({display_name!r})" if entity_name else ""
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""
    props_code = (
        f".props({', '.join(f'{k}={v!r}' for k, v in props.items())})" if props else ""
    )
    snap_import = ""
    snap_code = ""
    if snap_axis:
        snap_import = "from blender_cli.snap import SnapPolicy\n"
        to_part = (
            f", to_where={snap_to_where!r}"
            if snap_to_where
            else (
                f", to=scene.ignore(tags={set(ignore_tags)!r})" if ignore_tags else ""
            )
        )
        snap_code = f".snap(scene, axis={snap_axis!r}, policy=SnapPolicy.{snap_policy}{to_part})"

    # Position expression: use anchor expression or literal Vec3
    if anchor:
        at_expr = f"scene.anchor({anchor!r}).location()"
    else:
        at_expr = f"Vec3({x}, {y}, {z})"

    chain = f".at({at_expr}){normalize_code}{scale_code}{yaw_code}{snap_code}{named_code}{tag_code}{props_code}"

    code = (
        f'"""Build step: add prefab."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.assets import Prefab\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"{snap_import}\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    prefab = Prefab(Path({prefab_abs!r}))\n"
        f"    ent = prefab.spawn({display_name!r}){chain}\n"
        f"    scene.add(ent)\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


@op.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option(
    "--delta", nargs=3, type=float, default=None, help="Relative translation dx dy dz."
)
@click.option(
    "--to",
    "to_pos",
    nargs=3,
    type=float,
    default=None,
    help="Absolute world position x y z.",
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def move(
    glb: str,
    where: str,
    delta: tuple[float, float, float] | None,
    to_pos: tuple[float, float, float] | None,
    write_step: str,
    dry_run: bool,
) -> None:
    """
    Generate a build step that moves matching entities.

    Use --delta for relative translation or --to for absolute positioning.
    Exactly one of --delta or --to must be provided.
    """
    if delta is None and to_pos is None:
        msg = "Provide either --delta dx dy dz or --to x y z."
        raise click.UsageError(msg)
    if delta is not None and to_pos is not None:
        msg = "--delta and --to are mutually exclusive."
        raise click.UsageError(msg)

    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        count = sel.count()
        uids = sel.uids()

    if delta is not None:
        dx, dy, dz = delta
        feedback: dict[str, object] = {
            "status": "ok",
            "action": "move",
            "result": {
                "match_count": count,
                "uids": uids,
                "delta": {"dx": dx, "dy": dy, "dz": dz},
            },
            "warnings": [],
        }
    else:
        assert to_pos is not None
        tx, ty, tz = to_pos
        feedback = {
            "status": "ok",
            "action": "move_to",
            "result": {
                "match_count": count,
                "uids": uids,
                "to": {"x": tx, "y": ty, "z": tz},
            },
            "warnings": [],
        }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    select_expr, extra_import = _codegen_select(where)

    if delta is not None:
        dx, dy, dz = delta
        code = (
            f'"""Build step: move entities by ({dx}, {dy}, {dz})."""\n'
            f"{extra_import}"
            f"from blender_cli.build import BuildContext\n"
            f"from blender_cli.scene import Scene\n\n\n"
            f"def run(ctx: BuildContext) -> None:\n"
            f"    scene = Scene(ctx.open_prev())\n"
            f"    sel = {select_expr}\n"
            f"    scene.transform(sel).move({dx}, {dy}, {dz})\n"
            f"    scene.save(ctx.out_path())\n"
        )
    else:
        assert to_pos is not None
        tx, ty, tz = to_pos
        code = (
            f'"""Build step: move entities to ({tx}, {ty}, {tz})."""\n'
            f"{extra_import}"
            f"from blender_cli.build import BuildContext\n"
            f"from blender_cli.scene import Scene\n"
            f"from blender_cli.scene.entity import Entity\n\n\n"
            f"def run(ctx: BuildContext) -> None:\n"
            f"    scene = Scene(ctx.open_prev())\n"
            f"    sel = {select_expr}\n"
            f"    for obj in sel:\n"
            f"        Entity(obj).at({tx}, {ty}, {tz})\n"
            f"    scene.save(ctx.out_path())\n"
        )

    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


@op.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def delete(
    glb: str,
    where: str,
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that deletes matching entities."""
    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        count = sel.count()
        uids = sel.uids()

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "delete",
        "result": {
            "match_count": count,
            "uids": uids,
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    select_expr, extra_import = _codegen_select(where)
    code = (
        f'"""Build step: delete entities matching {where!r}."""\n'
        f"{extra_import}"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene\n\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    sel = {select_expr}\n"
        f"    scene.delete(sel)\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


@op.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--prefab", required=True, type=click.Path(exists=True))
@click.option(
    "--density", required=True, type=click.Path(exists=True), help="Density image path."
)
@click.option(
    "--density-value",
    default=1.0,
    type=float,
    show_default=True,
    help="Poisson density.",
)
@click.option(
    "--seed", default=None, type=int, help="RNG seed (default: context-based)."
)
@click.option(
    "--scale-min", default=1.0, type=float, show_default=True, help="Min random scale."
)
@click.option(
    "--scale-max", default=1.0, type=float, show_default=True, help="Max random scale."
)
@click.option(
    "--yaw-min",
    default=0.0,
    type=float,
    show_default=True,
    help="Min random yaw (degrees).",
)
@click.option(
    "--yaw-max",
    default=360.0,
    type=float,
    show_default=True,
    help="Max random yaw (degrees).",
)
@click.option(
    "--tag", "tags", multiple=True, help="Tag(s) to attach to scattered instances."
)
@click.option(
    "--name", "entity_name", default=None, type=str, help="Instance group name."
)
@click.option(
    "--align",
    default="center",
    type=click.Choice(["center", "bottom", "top"]),
    show_default=True,
    help="Vertical alignment of instances.",
)
@click.option(
    "--snap",
    "snap_axis",
    default="-Z",
    type=click.Choice(sorted(AXIS_DIR)),
    show_default=True,
)
@click.option(
    "--snap-policy",
    default="FIRST",
    type=click.Choice(["FIRST", "LAST", "HIGHEST", "LOWEST", "AVERAGE", "ORIENT"]),
    show_default=True,
)
@click.option(
    "--ignore", "ignore_tags", multiple=True, help="Tags to ignore during snap."
)
@click.option(
    "--snap-to-where", default=None, help="DSL expression for snap target selection."
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def scatter(
    glb: str,
    prefab: str,
    density: str,
    density_value: float,
    seed: int | None,
    scale_min: float,
    scale_max: float,
    yaw_min: float,
    yaw_max: float,
    tags: tuple[str, ...],
    entity_name: str | None,
    align: str,
    snap_axis: str,
    snap_policy: str,
    ignore_tags: tuple[str, ...],
    snap_to_where: str | None,
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that scatters prefab instances using a density map."""
    prefab_abs = str(Path(prefab).resolve())
    density_abs = str(Path(density).resolve())

    with _quiet():
        from blender_cli.geometry import Mask, PointSet

        mask = Mask.from_image(density_abs)
        preview_seed = seed if seed is not None else 42
        pts = PointSet.poisson(mask, density=density_value, seed=preview_seed)

    stem = Path(prefab).stem
    display_name = entity_name or stem
    tag_set = set(tags) if tags else {"scattered"}

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "scatter",
        "result": {
            "point_count": pts.count,
            "prefab": prefab_abs,
            "density_value": density_value,
            "seed": seed,
            "scale_range": [scale_min, scale_max],
            "yaw_range": [yaw_min, yaw_max],
            "tags": sorted(tag_set),
            "name": display_name,
            "align": align,
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    # Seed expression: explicit or context-based
    seed_expr = str(seed) if seed is not None else 'ctx.rng("scatter").int()'
    rand_seed_expr = (
        str(seed + 1) if seed is not None else 'ctx.rng("scatter_rand").int()'
    )

    # Randomize args
    randomize_args: list[str] = [f"seed={rand_seed_expr}"]
    if scale_min != 1.0 or scale_max != 1.0:
        randomize_args.append(f"scale=({scale_min}, {scale_max})")
    if yaw_min != 0.0 or yaw_max != 360.0:
        randomize_args.append(f"yaw=({yaw_min}, {yaw_max})")
    randomize_call = f"    pts = pts.randomize({', '.join(randomize_args)})\n"

    # Snap codegen
    snap_import = ""
    snap_code = f"    pts = pts.snap(scene, axis={snap_axis!r})\n"
    if snap_to_where or ignore_tags or snap_policy != "FIRST":
        snap_import = "from blender_cli.snap import SnapPolicy\n"
        to_part = ""
        if snap_to_where:
            to_part = f", to_where={snap_to_where!r}"
        elif ignore_tags:
            to_part = f", to=scene.ignore(tags={set(ignore_tags)!r})"
        snap_code = f"    pts = pts.snap(scene, axis={snap_axis!r}, policy=SnapPolicy.{snap_policy}{to_part})\n"

    tags_literal = "{" + ", ".join(repr(t) for t in sorted(tag_set)) + "}"

    code = (
        f'"""Build step: scatter {Path(prefab).name} using density map."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Instances\n"
        f"from blender_cli.geometry import Mask\n"
        f"from blender_cli.geometry import PointSet\n"
        f"from blender_cli.assets import Prefab\n"
        f"from blender_cli.scene import Scene\n"
        f"{snap_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    prefab = Prefab(Path({prefab_abs!r}))\n"
        f"    mask = Mask.from_image({density_abs!r})\n"
        f"    pts = PointSet.poisson(mask, density={density_value}, seed={seed_expr})\n"
        f"{snap_code}"
        f"{randomize_call}"
        f"    inst = Instances.from_points(prefab, pts, align={align!r})\n"
        f"    scene.add(inst, name={display_name!r}, tags={tags_literal})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


@op.command("scatter_near")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--anchor", default=None, help="Anchor name/annotation for center point.")
@click.option(
    "--center", default=None, nargs=3, type=float, help="Fallback center x y z."
)
@click.option(
    "--spec", required=True, type=click.Path(exists=True), help="JSON spec file."
)
@click.option("--snap", "snap_axis", default=None, type=click.Choice(sorted(AXIS_DIR)))
@click.option(
    "--ignore", "ignore_tags", multiple=True, help="Tags to ignore during snap."
)
@click.option(
    "--snap-to-where", default=None, help="DSL expression for snap target selection."
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def scatter_near(
    glb: str,
    anchor: str | None,
    center: tuple[float, float, float] | None,
    spec: str,
    snap_axis: str | None,
    ignore_tags: tuple[str, ...],
    snap_to_where: str | None,
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step for clustered mixed-prop placement from a JSON spec."""
    spec_abs = str(Path(spec).resolve())
    payload = json.loads(Path(spec_abs).read_text(encoding="utf-8"))
    raw_items = payload.get("items", payload if isinstance(payload, list) else [])
    if not isinstance(raw_items, list):
        msg = "Spec must be a list or an object with an 'items' list"
        raise click.ClickException(msg)
    for item in raw_items:
        if not isinstance(item, dict) or "prefab" not in item:
            msg = "All scatter spec entries must be objects with a 'prefab' field"
            raise click.ClickException(msg)

    with _quiet():
        scene = Scene.load(glb)
        if anchor:
            a = scene.anchor(anchor)
            if a is None:
                msg = f"Anchor not found: {anchor}"
                raise click.ClickException(msg)
            c = a.location()
            center_pos = {"x": c.x, "y": c.y, "z": c.z}
        elif center is not None:
            center_pos = {"x": center[0], "y": center[1], "z": center[2]}
        else:
            msg = "Provide --anchor or --center"
            raise click.ClickException(msg)
        ctx = _scene_context(scene)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "scatter_near",
        "result": {
            "center": center_pos,
            "item_count": len(raw_items),
        },
        "warnings": [],
        "context": ctx,
    }
    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    center_expr = (
        f"scene.anchor({anchor!r})"
        if anchor
        else f"Vec3({center_pos['x']}, {center_pos['y']}, {center_pos['z']})"
    )
    snap_expr = "None"
    if snap_axis:
        if snap_to_where:
            snap_expr = f"SnapSpec(axis={snap_axis!r}, to_where={snap_to_where!r})"
        elif ignore_tags:
            snap_expr = (
                "SnapSpec("
                f"axis={snap_axis!r}, "
                f"to=scene.ignore(tags={set(ignore_tags)!r})"
                ")"
            )
        else:
            snap_expr = f"SnapSpec(axis={snap_axis!r})"

    code = (
        '"""Build step: scatter mixed props near a center using JSON spec."""\n'
        "import json\n"
        "from pathlib import Path\n\n"
        "from blender_cli.build import BuildContext\n"
        "from blender_cli.assets import Prefab\n"
        "from blender_cli.scene import SnapSpec, Scene\n"
        "from blender_cli.types import Vec3\n\n\n"
        "def run(ctx: BuildContext) -> None:\n"
        "    scene = Scene(ctx.open_prev())\n"
        f"    spec = json.loads(Path({spec_abs!r}).read_text())\n"
        "    raw_items = spec.get('items', spec if isinstance(spec, list) else [])\n"
        f"    center = {center_expr}\n"
        f"    snap = {snap_expr}\n"
        "    for item in raw_items:\n"
        "        prefab = Prefab(Path(item['prefab']))\n"
        "        offset = item.get('offset', [0, 0, 0])\n"
        "        pos = center + Vec3(*offset)\n"
        "        ent = prefab.spawn(item.get('name', prefab.path.stem)).at(pos)"
        ".scale(item.get('scale', 1.0))\n"
        "        if snap is not None:\n"
        "            ent.snap(scene, spec=snap)\n"
        "        scene.add(ent)\n"
        "    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op init — bootstrap a new scene
# ---------------------------------------------------------------------------


@op.command("init")
@click.option("--name", required=True, help="Scene name.")
@click.option(
    "--material", "materials", multiple=True, help="id=folder_path (repeatable)."
)
@click.option("--prefab", "prefabs", multiple=True, help="id=file_path (repeatable).")
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def init(
    name: str,
    materials: tuple[str, ...],
    prefabs: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that creates a new empty scene with registries."""
    mat_pairs = _parse_kv_pairs(materials)
    prefab_pairs = _parse_kv_pairs(prefabs)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "init",
        "result": {
            "name": name,
            "materials": list(mat_pairs.keys()),
            "prefabs": list(prefab_pairs.keys()),
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_lines = ""
    for mid, folder in mat_pairs.items():
        folder_abs = str(Path(folder).resolve())
        mat_lines += (
            f"    scene.materials.index_material({mid!r}, Path({folder_abs!r}))\n"
        )

    prefab_lines = ""
    for pid, fpath in prefab_pairs.items():
        fpath_abs = str(Path(fpath).resolve())
        prefab_lines += f"    scene.assets.index_prefab({pid!r}, Path({fpath_abs!r}))\n"

    code = (
        f'"""Build step: init scene {name!r}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene\n\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene.new()\n"
        f"{mat_lines}"
        f"{prefab_lines}"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op primitive — box / cylinder / sphere / plane
# ---------------------------------------------------------------------------


@op.command("primitive")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--type",
    "prim_type",
    required=True,
    type=click.Choice(["box", "cylinder", "sphere", "plane", "cone", "torus"]),
)
@click.option("--name", required=True, help="Object name.")
@click.option(
    "--size",
    nargs=3,
    type=float,
    default=None,
    help="x y z (box) or w h (plane, z ignored).",
)
@click.option("--radius", type=float, default=None)
@click.option("--height", type=float, default=None)
@click.option("--segments", type=int, default=None)
@click.option("--rings", type=int, default=None)
@click.option("--radius2", type=float, default=None, help="Top radius for cone.")
@click.option(
    "--major-radius",
    "major_radius",
    type=float,
    default=None,
    help="Torus major radius.",
)
@click.option(
    "--minor-radius",
    "minor_radius",
    type=float,
    default=None,
    help="Torus minor radius.",
)
@click.option(
    "--major-segments",
    "major_segments",
    type=int,
    default=None,
    help="Torus major segments.",
)
@click.option(
    "--minor-segments",
    "minor_segments",
    type=int,
    default=None,
    help="Torus minor segments.",
)
@click.option("--material", "material_id", default=None, help="Registered material id.")
@click.option("--tile", type=float, default=None)
@click.option("--color", nargs=4, type=float, default=None, help="r g b a")
@click.option("--metallic", type=float, default=None)
@click.option("--roughness", type=float, default=None)
@click.option(
    "--at", "position", default=None, nargs=3, type=float, help="Position x y z."
)
@click.option("--anchor", default=None, help="Anchor name (alternative to --at).")
@click.option("--snap", "snap_axis", default=None, type=click.Choice(sorted(AXIS_DIR)))
@click.option(
    "--snap-policy",
    default="FIRST",
    type=click.Choice(["FIRST", "LAST", "HIGHEST", "LOWEST", "AVERAGE", "ORIENT"]),
    show_default=True,
)
@click.option(
    "--ignore", "ignore_tags", multiple=True, help="Tags to ignore during snap."
)
@click.option("--snap-to-where", default=None)
@click.option("--tag", "tags", multiple=True)
@click.option("--prop", "props_kv", multiple=True, help="key=value (repeatable).")
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def primitive(
    glb: str,
    prim_type: str,
    name: str,
    size: tuple[float, float, float] | None,
    radius: float | None,
    height: float | None,
    segments: int | None,
    rings: int | None,
    radius2: float | None,
    major_radius: float | None,
    minor_radius: float | None,
    major_segments: int | None,
    minor_segments: int | None,
    material_id: str | None,
    tile: float | None,
    color: tuple[float, float, float, float] | None,
    metallic: float | None,
    roughness: float | None,
    position: tuple[float, float, float] | None,
    anchor: str | None,
    snap_axis: str | None,
    snap_policy: str,
    ignore_tags: tuple[str, ...],
    snap_to_where: str | None,
    tags: tuple[str, ...],
    props_kv: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that adds a primitive shape to the scene."""
    if position is None and anchor is None:
        msg = "Provide either --at x y z or --anchor NAME."
        raise click.UsageError(msg)
    if position is not None and anchor is not None:
        msg = "--at and --anchor are mutually exclusive."
        raise click.UsageError(msg)

    props = _parse_kv_pairs(props_kv)

    snapped = False
    snap_surface: str | None = None
    warn_list: list[str] = []

    with _quiet():
        scene = Scene.load(glb)
        pos = _resolve_anchor_or_pos(scene, anchor, position)
        x, y, z = pos.x, pos.y, pos.z
        final_pos = pos
        if snap_axis:
            from blender_cli.snap import snap as snap_fn

            if snap_to_where:
                snap_scene = scene.snap_targets(snap_to_where)
            elif ignore_tags:
                snap_scene = scene.ignore(tags=set(ignore_tags))
            else:
                snap_scene = scene

            results = snap_fn([Vec3(x, y, z)], snap_scene, snap_axis)
            hit = results[0]
            if hit.hit:
                final_pos = hit.hit_pos
                snapped = True
                snap_surface = hit.hit_uid
            else:
                warn_list.append(f"Snap miss at ({x}, {y}, {z}) axis={snap_axis}")
            warn_list.extend(results.summary.warnings)
        nearby = _nearby_at_position(scene, final_pos)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "primitive",
        "result": {
            "type": prim_type,
            "name": name,
            "position": {
                "x": round(final_pos.x, 4),
                "y": round(final_pos.y, 4),
                "z": round(final_pos.z, 4),
            },
            "snapped": snapped,
            "snap_surface": snap_surface,
            "nearby": nearby,
            "tags": sorted(tags),
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    # Build shape-specific args for codegen
    if prim_type == "box":
        sz = size or (1.0, 1.0, 1.0)
        shape_args = f"size=({sz[0]}, {sz[1]}, {sz[2]})"
    elif prim_type == "cylinder":
        r = radius if radius is not None else 0.5
        h = height if height is not None else 1.0
        seg = segments if segments is not None else 32
        shape_args = f"radius={r}, height={h}, segments={seg}"
    elif prim_type == "sphere":
        r = radius if radius is not None else 0.5
        seg = segments if segments is not None else 32
        rng = rings if rings is not None else 16
        shape_args = f"radius={r}, segments={seg}, rings={rng}"
    elif prim_type == "cone":
        r1 = radius if radius is not None else 1.0
        r2 = radius2 if radius2 is not None else 0.0
        d = height if height is not None else 2.0
        verts = segments if segments is not None else 32
        shape_args = f"radius1={r1}, radius2={r2}, depth={d}, vertices={verts}"
    elif prim_type == "torus":
        majr = major_radius if major_radius is not None else 1.0
        minr = minor_radius if minor_radius is not None else 0.25
        majs = major_segments if major_segments is not None else 48
        mins = minor_segments if minor_segments is not None else 12
        shape_args = f"major_radius={majr}, minor_radius={minr}, major_segments={majs}, minor_segments={mins}"
    else:  # plane
        sz = size or (1.0, 1.0, 1.0)
        shape_args = f"size=({sz[0]}, {sz[1]})"

    mat_code, mat_import = _material_codegen(
        material_id, tile, color, metallic, roughness
    )
    mat_arg = ", material=mat" if material_id or color else ""
    mat_assign = f"    mat = {mat_code}\n" if mat_arg else ""

    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""
    props_code = (
        f".props({', '.join(f'{k}={v!r}' for k, v in props.items())})" if props else ""
    )

    snap_import = ""
    snap_code = ""
    if snap_axis:
        snap_import = "from blender_cli.snap import SnapPolicy\n"
        to_part = (
            f", to_where={snap_to_where!r}"
            if snap_to_where
            else (
                f", to=scene.ignore(tags={set(ignore_tags)!r})" if ignore_tags else ""
            )
        )
        snap_code = f"\n        .snap(scene, axis={snap_axis!r}, policy=SnapPolicy.{snap_policy}{to_part})"

    # Position expression: use anchor expression or literal coords
    at_expr = f"scene.anchor({anchor!r}).location()" if anchor else f"{x}, {y}, {z}"

    code = (
        f'"""Build step: add {prim_type} {name!r}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene, {prim_type}\n"
        f"from blender_cli.types import Vec3\n"
        f"{snap_import}"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"    scene.add(\n"
        f"        {prim_type}({name!r}, {shape_args}{mat_arg})\n"
        f"        .at({at_expr}){snap_code}\n"
        f"        .named({name!r}){tag_code}{props_code}\n"
        f"    )\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op anchor — create named waypoints
# ---------------------------------------------------------------------------


@op.command("anchor")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--anchor",
    "anchors_raw",
    multiple=True,
    required=True,
    help="name=ANNOTATION=x,y,z (repeatable).",
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def anchor_cmd(
    glb: str,
    anchors_raw: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that creates named anchors."""
    parsed: dict[str, tuple[str, tuple[float, float, float]]] = {}
    warn_list: list[str] = []

    for raw in anchors_raw:
        parts = raw.split("=", 2)
        if len(parts) != 3:
            warn_list.append(
                f"Invalid anchor format (expected name=ANN=x,y,z): {raw!r}"
            )
            continue
        aname, ann, coords = parts
        try:
            point = _parse_point(coords)
        except Exception:
            warn_list.append(f"Invalid coordinates in anchor: {raw!r}")
            continue
        parsed[aname.strip()] = (ann.strip(), point)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "anchor",
        "result": {
            "anchors": {
                k: {
                    "annotation": v[0],
                    "position": {"x": v[1][0], "y": v[1][1], "z": v[1][2]},
                }
                for k, v in parsed.items()
            },
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    anchor_dict_code = "{\n"
    for aname, (ann, (ax, ay, az)) in parsed.items():
        anchor_dict_code += f"        {aname!r}: ({ann!r}, ({ax}, {ay}, {az})),\n"
    anchor_dict_code += "    }"

    code = (
        f'"""Build step: create anchors."""\n'
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene\n\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    scene.anchors({anchor_dict_code})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op terrain — heightmap-based terrain
# ---------------------------------------------------------------------------


@op.command("terrain")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--heightmap",
    default=None,
    type=click.Path(exists=True),
    help="Heightmap image path.",
)
@click.option(
    "--flat", is_flag=True, default=False, help="Use flat terrain instead of heightmap."
)
@click.option(
    "--flat-width", type=float, default=None, help="Flat terrain width (metres)."
)
@click.option(
    "--flat-height", type=float, default=None, help="Flat terrain height (metres)."
)
@click.option(
    "--flat-z",
    type=float,
    default=0.0,
    show_default=True,
    help="Flat terrain elevation.",
)
@click.option(
    "--remap",
    nargs=2,
    type=float,
    default=(0.0, 10.0),
    show_default=True,
    help="min max elevation.",
)
@click.option("--meters-per-px", type=float, default=1.0, show_default=True)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--skirts", type=float, default=0.0, show_default=True)
@click.option("--lod", type=int, default=1, show_default=True)
@click.option("--noise-type", default=None, type=click.Choice(["fbm", "ridged"]))
@click.option("--noise-amp", type=float, default=None)
@click.option("--noise-freq", type=float, default=None)
@click.option("--noise-seed", type=int, default=0, show_default=True)
@click.option("--noise-octaves", type=int, default=6, show_default=True)
# Erosion options
@click.option("--erode-type", default=None, type=click.Choice(["hydraulic", "thermal"]))
@click.option("--erode-iterations", type=int, default=50, show_default=True)
@click.option("--erode-seed", type=int, default=0, show_default=True)
@click.option(
    "--talus-angle",
    type=float,
    default=35.0,
    show_default=True,
    help="Talus angle for thermal erosion.",
)
# Smooth options
@click.option(
    "--smooth-radius", type=int, default=None, help="Gaussian smooth pixel radius."
)
@click.option(
    "--smooth-iters",
    type=int,
    default=1,
    show_default=True,
    help="Number of smooth passes.",
)
# Stamp options
@click.option("--stamp-shape", default=None, type=click.Choice(["circle", "ring"]))
@click.option("--stamp-center", default=None, help="x,y stamp center.")
@click.option("--stamp-radius", type=float, default=None)
@click.option(
    "--stamp-op",
    default="add",
    type=click.Choice(["add", "sub", "set"]),
    show_default=True,
)
@click.option("--stamp-amount", type=float, default=1.0, show_default=True)
@click.option(
    "--stamp-falloff",
    default="smooth",
    type=click.Choice(["smooth", "linear", "sharp"]),
    show_default=True,
)
# Radial falloff options
@click.option("--radial-center", default=None, help="x,y radial center.")
@click.option("--radial-radius", type=float, default=None)
@click.option("--radial-edge-width", type=float, default=0.0, show_default=True)
# Remap curve option
@click.option(
    "--remap-curve",
    "remap_curve_pairs",
    multiple=True,
    help="in,out pair (repeatable).",
)
@click.option("--name", default="terrain", show_default=True)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def terrain_cmd(
    glb: str,
    heightmap: str | None,
    flat: bool,
    flat_width: float | None,
    flat_height: float | None,
    flat_z: float,
    remap: tuple[float, float],
    meters_per_px: float,
    material_id: str | None,
    tile: float | None,
    skirts: float,
    lod: int,
    noise_type: str | None,
    noise_amp: float | None,
    noise_freq: float | None,
    noise_seed: int,
    noise_octaves: int,
    erode_type: str | None,
    erode_iterations: int,
    erode_seed: int,
    talus_angle: float,
    smooth_radius: int | None,
    smooth_iters: int,
    stamp_shape: str | None,
    stamp_center: str | None,
    stamp_radius: float | None,
    stamp_op: str,
    stamp_amount: float,
    stamp_falloff: str,
    radial_center: str | None,
    radial_radius: float | None,
    radial_edge_width: float,
    remap_curve_pairs: tuple[str, ...],
    name: str,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that creates terrain from a heightmap image or flat grid."""
    if not heightmap and not flat:
        msg = "Provide either --heightmap or --flat"
        raise click.ClickException(msg)

    heightmap_abs = str(Path(heightmap).resolve()) if heightmap else None
    remap_min, remap_max = remap

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "terrain",
        "result": {
            "heightmap": heightmap_abs,
            "flat": flat,
            "remap": [remap_min, remap_max],
            "meters_per_px": meters_per_px,
            "name": name,
            "tags": sorted(tags),
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, None, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id else ""
    mat_arg = ", material=mat" if material_id else ""
    tile_arg = f", tile_scale={tile}" if tile else ""

    # Heightfield source
    if flat:
        fw = flat_width if flat_width is not None else 64.0
        fh = flat_height if flat_height is not None else 64.0
        hf_source = f"    hf = Heightfield.flat({int(fw)}, {int(fh)}, z={flat_z}, meters_per_px={meters_per_px})\n"
    else:
        hf_source = (
            f"    field = Field2D.from_image({heightmap_abs!r}, meters_per_px={meters_per_px})\n"
            f"    field = field.remap({remap_min}, {remap_max})\n"
            f"    hf = Heightfield(field)\n"
        )

    noise_code = ""
    if noise_type and noise_amp is not None and noise_freq is not None:
        noise_code = (
            f"    hf = hf.add_noise(type={noise_type!r}, amp={noise_amp}, "
            f"freq={noise_freq}, seed={noise_seed}, octaves={noise_octaves})\n"
        )

    erode_code = ""
    if erode_type:
        if erode_type == "thermal":
            erode_code = (
                f"    hf = hf.erode(type={erode_type!r}, iterations={erode_iterations}, "
                f"talus_angle={talus_angle})\n"
            )
        else:
            erode_code = (
                f"    hf = hf.erode(type={erode_type!r}, iterations={erode_iterations}, "
                f"seed={erode_seed})\n"
            )

    smooth_code = ""
    if smooth_radius is not None:
        smooth_code = (
            f"    hf = hf.smooth(radius={smooth_radius}, iters={smooth_iters})\n"
        )

    stamp_code = ""
    if stamp_shape and stamp_center and stamp_radius is not None:
        sc = stamp_center.split(",")
        sx, sy = float(sc[0]), float(sc[1])
        stamp_code = (
            f"    hf = hf.stamp({stamp_shape!r}, center=({sx}, {sy}), "
            f"radius={stamp_radius}, operation={stamp_op!r}, "
            f"amount={stamp_amount}, falloff={stamp_falloff!r})\n"
        )

    radial_code = ""
    if radial_center and radial_radius is not None:
        rc = radial_center.split(",")
        rx, ry = float(rc[0]), float(rc[1])
        radial_code = (
            f"    hf = hf.radial_falloff(center=({rx}, {ry}), "
            f"radius={radial_radius}, edge_width={radial_edge_width})\n"
        )

    remap_curve_code = ""
    if remap_curve_pairs:
        points_list: list[tuple[float, float]] = []
        for pair in remap_curve_pairs:
            parts = pair.split(",")
            points_list.append((float(parts[0]), float(parts[1])))
        pts_repr = repr(points_list)
        remap_curve_code = f"    hf = hf.remap_curve({pts_repr})\n"

    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""

    code = (
        f'"""Build step: terrain from {"flat grid" if flat else "heightmap"}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Field2D, Heightfield\n"
        f"from blender_cli.scene import Scene\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"{hf_source}"
        f"{noise_code}"
        f"{erode_code}"
        f"{smooth_code}"
        f"{stamp_code}"
        f"{radial_code}"
        f"{remap_curve_code}"
        f"    ent = hf.to_mesh(lod={lod}, skirts={skirts}{mat_arg}{tile_arg}){tag_code}\n"
        f"    scene.add(ent, name={name!r})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op path — spline sweep
# ---------------------------------------------------------------------------


@op.command("path")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--name", required=True)
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor",
    "anchor_names",
    multiple=True,
    help="Anchor name (repeatable, resolved to positions).",
)
@click.option("--width", type=float, default=2.0, show_default=True)
@click.option("--resample-step", type=float, default=1.0, show_default=True)
@click.option("--z-offset", type=float, default=0.0, show_default=True)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--color", nargs=4, type=float, default=None, help="r g b a")
@click.option("--closed", is_flag=True, default=False)
@click.option(
    "--snap-spline", is_flag=True, default=False, help="Snap spline to scene geometry."
)
@click.option(
    "--conform-to-terrain",
    is_flag=True,
    default=False,
    help="Conform sweep mesh to terrain.",
)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def path_cmd(
    glb: str,
    name: str,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    width: float,
    resample_step: float,
    z_offset: float,
    material_id: str | None,
    tile: float | None,
    color: tuple[float, float, float, float] | None,
    closed: bool,
    snap_spline: bool,
    conform_to_terrain: bool,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that sweeps a profile along a spline path."""
    # Parse explicit points
    control_points: list[tuple[float, float, float]] = [
        _parse_point(raw) for raw in points_raw
    ]

    # Resolve anchor names
    anchor_resolved: list[str] = list(anchor_names)

    if not control_points and not anchor_resolved:
        msg = "Provide at least --point or --anchor values"
        raise click.ClickException(msg)

    warn_list: list[str] = []
    if anchor_resolved:
        with _quiet():
            scene = Scene.load(glb)
            for aname in anchor_resolved:
                a = scene.anchor(aname)
                if a is None:
                    warn_list.append(f"Anchor not found: {aname}")
                else:
                    loc = a.location()
                    control_points.append((loc.x, loc.y, loc.z))

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "path",
        "result": {
            "name": name,
            "control_points": len(control_points),
            "width": width,
            "closed": closed,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, color, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id or color else ""
    mat_arg = ", material=mat" if material_id or color else ""

    pts_code = ", ".join(f"Vec3({p[0]}, {p[1]}, {p[2]})" for p in control_points)
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""
    snap_code = "    spline = spline.snap(scene)\n" if snap_spline else ""
    closed_arg = ", closed=True" if closed else ""

    code = (
        f'"""Build step: path {name!r}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Spline\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"from blender_cli.utils.sweep import sweep\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"    spline = Spline.catmull([{pts_code}]{closed_arg})\n"
        f"    spline = spline.resample({resample_step})\n"
        f"{snap_code}"
        f"    ent = sweep({name!r}, spline, width={width}{mat_arg}){tag_code}\n"
        f"    scene.add(ent, name={name!r})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op water — flat water plane
# ---------------------------------------------------------------------------


@op.command("water")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--name", required=True)
@click.option("--size", nargs=2, type=float, required=True, help="width height")
@click.option("--at", "position", nargs=3, type=float, required=True, help="x y z")
@click.option(
    "--color", nargs=4, type=float, default=(0.1, 0.3, 0.6, 0.8), show_default=True
)
@click.option("--metallic", type=float, default=0.3, show_default=True)
@click.option("--roughness", type=float, default=0.1, show_default=True)
@click.option(
    "--material",
    "material_id",
    default=None,
    help="Registered material id (overrides color).",
)
@click.option("--tile", type=float, default=None)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def water_cmd(
    glb: str,
    name: str,
    size: tuple[float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float, float],
    metallic: float,
    roughness: float,
    material_id: str | None,
    tile: float | None,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that adds a flat water plane."""
    x, y, z = position
    w, h = size

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "water",
        "result": {
            "name": name,
            "size": [w, h],
            "position": {"x": x, "y": y, "z": z},
            "tags": sorted(tags),
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""

    if material_id:
        mat_code, mat_import = _material_codegen(material_id, tile, None, None, None)
        mat_assign = f"    mat = {mat_code}\n"
        mat_arg = ", material=mat"
    else:
        mat_import = ""
        rgba = ", ".join(str(c) for c in color)
        water_mat_name = f"{name}_water"
        mat_assign = (
            f"    import bpy\n"
            f"    mat_bpy = bpy.data.materials.new(name={water_mat_name!r})\n"
            f"    mat_bpy.use_nodes = True\n"
            f"    bsdf = mat_bpy.node_tree.nodes['Principled BSDF']\n"
            f"    bsdf.inputs['Base Color'].default_value = ({rgba})\n"
            f"    bsdf.inputs['Metallic'].default_value = {metallic}\n"
            f"    bsdf.inputs['Roughness'].default_value = {roughness}\n"
        )
        mat_arg = ""

    code = (
        f'"""Build step: water plane {name!r}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene, plane\n"
        f"from blender_cli.types import Vec3\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"    ent = plane({name!r}, size=({w}, {h}){mat_arg})\n"
        f"    ent.at({x}, {y}, {z}).named({name!r}){tag_code}\n"
    )
    if not material_id:
        # Apply raw bpy material to the plane object
        code += "    ent.target.data.materials.append(mat_bpy)\n"
    code += "    scene.add(ent)\n    scene.save(ctx.out_path())\n"
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op grade — grade terrain along spline (SplineOp.grade)
# ---------------------------------------------------------------------------


@op.command("grade")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--heightmap", required=True, type=click.Path(exists=True))
@click.option("--remap", nargs=2, type=float, default=(0.0, 10.0), show_default=True)
@click.option("--meters-per-px", type=float, default=1.0, show_default=True)
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor", "anchor_names", multiple=True, help="Anchor name (repeatable)."
)
@click.option(
    "--width", type=float, required=True, help="Grade corridor width in metres."
)
@click.option("--shoulder", type=float, default=0.0, show_default=True)
@click.option("--cut", type=float, default=None, help="Max cut depth.")
@click.option("--fill", type=float, default=None, help="Max fill height.")
@click.option("--closed", is_flag=True, default=False)
@click.option("--resample-step", type=float, default=1.0, show_default=True)
@click.option(
    "--snap-spline", is_flag=True, default=False, help="Snap spline to scene geometry."
)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--skirts", type=float, default=0.0, show_default=True)
@click.option("--lod", type=int, default=1, show_default=True)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def grade_cmd(
    glb: str,
    heightmap: str,
    remap: tuple[float, float],
    meters_per_px: float,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    width: float,
    shoulder: float,
    cut: float | None,
    fill: float | None,
    closed: bool,
    resample_step: float,
    snap_spline: bool,
    material_id: str | None,
    tile: float | None,
    skirts: float,
    lod: int,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that grades terrain along a spline."""
    heightmap_abs = str(Path(heightmap).resolve())
    remap_min, remap_max = remap

    control_points: list[tuple[float, float, float]] = [
        _parse_point(raw) for raw in points_raw
    ]

    warn_list: list[str] = []
    if anchor_names:
        with _quiet():
            scene = Scene.load(glb)
            for aname in anchor_names:
                a = scene.anchor(aname)
                if a is None:
                    warn_list.append(f"Anchor not found: {aname}")
                else:
                    loc = a.location()
                    control_points.append((loc.x, loc.y, loc.z))

    if not control_points:
        msg = "Provide at least --point or --anchor values"
        raise click.ClickException(msg)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "grade",
        "result": {
            "heightmap": heightmap_abs,
            "control_points": len(control_points),
            "width": width,
            "shoulder": shoulder,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, None, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id else ""
    mat_arg = ", material=mat" if material_id else ""
    tile_arg = f", tile_scale={tile}" if tile else ""
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""
    cut_arg = f", cut={cut}" if cut is not None else ""
    fill_arg = f", fill={fill}" if fill is not None else ""

    hf_spline = _spline_hf_codegen(
        heightmap_abs,
        meters_per_px,
        remap_min,
        remap_max,
        control_points,
        resample_step,
        closed,
        snap_spline,
    )

    code = (
        f'"""Build step: grade terrain along spline."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Field2D, Heightfield, Spline\n"
        f"from blender_cli.geometry.spline_ops import SplineOp\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"{hf_spline}"
        f"    hf = SplineOp.grade(hf, spline, width={width}, shoulder={shoulder}{cut_arg}{fill_arg})\n"
        f"    # Delete old terrain if present\n"
        f"    try:\n"
        f"        sel = scene.select(\"tags.has('terrain')\")\n"
        f"        scene.delete(sel)\n"
        f"    except Exception:\n"
        f"        pass\n"
        f"    ent = hf.to_mesh(lod={lod}, skirts={skirts}{mat_arg}{tile_arg}){tag_code}\n"
        f"    scene.add(ent, name='terrain')\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op carve — dig channels along spline
# ---------------------------------------------------------------------------


@op.command("carve")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--heightmap", required=True, type=click.Path(exists=True))
@click.option("--remap", nargs=2, type=float, default=(0.0, 10.0), show_default=True)
@click.option("--meters-per-px", type=float, default=1.0, show_default=True)
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor", "anchor_names", multiple=True, help="Anchor name (repeatable)."
)
@click.option("--width", type=float, required=True, help="Channel width in metres.")
@click.option("--depth", type=float, required=True, help="Channel depth.")
@click.option("--shoulder", type=float, default=0.0, show_default=True)
@click.option(
    "--profile",
    default="parabolic",
    type=click.Choice(["parabolic", "flat", "v_shape"]),
    show_default=True,
    help="Channel cross-section profile.",
)
@click.option("--closed", is_flag=True, default=False)
@click.option("--resample-step", type=float, default=1.0, show_default=True)
@click.option(
    "--snap-spline", is_flag=True, default=False, help="Snap spline to scene geometry."
)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--skirts", type=float, default=0.0, show_default=True)
@click.option("--lod", type=int, default=1, show_default=True)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def carve_cmd(
    glb: str,
    heightmap: str,
    remap: tuple[float, float],
    meters_per_px: float,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    width: float,
    depth: float,
    shoulder: float,
    profile: str,
    closed: bool,
    resample_step: float,
    snap_spline: bool,
    material_id: str | None,
    tile: float | None,
    skirts: float,
    lod: int,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that carves a channel along a spline."""
    heightmap_abs = str(Path(heightmap).resolve())
    remap_min, remap_max = remap

    control_points: list[tuple[float, float, float]] = [
        _parse_point(raw) for raw in points_raw
    ]

    warn_list: list[str] = []
    if anchor_names:
        with _quiet():
            scene = Scene.load(glb)
            for aname in anchor_names:
                a = scene.anchor(aname)
                if a is None:
                    warn_list.append(f"Anchor not found: {aname}")
                else:
                    loc = a.location()
                    control_points.append((loc.x, loc.y, loc.z))

    if not control_points:
        msg = "Provide at least --point or --anchor values"
        raise click.ClickException(msg)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "carve",
        "result": {
            "heightmap": heightmap_abs,
            "control_points": len(control_points),
            "width": width,
            "depth": depth,
            "shoulder": shoulder,
            "profile": profile,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, None, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id else ""
    mat_arg = ", material=mat" if material_id else ""
    tile_arg = f", tile_scale={tile}" if tile else ""
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""

    hf_spline = _spline_hf_codegen(
        heightmap_abs,
        meters_per_px,
        remap_min,
        remap_max,
        control_points,
        resample_step,
        closed,
        snap_spline,
    )

    code = (
        f'"""Build step: carve channel along spline."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Field2D, Heightfield, Spline\n"
        f"from blender_cli.geometry.spline_ops import SplineOp\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"{hf_spline}"
        f"    hf = SplineOp.carve(hf, spline, width={width}, depth={depth}, shoulder={shoulder}, profile={profile!r})\n"
        f"    # Delete old terrain if present\n"
        f"    try:\n"
        f"        sel = scene.select(\"tags.has('terrain')\")\n"
        f"        scene.delete(sel)\n"
        f"    except Exception:\n"
        f"        pass\n"
        f"    ent = hf.to_mesh(lod={lod}, skirts={skirts}{mat_arg}{tile_arg}){tag_code}\n"
        f"    scene.add(ent, name='terrain')\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op embank — raise terrain along spline
# ---------------------------------------------------------------------------


@op.command("embank")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--heightmap", required=True, type=click.Path(exists=True))
@click.option("--remap", nargs=2, type=float, default=(0.0, 10.0), show_default=True)
@click.option("--meters-per-px", type=float, default=1.0, show_default=True)
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor", "anchor_names", multiple=True, help="Anchor name (repeatable)."
)
@click.option("--width", type=float, required=True, help="Embankment width in metres.")
@click.option(
    "--height", "embank_height", type=float, required=True, help="Embankment height."
)
@click.option("--shoulder", type=float, default=0.0, show_default=True)
@click.option("--closed", is_flag=True, default=False)
@click.option("--resample-step", type=float, default=1.0, show_default=True)
@click.option(
    "--snap-spline", is_flag=True, default=False, help="Snap spline to scene geometry."
)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--skirts", type=float, default=0.0, show_default=True)
@click.option("--lod", type=int, default=1, show_default=True)
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def embank_cmd(
    glb: str,
    heightmap: str,
    remap: tuple[float, float],
    meters_per_px: float,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    width: float,
    embank_height: float,
    shoulder: float,
    closed: bool,
    resample_step: float,
    snap_spline: bool,
    material_id: str | None,
    tile: float | None,
    skirts: float,
    lod: int,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that raises terrain (embankment) along a spline."""
    heightmap_abs = str(Path(heightmap).resolve())
    remap_min, remap_max = remap

    control_points: list[tuple[float, float, float]] = [
        _parse_point(raw) for raw in points_raw
    ]

    warn_list: list[str] = []
    if anchor_names:
        with _quiet():
            scene = Scene.load(glb)
            for aname in anchor_names:
                a = scene.anchor(aname)
                if a is None:
                    warn_list.append(f"Anchor not found: {aname}")
                else:
                    loc = a.location()
                    control_points.append((loc.x, loc.y, loc.z))

    if not control_points:
        msg = "Provide at least --point or --anchor values"
        raise click.ClickException(msg)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "embank",
        "result": {
            "heightmap": heightmap_abs,
            "control_points": len(control_points),
            "width": width,
            "height": embank_height,
            "shoulder": shoulder,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, None, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id else ""
    mat_arg = ", material=mat" if material_id else ""
    tile_arg = f", tile_scale={tile}" if tile else ""
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""

    hf_spline = _spline_hf_codegen(
        heightmap_abs,
        meters_per_px,
        remap_min,
        remap_max,
        control_points,
        resample_step,
        closed,
        snap_spline,
    )

    code = (
        f'"""Build step: embank terrain along spline."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Field2D, Heightfield, Spline\n"
        f"from blender_cli.geometry.spline_ops import SplineOp\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"{hf_spline}"
        f"    hf = SplineOp.embank(hf, spline, width={width}, height={embank_height}, shoulder={shoulder})\n"
        f"    # Delete old terrain if present\n"
        f"    try:\n"
        f"        sel = scene.select(\"tags.has('terrain')\")\n"
        f"        scene.delete(sel)\n"
        f"    except Exception:\n"
        f"        pass\n"
        f"    ent = hf.to_mesh(lod={lod}, skirts={skirts}{mat_arg}{tile_arg}){tag_code}\n"
        f"    scene.add(ent, name='terrain')\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op strip — spline texture overlay
# ---------------------------------------------------------------------------


@op.command("strip")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--name", required=True)
@click.option("--heightmap", required=True, type=click.Path(exists=True))
@click.option("--remap", nargs=2, type=float, default=(0.0, 10.0), show_default=True)
@click.option("--meters-per-px", type=float, default=1.0, show_default=True)
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor", "anchor_names", multiple=True, help="Anchor name (repeatable)."
)
@click.option("--width", type=float, required=True, help="Strip width in metres.")
@click.option(
    "--v-density",
    type=float,
    default=1.0,
    show_default=True,
    help="Tiles per metre along path.",
)
@click.option(
    "--u-repeats",
    type=float,
    default=1.0,
    show_default=True,
    help="Tiles across width.",
)
@click.option(
    "--falloff",
    type=float,
    default=0.0,
    show_default=True,
    help="Edge fadeout in metres.",
)
@click.option(
    "--z-offset",
    type=float,
    default=0.01,
    show_default=True,
    help="Z lift to prevent z-fighting.",
)
@click.option("--opacity", type=float, default=1.0, show_default=True)
@click.option("--closed", is_flag=True, default=False)
@click.option("--resample-step", type=float, default=1.0, show_default=True)
@click.option(
    "--snap-spline", is_flag=True, default=False, help="Snap spline to scene geometry."
)
@click.option("--material", "material_id", default=None)
@click.option("--tile", type=float, default=None)
@click.option("--color", nargs=4, type=float, default=None, help="r g b a")
@click.option("--tag", "tags", multiple=True)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def strip_cmd(
    glb: str,
    name: str,
    heightmap: str,
    remap: tuple[float, float],
    meters_per_px: float,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    width: float,
    v_density: float,
    u_repeats: float,
    falloff: float,
    z_offset: float,
    opacity: float,
    closed: bool,
    resample_step: float,
    snap_spline: bool,
    material_id: str | None,
    tile: float | None,
    color: tuple[float, float, float, float] | None,
    tags: tuple[str, ...],
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that creates a spline-conforming texture strip."""
    heightmap_abs = str(Path(heightmap).resolve())
    remap_min, remap_max = remap

    control_points: list[tuple[float, float, float]] = [
        _parse_point(raw) for raw in points_raw
    ]

    warn_list: list[str] = []
    if anchor_names:
        with _quiet():
            scene = Scene.load(glb)
            for aname in anchor_names:
                a = scene.anchor(aname)
                if a is None:
                    warn_list.append(f"Anchor not found: {aname}")
                else:
                    loc = a.location()
                    control_points.append((loc.x, loc.y, loc.z))

    if not control_points:
        msg = "Provide at least --point or --anchor values"
        raise click.ClickException(msg)

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "strip",
        "result": {
            "name": name,
            "heightmap": heightmap_abs,
            "control_points": len(control_points),
            "width": width,
        },
        "warnings": warn_list,
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    mat_code, mat_import = _material_codegen(material_id, tile, color, None, None)
    mat_assign = f"    mat = {mat_code}\n" if material_id or color else ""
    tag_code = f".tag({', '.join(repr(t) for t in tags)})" if tags else ""

    hf_spline = _spline_hf_codegen(
        heightmap_abs,
        meters_per_px,
        remap_min,
        remap_max,
        control_points,
        resample_step,
        closed,
        snap_spline,
    )

    # Build optional kwargs for spline_strip
    strip_kwargs = ""
    if v_density != 1.0:
        strip_kwargs += f", v_density={v_density}"
    if u_repeats != 1.0:
        strip_kwargs += f", u_repeats={u_repeats}"
    if falloff != 0.0:
        strip_kwargs += f", falloff={falloff}"
    if z_offset != 0.005:
        strip_kwargs += f", z_offset={z_offset}"
    if opacity != 1.0:
        strip_kwargs += f", opacity={opacity}"

    code = (
        f'"""Build step: spline strip {name!r}."""\n'
        f"from pathlib import Path\n\n"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.geometry import Field2D, Heightfield, Spline\n"
        f"from blender_cli.scene import Scene\n"
        f"from blender_cli.types import Vec3\n"
        f"from blender_cli.utils.spline_strip import spline_strip\n"
        f"{mat_import}\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"{mat_assign}"
        f"{hf_spline}"
        f"    ent = spline_strip({name!r}, spline, mat, width={width}, conform_to=hf{strip_kwargs}){tag_code}\n"
        f"    scene.add(ent, name={name!r})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op rotate — rotate matching entities
# ---------------------------------------------------------------------------


@op.command("rotate")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--angle", required=True, type=float, help="Rotation in degrees.")
@click.option(
    "--axis", default="Z", type=click.Choice(["X", "Y", "Z"]), show_default=True
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def rotate_cmd(
    glb: str,
    where: str,
    angle: float,
    axis: str,
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that rotates matching entities."""
    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        count = sel.count()
        uids = sel.uids()

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "rotate",
        "result": {
            "match_count": count,
            "uids": uids,
            "angle": angle,
            "axis": axis,
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    select_expr, extra_import = _codegen_select(where)

    # Map axis letter to keyword arg
    axis_kw = {"X": "pitch_deg", "Y": "roll_deg", "Z": "yaw_deg"}
    kw = axis_kw.get(axis, "yaw_deg")

    code = (
        f'"""Build step: rotate entities by {angle} deg around {axis}."""\n'
        f"{extra_import}"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene\n\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    sel = {select_expr}\n"
        f"    scene.transform(sel).rotate({kw}={angle})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))


# ---------------------------------------------------------------------------
# op scale — scale matching entities
# ---------------------------------------------------------------------------


@op.command("scale")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--factor", default=None, nargs=3, type=float, help="Scale sx sy sz.")
@click.option(
    "--uniform",
    default=None,
    type=float,
    help="Uniform scale (alternative to --factor).",
)
@click.option("--write-step", "write_step", required=True, type=click.Path())
@click.option("--dry-run", "dry_run", is_flag=True, default=False)
def scale_cmd(
    glb: str,
    where: str,
    factor: tuple[float, float, float] | None,
    uniform: float | None,
    write_step: str,
    dry_run: bool,
) -> None:
    """Generate a build step that scales matching entities."""
    if factor is None and uniform is None:
        msg = "Provide either --factor sx sy sz or --uniform N."
        raise click.UsageError(msg)
    if factor is not None and uniform is not None:
        msg = "--factor and --uniform are mutually exclusive."
        raise click.UsageError(msg)
    if uniform is not None:
        sx = sy = sz = uniform
    else:
        assert factor is not None
        sx, sy, sz = factor

    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        count = sel.count()
        uids = sel.uids()

    feedback: dict[str, object] = {
        "status": "ok",
        "action": "scale",
        "result": {
            "match_count": count,
            "uids": uids,
            "factor": {"sx": sx, "sy": sy, "sz": sz},
        },
        "warnings": [],
    }

    if dry_run:
        click.echo(json.dumps(feedback, indent=2))
        return

    select_expr, extra_import = _codegen_select(where)
    code = (
        f'"""Build step: scale entities by ({sx}, {sy}, {sz})."""\n'
        f"{extra_import}"
        f"from blender_cli.build import BuildContext\n"
        f"from blender_cli.scene import Scene\n\n\n"
        f"def run(ctx: BuildContext) -> None:\n"
        f"    scene = Scene(ctx.open_prev())\n"
        f"    sel = {select_expr}\n"
        f"    scene.transform(sel).scale({sx}, {sy}, {sz})\n"
        f"    scene.save(ctx.out_path())\n"
    )
    _write_script(write_step, code)
    feedback["script"] = write_step
    click.echo(json.dumps(feedback, indent=2))
