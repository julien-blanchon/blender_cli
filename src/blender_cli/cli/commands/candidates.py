"""Candidate point generation commands."""

from __future__ import annotations

import click

from blender_cli.cli.common import (
    _output,
    _parse_point,
    _quiet,
    _resolve_where,
    _scene_context,
)
from blender_cli.scene import Scene
from blender_cli.types import Vec3


@click.group()
def candidates() -> None:
    """Candidate point generation and inspection."""


@candidates.command("perimeter")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True, help="Selection for perimeter object(s).")
@click.option("--count", required=True, type=int)
@click.option("--inset", default=0.0, type=float, show_default=True)
@click.option(
    "--face",
    default="inward",
    type=click.Choice(["inward", "outward"]),
    show_default=True,
)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--out", default=None, type=click.Path())
def candidates_perimeter(
    glb: str,
    where: str,
    count: int,
    inset: float,
    face: str,
    seed: int,
    out: str | None,
) -> None:
    """Generate perimeter placement points with orientation metadata."""
    from blender_cli.utils.placement import perimeter_points

    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        pts = perimeter_points(
            sel,
            count=count,
            inset=inset,
            face=face,
            seed=seed,
        )
        ctx = _scene_context(scene)

    yaws = pts.attr("yaw") if pts.count > 0 else []
    data = {
        "count": pts.count,
        "points": [
            {
                "pos": {"x": p.x, "y": p.y, "z": p.z},
                "yaw": yaws[i],
            }
            for i, p in enumerate(pts.points)
        ],
        "context": ctx,
    }
    _output(data, out)


# ---------------------------------------------------------------------------
# candidates circle
# ---------------------------------------------------------------------------


@candidates.command("circle")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--center", nargs=3, type=float, default=None, help="x y z")
@click.option("--anchor", default=None, help="Anchor name for center.")
@click.option("--radius", required=True, type=float)
@click.option("--count", required=True, type=int)
@click.option("--out", default=None, type=click.Path())
def candidates_circle(
    glb: str,
    center: tuple[float, float, float] | None,
    anchor: str | None,
    radius: float,
    count: int,
    out: str | None,
) -> None:
    """Generate evenly spaced points on a circle."""
    from blender_cli.utils.placement import circle_points

    with _quiet():
        scene = Scene.load(glb)
        if anchor:
            a = scene.anchor(anchor)
            if a is None:
                msg = f"Anchor not found: {anchor}"
                raise click.ClickException(msg)
            origin = a.location()
        elif center is not None:
            origin = Vec3(*center)
        else:
            msg = "Provide --center or --anchor"
            raise click.ClickException(msg)

        pts = circle_points(origin, radius, count)
        ctx = _scene_context(scene)

    data = {
        "count": pts.count,
        "points": [{"pos": {"x": p.x, "y": p.y, "z": p.z}} for p in pts.points],
        "context": ctx,
    }
    _output(data, out)


# ---------------------------------------------------------------------------
# candidates line
# ---------------------------------------------------------------------------


@candidates.command("line")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--start", nargs=3, type=float, default=None, help="x y z")
@click.option("--end", nargs=3, type=float, default=None, help="x y z")
@click.option("--start-anchor", default=None, help="Anchor name for start.")
@click.option("--end-anchor", default=None, help="Anchor name for end.")
@click.option("--count", required=True, type=int)
@click.option("--out", default=None, type=click.Path())
def candidates_line(
    glb: str,
    start: tuple[float, float, float] | None,
    end: tuple[float, float, float] | None,
    start_anchor: str | None,
    end_anchor: str | None,
    count: int,
    out: str | None,
) -> None:
    """Generate evenly spaced points along a line."""
    from blender_cli.utils.placement import line_points

    with _quiet():
        scene = Scene.load(glb)

        if start_anchor:
            a = scene.anchor(start_anchor)
            if a is None:
                msg = f"Anchor not found: {start_anchor}"
                raise click.ClickException(msg)
            start_pos = a.location()
        elif start is not None:
            start_pos = Vec3(*start)
        else:
            msg = "Provide --start or --start-anchor"
            raise click.ClickException(msg)

        if end_anchor:
            a = scene.anchor(end_anchor)
            if a is None:
                msg = f"Anchor not found: {end_anchor}"
                raise click.ClickException(msg)
            end_pos = a.location()
        elif end is not None:
            end_pos = Vec3(*end)
        else:
            msg = "Provide --end or --end-anchor"
            raise click.ClickException(msg)

        pts = line_points(start_pos, end_pos, count)
        ctx = _scene_context(scene)

    data = {
        "count": pts.count,
        "points": [{"pos": {"x": p.x, "y": p.y, "z": p.z}} for p in pts.points],
        "context": ctx,
    }
    _output(data, out)


# ---------------------------------------------------------------------------
# candidates grid
# ---------------------------------------------------------------------------


@candidates.command("grid")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--corner1", required=True, nargs=3, type=float, help="x y z")
@click.option("--corner2", required=True, nargs=3, type=float, help="x y z")
@click.option("--step-x", required=True, type=float)
@click.option("--step-y", required=True, type=float)
@click.option("--jitter", default=0.0, type=float, show_default=True)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--out", default=None, type=click.Path())
def candidates_grid(
    glb: str,
    corner1: tuple[float, float, float],
    corner2: tuple[float, float, float],
    step_x: float,
    step_y: float,
    jitter: float,
    seed: int,
    out: str | None,
) -> None:
    """Generate a regular grid of points within a rectangle."""
    from blender_cli.utils.placement import grid_points

    with _quiet():
        scene = Scene.load(glb)
        pts = grid_points(
            Vec3(*corner1),
            Vec3(*corner2),
            step_x,
            step_y,
            jitter=jitter,
            seed=seed,
        )
        ctx = _scene_context(scene)

    data = {
        "count": pts.count,
        "points": [{"pos": {"x": p.x, "y": p.y, "z": p.z}} for p in pts.points],
        "context": ctx,
    }
    _output(data, out)


# ---------------------------------------------------------------------------
# candidates spline
# ---------------------------------------------------------------------------


@candidates.command("spline")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--point", "points_raw", multiple=True, help="x,y,z (repeatable control points)."
)
@click.option(
    "--anchor", "anchor_names", multiple=True, help="Anchor name (repeatable)."
)
@click.option("--every", required=True, type=float, help="Spacing in metres.")
@click.option("--jitter", default=0.0, type=float, show_default=True)
@click.option("--seed", default=0, type=int, show_default=True)
@click.option("--closed", is_flag=True, default=False)
@click.option("--out", default=None, type=click.Path())
def candidates_spline(
    glb: str,
    points_raw: tuple[str, ...],
    anchor_names: tuple[str, ...],
    every: float,
    jitter: float,
    seed: int,
    closed: bool,
    out: str | None,
) -> None:
    """Generate placement points at regular intervals along a spline."""
    from blender_cli.geometry import Spline
    from blender_cli.utils.placement import sample_along_spline

    control_points: list[Vec3] = [Vec3(*_parse_point(raw)) for raw in points_raw]

    with _quiet():
        scene = Scene.load(glb)
        for aname in anchor_names:
            a = scene.anchor(aname)
            if a is None:
                msg = f"Anchor not found: {aname}"
                raise click.ClickException(msg)
            control_points.append(a.location())

        if len(control_points) < 2:
            msg = "Need at least 2 control points (--point or --anchor)"
            raise click.ClickException(msg)

        spline = Spline.catmull(control_points, closed=closed)
        pts = sample_along_spline(spline, every_m=every, jitter_m=jitter, seed=seed)
        ctx = _scene_context(scene)

    data = {
        "count": pts.count,
        "points": [{"pos": {"x": p.x, "y": p.y, "z": p.z}} for p in pts.points],
        "context": ctx,
    }
    _output(data, out)
