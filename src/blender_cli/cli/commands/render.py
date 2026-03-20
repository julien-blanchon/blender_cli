"""Render commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from blender_cli.cli.common import _quiet, _resolve_where, _scene_context
from blender_cli.scene import Scene


@click.group()
def render() -> None:
    """Render commands."""


def _resolve_camera(
    name: str,
    scene: Scene,
    project_path: str | None,
) -> "Camera | None":
    """Resolve a camera by name from GLB scene or project.json (including ghosts)."""
    import math

    import bpy

    from blender_cli.render import Camera

    # 1. Try to find in GLB scene
    for obj in bpy.context.scene.objects:
        if obj.type == "CAMERA" and obj.name == name:
            cam = Camera.perspective(fov=obj.data.lens, name=name)
            cam.at(obj.location.x, obj.location.y, obj.location.z)
            cam._bpy_object.rotation_euler = obj.rotation_euler
            return cam

    # 2. Try project.json (ghost cameras)
    if project_path:
        from blender_cli.project.project_file import ProjectFile
        pf = ProjectFile.load(project_path)
        for cam_data in pf.data.get("cameras", []):
            if cam_data.get("name") == name:
                loc = cam_data.get("location", [0, 0, 0])
                rot = cam_data.get("rotation", [0, 0, 0])
                cam = Camera.perspective(fov=cam_data.get("lens", 50.0), name=name)
                cam.at(loc[0], loc[1], loc[2])
                cam._bpy_object.rotation_euler = (rot[0], rot[1], rot[2])
                return cam

    msg = f"Camera {name!r} not found in scene or project"
    raise click.ClickException(msg)


def _apply_project_world(project_path: str | None) -> None:
    """Apply world settings from a project.json to the current bpy scene."""
    if not project_path:
        return
    import bpy
    from blender_cli.project.project_file import ProjectFile
    pf = ProjectFile.load(project_path)
    world = pf.data.get("world", {})
    if world.get("use_hdri") and world.get("hdri_path"):
        pass  # TODO: full HDRI node setup
    elif world.get("background_color"):
        bg = world["background_color"]
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        w = bpy.context.scene.world
        w.use_nodes = True
        tree = w.node_tree
        if tree:
            tree.nodes.clear()
            bg_node = tree.nodes.new("ShaderNodeBackground")
            bg_node.inputs["Color"].default_value = (bg[0], bg[1], bg[2], 1.0)
            bg_node.inputs["Strength"].default_value = 1.0
            out_node = tree.nodes.new("ShaderNodeOutputWorld")
            tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])


@render.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--project", "project_path", default=None, type=click.Path(exists=True),
              help="Apply world/render settings from project.json.")
@click.option("--preset", default="top", show_default=True)
@click.option("--passes", default="beauty", help="Comma-separated pass names.")
@click.option(
    "--hide-tag", "hide_tags", multiple=True, help="Temporarily hide matching tags."
)
@click.option(
    "--show-tag",
    "show_tags",
    multiple=True,
    help="Force-show tags after hide filters.",
)
@click.option(
    "--hide-where", default=None, help="DSL expression to hide for this render."
)
@click.option("--highlight-where", default=None, help="DSL expression to highlight.")
@click.option("--ghost", "ghost_opacity", default=0.3, type=float, show_default=True)
@click.option(
    "--width", default=1920, type=int, show_default=True, help="Render width."
)
@click.option(
    "--height", default=1080, type=int, show_default=True, help="Render height."
)
@click.option(
    "--quality",
    default="draft",
    type=click.Choice(["draft", "preview", "final"]),
    show_default=True,
)
@click.option(
    "--mode",
    default="solid",
    type=click.Choice(["solid", "wireframe"]),
    show_default=True,
)
@click.option("--camera", default=None, help="Use a named camera from the scene or project.json.")
@click.option("--out", required=True, type=click.Path())
def still(
    glb: str,
    project_path: str | None,
    preset: str,
    passes: str,
    hide_tags: tuple[str, ...],
    show_tags: tuple[str, ...],
    hide_where: str | None,
    highlight_where: str | None,
    ghost_opacity: float,
    width: int,
    height: int,
    quality: str,
    mode: str,
    camera: str | None,
    out: str,
) -> None:
    """Render a still image of the full scene."""
    from blender_cli.render import Camera, RenderContext

    with _quiet():
        scene = Scene.load(glb)
        _apply_project_world(project_path)

        # Resolve named camera (from GLB scene or project.json ghost cameras)
        cam_obj = None
        if camera:
            cam_obj = _resolve_camera(camera, scene, project_path)

        rc = RenderContext(
            resolution=(width, height),
            quality=quality,  # type: ignore[arg-type]
        )
        rc.still(
            scene,
            preset=preset,
            out=out,
            passes=[p.strip() for p in passes.split(",")],
            hide_tags=set(hide_tags),
            show_tags=set(show_tags),
            camera=cam_obj,
            hide_where=hide_where,
            highlight_where=highlight_where,
            ghost_opacity=ghost_opacity,
            mode=mode,  # type: ignore[arg-type]
        )
        ctx = _scene_context(scene)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "render_still",
                "output": out,
                "resolution": [width, height],
                "quality": quality,
                "mode": mode,
                "context": ctx,
            },
            indent=2,
        )
    )


@render.command()
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--where", required=True)
@click.option("--preset", default="iso_close", show_default=True)
@click.option("--passes", default="beauty")
@click.option(
    "--hide-tag", "hide_tags", multiple=True, help="Temporarily hide matching tags."
)
@click.option(
    "--show-tag",
    "show_tags",
    multiple=True,
    help="Force-show tags after hide filters.",
)
@click.option(
    "--hide-where", default=None, help="DSL expression to hide for this render."
)
@click.option(
    "--width", default=1920, type=int, show_default=True, help="Render width."
)
@click.option(
    "--height", default=1080, type=int, show_default=True, help="Render height."
)
@click.option(
    "--quality",
    default="draft",
    type=click.Choice(["draft", "preview", "final"]),
    show_default=True,
)
@click.option("--out", required=True, type=click.Path())
def focus(
    glb: str,
    where: str,
    preset: str,
    passes: str,
    hide_tags: tuple[str, ...],
    show_tags: tuple[str, ...],
    hide_where: str | None,
    width: int,
    height: int,
    quality: str,
    out: str,
) -> None:
    """Render focused on matching entities."""
    from blender_cli.render import RenderContext

    with _quiet():
        scene = Scene.load(glb)
        sel = _resolve_where(scene, where)
        rc = RenderContext(
            resolution=(width, height),
            quality=quality,  # type: ignore[arg-type]
        )
        rc.focus(
            scene,
            target=sel,
            preset=preset,
            out=out,
            passes=[p.strip() for p in passes.split(",")],
            hide_tags=set(hide_tags),
            show_tags=set(show_tags),
            hide_where=hide_where,
        )
        ctx = _scene_context(scene)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "render_focus",
                "output": out,
                "resolution": [width, height],
                "quality": quality,
                "context": ctx,
            },
            indent=2,
        )
    )


@render.command("batch")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--spec",
    required=True,
    type=click.Path(exists=True),
    help="JSON list of render specs.",
)
@click.option("--out-dir", "out_dir", required=True, type=click.Path())
def render_batch(glb: str, spec: str, out_dir: str) -> None:
    """Render multiple shots from a JSON batch spec."""
    from blender_cli.render import RenderContext

    specs = json.loads(Path(spec).read_text(encoding="utf-8"))
    if not isinstance(specs, list):
        msg = "Batch spec must be a JSON list"
        raise click.ClickException(msg)

    with _quiet():
        scene = Scene.load(glb)
        paths = RenderContext().batch(scene, specs, out_dir=out_dir)
        ctx = _scene_context(scene)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "render_batch",
                "outputs": [str(p) for p in paths],
                "context": ctx,
            },
            indent=2,
        )
    )


@render.command("showcase")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option(
    "--target",
    "targets",
    multiple=True,
    required=True,
    help="DSL expression for targets (repeatable).",
)
@click.option(
    "--out-dir",
    "out_dir",
    required=True,
    type=click.Path(),
    help="Output directory for frames.",
)
@click.option(
    "--video",
    default=None,
    type=click.Path(),
    help="Encode frames into MP4 at this path.",
)
@click.option("--width", default=960, type=int, show_default=True, help="Render width.")
@click.option(
    "--height", default=540, type=int, show_default=True, help="Render height."
)
@click.option(
    "--quality",
    default="draft",
    type=click.Choice(["draft", "preview", "final"]),
    show_default=True,
)
@click.option(
    "--overview/--no-overview",
    default=True,
    show_default=True,
    help="Start with overview shot.",
)
@click.option(
    "--hold-sec",
    default=1.5,
    type=float,
    show_default=True,
    help="Seconds to hold on each target.",
)
@click.option(
    "--transition-sec",
    default=1.0,
    type=float,
    show_default=True,
    help="Seconds for transitions.",
)
@click.option(
    "--fps", default=24, type=int, show_default=True, help="Frames per second."
)
@click.option(
    "--elevation",
    default=35.0,
    type=float,
    show_default=True,
    help="Camera elevation in degrees.",
)
@click.option(
    "--radius-factor",
    default=2.0,
    type=float,
    show_default=True,
    help="Camera distance factor.",
)
@click.option(
    "--passes", "passes_str", default="beauty", help="Comma-separated pass names."
)
def showcase(
    glb: str,
    targets: tuple[str, ...],
    out_dir: str,
    video: str | None,
    width: int,
    height: int,
    quality: str,
    overview: bool,
    hold_sec: float,
    transition_sec: float,
    fps: int,
    elevation: float,
    radius_factor: float,
    passes_str: str,
) -> None:
    """Render a showcase visiting multiple targets with smooth camera transitions."""
    from blender_cli.render import RenderContext

    with _quiet():
        scene = Scene.load(glb)
        rc = RenderContext(
            resolution=(width, height),
            quality=quality,  # type: ignore[arg-type]
        )
        passes = [p.strip() for p in passes_str.split(",")]
        frames = rc.showcase(
            scene,
            targets=list(targets),
            out_dir=out_dir,
            overview=overview,
            hold_sec=hold_sec,
            transition_sec=transition_sec,
            fps=fps,
            elevation=elevation,
            radius_factor=radius_factor,
            video=video,
            passes=passes,
        )
        ctx = _scene_context(scene)
    result: dict[str, object] = {
        "status": "ok",
        "action": "render_showcase",
        "frame_count": len(frames),
        "out_dir": out_dir,
        "context": ctx,
    }
    if video:
        result["video"] = video
    click.echo(json.dumps(result, indent=2))


@render.command("decomposition")
@click.option("--glb", required=True, type=click.Path(exists=True))
@click.option("--preset", default="top", show_default=True)
@click.option(
    "--color-by",
    default="entity",
    type=click.Choice(["entity", "tag"]),
    show_default=True,
    help="Colour strategy: unique per entity or per tag group.",
)
@click.option(
    "--hide-tag", "hide_tags", multiple=True, help="Temporarily hide matching tags."
)
@click.option(
    "--width", default=1920, type=int, show_default=True, help="Render width."
)
@click.option(
    "--height", default=1080, type=int, show_default=True, help="Render height."
)
@click.option("--out", required=True, type=click.Path())
def decomposition(
    glb: str,
    preset: str,
    color_by: str,
    hide_tags: tuple[str, ...],
    width: int,
    height: int,
    out: str,
) -> None:
    """Render flat-colour decomposition — one unique colour per entity or tag."""
    from blender_cli.render import RenderContext

    with _quiet():
        scene = Scene.load(glb)
        # Hide tags before decomposition render
        hidden_objs: list = []
        if hide_tags:
            for obj in scene.bpy_scene.collection.all_objects:
                from blender_cli.scene import Scene as _S
                obj_tags = _S.tags(obj)
                if obj_tags & set(hide_tags):
                    obj.hide_render = True
                    obj.hide_viewport = True
                    hidden_objs.append(obj)
        rc = RenderContext(resolution=(width, height))
        rc.decomposition(
            scene,
            out=out,
            preset=preset,
            color_by=color_by,  # type: ignore[arg-type]
        )
        # Restore hidden
        for obj in hidden_objs:
            obj.hide_render = False
            obj.hide_viewport = False
        ctx = _scene_context(scene)
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "render_decomposition",
                "output": out,
                "color_by": color_by,
                "resolution": [width, height],
                "context": ctx,
            },
            indent=2,
        )
    )


# Register render settings subcommands
from blender_cli.cli.commands.render_settings_cmd import (
    render_info_cmd,
    render_presets_cmd,
    render_settings_cmd,
)

render.add_command(render_settings_cmd, "settings")
render.add_command(render_presets_cmd, "presets")
render.add_command(render_info_cmd, "info")
