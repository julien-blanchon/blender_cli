"""
RenderContext — camera presets, multi-pass still & animated rendering.

Uses Blender 5.0 compositor API: ``compositing_node_group`` with
``NodeGroupOutput`` to route individual render passes to PNG output.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import bpy
import mathutils

from blender_cli.render.camera import Camera, _look_at_bpy
from blender_cli.render.camera_path import (
    CameraPath,
    _cr_eval_vec3,
    _smoothstep,
)
from blender_cli.types import RenderSpec, Vec3

if TYPE_CHECKING:
    from blender_cli.geometry.spline import Spline
    from blender_cli.scene.scene import Scene
    from blender_cli.scene.selection import Selection
    from blender_cli.scene.selection import Selection as Sel

# Highlight defaults
_HIGHLIGHT_COLOR = (1.0, 0.3, 0.0, 1.0)  # bright orange
_GHOST_COLOR = (0.5, 0.5, 0.5, 1.0)  # neutral grey


def _mesh_data(obj: bpy.types.Object) -> bpy.types.Mesh:
    """
    Return ``obj.data`` cast to :class:`bpy.types.Mesh`.

    Raises :class:`RuntimeError` if the object has no data or data is not a Mesh.
    """
    data = obj.data
    if not isinstance(data, bpy.types.Mesh):
        msg = f"Expected Mesh data on {obj.name!r}, got {type(data).__name__}"
        raise RuntimeError(msg)
    return data


def bundled_hdri(name: str = "forest.exr") -> Path | None:
    """
    Resolve a Blender-bundled studio light HDRI by filename.

    Returns the path if found, or ``None`` if the file does not exist.
    Common names: ``"forest.exr"``, ``"city.exr"``, ``"courtyard.exr"``.
    """
    pkg_dir = Path(bpy.__file__).resolve().parent.parent.parent.parent
    path = pkg_dir / "datafiles" / "studiolights" / "world" / name
    return path if path.exists() else None


# pass name → (view_layer enable attribute, Render Layers output socket)
_PASS_CFG: dict[str, tuple[str, str]] = {
    "depth": ("use_pass_z", "Depth"),
    "normal": ("use_pass_normal", "Normal"),
    "object_id": ("use_pass_object_index", "Object Index"),
    "instance_id": ("use_pass_object_index", "Object Index"),
    "albedo": ("use_pass_diffuse_color", "DiffCol"),
}

_HDRI_PREFIX = "_mc_hdri_"


@contextlib.contextmanager
def _temporary_visibility(
    scene: "Scene",
    hide_tags: set[str] | None = None,
    show_tags: set[str] | None = None,
    hide_where: str | None = None,
):
    """Temporarily override object visibility for a single render."""
    saved: dict[int, tuple[bool, bool]] = {}
    hide_set = set(hide_tags or set())
    show_set = set(show_tags or set())
    hide_ids: set[int] = set()
    if hide_where:
        hide_ids = {id(o) for o in scene.select(hide_where)}

    try:
        for obj in scene.objects():
            saved[id(obj)] = (bool(obj.hide_render), bool(obj.hide_viewport))
            tags = scene.tags(obj)
            should_hide = False
            if hide_set and tags & hide_set:
                should_hide = True
            if hide_ids and id(obj) in hide_ids:
                should_hide = True
            if show_set and tags & show_set:
                should_hide = False
            if should_hide:
                obj.hide_render = True
                obj.hide_viewport = True
        yield
    finally:
        for obj in scene.objects():
            state = saved.get(id(obj))
            if state is None:
                continue
            obj.hide_render = state[0]
            obj.hide_viewport = state[1]


def _world_has_background(bs: bpy.types.Scene) -> bool:
    """Check if the world has an active background (HDRI or color node)."""
    if bs.world is None:
        return False
    tree = bs.world.node_tree  # type: ignore[union-attr]
    if tree is None:
        return False
    # Check for HDRI environment texture or any Background shader node
    return any(
        n.name.startswith(_HDRI_PREFIX) or n.type == "BACKGROUND"
        for n in tree.nodes
    )


def setup_hdri_world(
    bs: bpy.types.Scene,
    path: str | Path,
    rotation: float = 0.0,
    strength: float = 1.0,
) -> None:
    """
    Wire an HDRI environment texture into the world background.

    Creates: TexCoord → Mapping → EnvironmentTexture → Background → WorldOutput.
    Idempotent — calling twice replaces the previous HDRI, doesn't stack.
    """
    resolved = Path(path).resolve()
    if not resolved.is_file():
        msg = f"HDRI file not found: {resolved}"
        raise FileNotFoundError(msg)

    if bs.world is None:
        bs.world = bpy.data.worlds.new("render_world")
    world = bs.world
    if world is None:
        msg = "Blender scene has no world"
        raise RuntimeError(msg)
    world.use_nodes = True  # type: ignore[attr-defined]
    tree = world.node_tree  # type: ignore[union-attr]
    if tree is None:
        msg = "World has no node tree"
        raise RuntimeError(msg)

    # Remove old HDRI nodes (idempotent)
    for node in list(tree.nodes):
        if node.name.startswith(_HDRI_PREFIX):
            tree.nodes.remove(node)

    # Reuse existing Background / World Output or create
    bg = tree.nodes.get("Background")
    if bg is None:
        bg = tree.nodes.new("ShaderNodeBackground")
    world_out = tree.nodes.get("World Output")
    if world_out is None:
        world_out = tree.nodes.new("ShaderNodeOutputWorld")

    # HDRI node chain
    tex_coord = tree.nodes.new("ShaderNodeTexCoord")
    tex_coord.name = f"{_HDRI_PREFIX}texcoord"

    mapping = tree.nodes.new("ShaderNodeMapping")
    mapping.name = f"{_HDRI_PREFIX}mapping"
    mapping.inputs["Rotation"].default_value[2] = rotation  # type: ignore[index]

    env_tex = tree.nodes.new("ShaderNodeTexEnvironment")
    env_tex.name = f"{_HDRI_PREFIX}envtex"

    # Load image (reuse if already loaded)
    img: bpy.types.Image | None = None
    for existing in bpy.data.images:
        if existing.filepath and Path(existing.filepath).resolve() == resolved:
            img = existing
            break
    if img is None:
        img = bpy.data.images.load(str(resolved))
    env_tex.image = img

    bg.inputs["Strength"].default_value = strength  # type: ignore[index]

    # Wire: TexCoord → Mapping → EnvTex → Background → WorldOutput
    tree.links.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
    tree.links.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
    tree.links.new(env_tex.outputs["Color"], bg.inputs["Color"])
    tree.links.new(bg.outputs["Background"], world_out.inputs["Surface"])


class RenderContext:
    """
    Multi-pass rendering with named camera presets.

    Presets: ``top`` (orthographic), ``iso`` (isometric 45°),
    ``iso_close`` (tight isometric).

    Passes: ``beauty``, ``depth``, ``normal``, ``object_id``, ``instance_id``.
    """

    __slots__ = (
        "_hdri_path",
        "_hdri_rotation",
        "_hdri_strength",
        "_quality",
        "_resolution",
    )

    def __init__(
        self,
        resolution: tuple[int, int] = (1920, 1080),
        quality: Literal["draft", "preview", "final"] = "draft",
    ) -> None:
        self._resolution = resolution
        self._quality: Literal["draft", "preview", "final"] = quality
        self._hdri_path: Path | None = None
        self._hdri_rotation: float = 0.0
        self._hdri_strength: float = 1.0

    def set_hdri(
        self,
        path: str | Path,
        rotation: float = 0.0,
        strength: float = 1.0,
    ) -> None:
        """
        Configure HDRI environment for all renders from this context.

        When active, ``film_transparent`` is disabled so the HDRI sky is visible.
        """
        p = Path(path).resolve()
        if not p.is_file():
            msg = f"HDRI file not found: {p}"
            raise FileNotFoundError(msg)
        self._hdri_path = p
        self._hdri_rotation = rotation
        self._hdri_strength = strength

    def still(
        self,
        scene: Scene,
        preset: str = "top",
        out: str | Path = "render.png",
        passes: list[str] | None = None,
        visibility_profile: str | None = None,
        mode: Literal["solid", "wireframe"] = "solid",
        highlight: Selection | None = None,
        highlight_where: str | None = None,
        ghost_opacity: float = 0.3,
        hide_tags: set[str] | list[str] | None = None,
        show_tags: set[str] | list[str] | None = None,
        hide_where: str | None = None,
        camera: Camera | None = None,
    ) -> None:
        """
        Render the full scene from a named camera preset.

        Parameters
        ----------
        mode:
            ``'solid'`` (default) or ``'wireframe'``.  Wireframe uses the
            BLENDER_WORKBENCH engine with wireframe shading.
        highlight:
            If given, a :class:`Selection` of objects to highlight.  Highlighted
            objects are rendered normally; all others are dimmed/ghosted.
        ghost_opacity:
            Alpha of non-highlighted objects when *highlight* is used (0–1).
        visibility_profile:
            Named profile to apply before rendering (restored after).

        """
        if highlight is None and highlight_where is not None:
            highlight = scene.select(highlight_where)

        with _temporary_visibility(
            scene,
            hide_tags=set(hide_tags or []),
            show_tags=set(show_tags or []),
            hide_where=hide_where,
        ):
            if visibility_profile:
                scene.apply_visibility_profile(visibility_profile)
            try:
                bbox = scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
                if mode == "wireframe":
                    self._do_render_wireframe(
                        scene,
                        bbox,
                        preset,
                        Path(out),
                    )
                elif highlight is not None:
                    self._do_render_highlight(
                        scene,
                        bbox,
                        preset,
                        Path(out),
                        passes or ["beauty"],
                        highlight,
                        ghost_opacity,
                    )
                else:
                    self._do_render(
                        scene,
                        bbox,
                        preset,
                        Path(out),
                        passes or ["beauty"],
                        camera=camera,
                    )
            finally:
                if visibility_profile:
                    scene.clear_visibility_profile()

    def batch(
        self,
        scene: Scene,
        renders: list[RenderSpec],
        out_dir: str | Path = ".",
    ) -> list[Path]:
        """
        Render multiple shots in one call.

        Each entry in *renders* is a dict with:

        - ``type``: ``"still"`` (default) or ``"focus"``
        - ``out``: output filename (relative to *out_dir*)
        - any other keys forwarded to :meth:`still` or :meth:`focus`
          (e.g. ``preset``, ``passes``, ``mode``, ``highlight``,
          ``visibility_profile``, ``target``)

        Returns a list of resolved output paths.
        """
        out_base = Path(out_dir)
        out_base.mkdir(parents=True, exist_ok=True)
        results: list[Path] = []
        for spec in renders:
            spec_d: dict[str, Any] = dict(spec)  # shallow copy so we can pop
            render_type = str(spec_d.pop("type", "still"))
            out_file = out_base / str(spec_d.pop("out", "render.png"))
            if render_type == "focus":
                where_expr = spec_d.pop("where", None)
                if where_expr is not None and "target" not in spec_d:
                    spec_d["target"] = scene.select(str(where_expr))
                self.focus(scene, out=out_file, **spec_d)
            else:
                self.still(scene, out=out_file, **spec_d)
            results.append(out_file.resolve())
        return results

    def focus(
        self,
        scene: Scene,
        target: Selection | None = None,
        preset: str = "iso_close",
        out: str | Path = "render.png",
        passes: list[str] | None = None,
        visibility_profile: str | None = None,
        where: str | None = None,
        hide_tags: set[str] | list[str] | None = None,
        show_tags: set[str] | list[str] | None = None,
        hide_where: str | None = None,
        camera: Camera | None = None,
    ) -> None:
        """Render focused on the bounding box of *target*."""
        if target is None:
            if where is None:
                msg = "focus() requires target or where"
                raise ValueError(msg)
            target = scene.select(where)
        with _temporary_visibility(
            scene,
            hide_tags=set(hide_tags or []),
            show_tags=set(show_tags or []),
            hide_where=hide_where,
        ):
            if visibility_profile:
                scene.apply_visibility_profile(visibility_profile)
            try:
                bbox = _sel_bbox(target)
                self._do_render(
                    scene,
                    bbox,
                    preset,
                    Path(out),
                    passes or ["beauty"],
                    camera=camera,
                )
            finally:
                if visibility_profile:
                    scene.clear_visibility_profile()

    def turntable(
        self,
        scene: Scene,
        target: Selection,
        frames: int = 12,
        out_dir: str | Path = "turntable",
        passes: list[str] | None = None,
        *,
        elevation: float = 35.0,
        radius: float | None = None,
        camera: Camera | None = None,
    ) -> None:
        """
        Render a 360° orbit around *target*.

        Outputs ``frame_001.png``, ``frame_002.png``, … in *out_dir*.

        Parameters
        ----------
        elevation:
            Camera elevation in degrees (default 35°).
        radius:
            Orbit radius in scene units. ``None`` = auto 1.5× diagonal.
        camera:
            Optional :class:`Camera` to use. If not given, a temporary camera
            is created.

        Combine frames into video with:
        ``ffmpeg -framerate 24 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p out.mp4``

        """
        bbox = _sel_bbox(target)
        lo, hi = bbox
        center = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
        orbit_radius = radius if radius is not None else diag * 1.5

        path = CameraPath.orbit(
            center, orbit_radius, elevation=elevation, frames=frames
        )
        self.animate(
            scene,
            path,
            camera=camera,
            out_dir=out_dir,
            passes=passes,
            _diag=diag,
            _bbox=bbox,
        )

    def orbit_grid(
        self,
        scene: Scene,
        target: Selection,
        elevations: list[float] | None = None,
        azimuths: int = 8,
        out_dir: str | Path = "orbit_grid",
        passes: list[str] | None = None,
        *,
        radius: float | None = None,
        video: str | Path | None = None,
        fps: int = 6,
        camera: Camera | None = None,
    ) -> None:
        """
        Render a grid of views at multiple elevations and azimuths.

        Outputs ``elev{DD}_az{NNN}.png`` in *out_dir*.

        Parameters
        ----------
        elevations:
            List of elevation angles in degrees (default ``[15, 35, 60]``).
        azimuths:
            Number of azimuth steps around the full 360° circle.
        radius:
            Orbit radius in scene units. ``None`` = auto 1.5× diagonal.
        video:
            If given, encode all frames into an MP4 at this path using ffmpeg.
            Frames are ordered by elevation then azimuth.
        fps:
            Frame rate for the video (default 6).
        camera:
            Optional :class:`Camera` to use.

        """
        if elevations is None:
            elevations = [15.0, 35.0, 60.0]

        bbox = _sel_bbox(target)
        lo, hi = bbox
        center = Vec3((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2)
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
        orbit_radius = radius if radius is not None else diag * 1.5

        path = CameraPath.orbit_grid(
            center, orbit_radius, elevations=elevations, azimuths=azimuths
        )

        # Custom naming: elev{DD}_az{NNN}.png
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)
        passes = passes or ["beauty"]

        self._setup_render(scene, bbox, passes)
        if camera is not None:
            camera._activate(scene.bpy_scene)
            cam_obj = camera.bpy_object
        else:
            cam_obj = self._make_reusable_camera(scene.bpy_scene, diag)

        frame_paths: list[Path] = []
        frame_idx = 0
        for elev_deg in elevations:
            for ai in range(azimuths):
                azim_deg = round(math.degrees(2 * math.pi * ai / azimuths))
                kf = path[frame_idx]
                cam_obj.location = (kf.position.x, kf.position.y, kf.position.z)
                _look_at_bpy(cam_obj, kf.look_at)
                frame_out = out_path / f"elev{int(elev_deg):02d}_az{azim_deg:03d}.png"
                self._render_frame(scene.bpy_scene, frame_out, passes)
                frame_paths.append(frame_out)
                frame_idx += 1

        if video is not None:
            _encode_video(frame_paths, Path(video).resolve(), fps=fps)

    def flythrough(
        self,
        scene: Scene,
        path_spline: Spline,
        frames: int = 24,
        out_dir: str | Path = "flythrough",
        look: Literal["ahead", "target"] = "ahead",
        look_target: Vec3 | Selection | None = None,
        passes: list[str] | None = None,
        camera: Camera | None = None,
    ) -> None:
        """
        Render camera following *path_spline*.

        *look*='ahead' points camera along the spline tangent;
        *look*='target' points at *look_target* (a Vec3 or Selection center).

        Outputs ``frame_001.png``, ``frame_002.png``, … in *out_dir*.

        Parameters
        ----------
        camera:
            Optional :class:`Camera` to use.

        Combine frames into video with:
        ``ffmpeg -framerate 24 -i frame_%03d.png -c:v libx264 -pix_fmt yuv420p out.mp4``

        """
        from blender_cli.scene.selection import Selection as Sel

        bbox = scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
        lo, hi = bbox
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))

        # Resolve look target for 'target' mode
        target_pos: Vec3 | None = None
        if look == "target":
            if isinstance(look_target, Sel):
                tbox = _sel_bbox(look_target)
                tlo, thi = tbox
                target_pos = Vec3(
                    (tlo.x + thi.x) / 2, (tlo.y + thi.y) / 2, (tlo.z + thi.z) / 2
                )
            elif isinstance(look_target, Vec3):
                target_pos = look_target
            else:
                # Default: scene center
                target_pos = Vec3(
                    (lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2
                )

        cam_path = CameraPath.from_spline(
            path_spline,
            frames=frames,
            look=look,
            look_target=target_pos,
        )
        self.animate(
            scene,
            cam_path,
            camera=camera,
            out_dir=out_dir,
            passes=passes,
            _diag=diag,
            _bbox=bbox,
        )

    # ---- public: showcase ----

    def showcase(
        self,
        scene: Scene,
        targets: list[Selection | str],
        out_dir: str | Path = "showcase",
        *,
        overview: bool = True,
        hold_sec: float = 1.5,
        transition_sec: float = 1.0,
        fps: int = 24,
        elevation: float = 35.0,
        radius_factor: float = 2.0,
        video: str | Path | None = None,
        passes: list[str] | None = None,
        candidates: int = 12,
        occlusion_samples: int = 5,
        camera: Camera | None = None,
    ) -> list[Path]:
        """
        Smart camera showcase visiting multiple targets with smooth transitions.

        Automatically selects occlusion-aware viewpoints for each target,
        orders them via nearest-neighbor to minimise camera travel, and
        interpolates the camera path with Catmull-Rom splines + smoothstep
        easing.

        Parameters
        ----------
        targets:
            Focus targets — either :class:`Selection` objects or query strings
            passed to ``scene.select()``.  Empty selections are skipped.
        overview:
            Start with an establishing shot of the whole scene.
        hold_sec:
            Seconds to hold the camera on each target.
        transition_sec:
            Seconds for the camera transition between targets.
        fps:
            Frames per second (default 24).
        elevation:
            Preferred camera elevation in degrees (default 35°).
        radius_factor:
            Camera distance = target diagonal × factor.
        video:
            If given, encode all frames into an MP4 at this path.
        passes:
            Render passes (default ``['beauty']``).
        candidates:
            Number of viewpoint candidates per target for scoring.
        occlusion_samples:
            Number of rays per candidate for occlusion checking.

        Returns
        -------
        list[Path]
            Rendered frame paths, or ``[]`` if no valid targets.

        """
        # Step 1 — Resolve targets
        resolved: list[tuple[Sel, str]] = []
        for t in targets:
            if isinstance(t, str):
                sel = scene.select(t)
                label = t
            else:
                sel = t
                label = f"selection({len(t)})"
            if len(sel) > 0:
                resolved.append((sel, label))

        if not resolved:
            return []

        scene_bbox = scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
        lo_s, hi_s = scene_bbox
        scene_size = Vec3(hi_s.x - lo_s.x, hi_s.y - lo_s.y, hi_s.z - lo_s.z)
        scene_diag = max(
            1.0,
            math.sqrt(
                scene_size.x**2 + scene_size.y**2 + scene_size.z**2,
            ),
        )

        hold_frames = max(1, round(hold_sec * fps))
        trans_frames = max(1, round(transition_sec * fps))
        passes = passes or ["beauty"]

        # Step 2 — Score viewpoints per target
        target_kfs: list[_Keyframe] = []
        for sel, label in resolved:
            bbox = _sel_bbox(sel)
            tlo, thi = bbox
            center = Vec3(
                (tlo.x + thi.x) * 0.5,
                (tlo.y + thi.y) * 0.5,
                (tlo.z + thi.z) * 0.5,
            )
            size = Vec3(thi.x - tlo.x, thi.y - tlo.y, thi.z - tlo.z)
            diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
            radius = diag * radius_factor
            obj_ids = {id(o) for o in sel}

            vp = _best_viewpoint(
                center,
                diag,
                radius,
                elevation,
                candidates,
                occlusion_samples,
                obj_ids,
                scene.bpy_scene,
                bbox,
            )
            target_kfs.append(
                _Keyframe(
                    pos=vp.pos,
                    look_at=center,
                    hold_frames=hold_frames,
                    label=label,
                    target_objs=list(sel),
                )
            )

        # Step 3 — Order keyframes
        overview_kf: _Keyframe | None = None
        if overview:
            overview_kf = _make_overview_keyframe(
                scene_bbox,
                hold_frames,
                elevation,
                radius_factor,
            )
        keyframes = _order_keyframes(overview_kf, target_kfs)

        if not keyframes:
            return []

        # Step 4 — Interpolate camera path
        frame_data = _interpolate_showcase_path(keyframes, trans_frames)

        # Step 5 — Render
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        self._setup_render(scene, scene_bbox, passes)
        if camera is not None:
            camera._activate(scene.bpy_scene)
            cam_obj = camera.bpy_object
        else:
            cam_obj = self._make_reusable_camera(scene.bpy_scene, scene_diag)

        frame_paths: list[Path] = []
        for i, (pos, look) in enumerate(frame_data):
            cam_obj.location = (pos.x, pos.y, pos.z)
            _look_at_bpy(cam_obj, look)
            frame_out = out_path / f"frame_{i + 1:03d}.png"
            self._render_frame(scene.bpy_scene, frame_out, passes)
            frame_paths.append(frame_out)

        if video is not None and frame_paths:
            _encode_video(frame_paths, Path(video).resolve(), fps=fps)

        return frame_paths

    # ---- public: animate (unified animation) ----

    def animate(
        self,
        scene: Scene,
        path: CameraPath,
        *,
        camera: Camera | None = None,
        out_dir: str | Path = "animation",
        passes: list[str] | None = None,
        video: str | Path | None = None,
        fps: int = 24,
        _diag: float | None = None,
        _bbox: tuple[Vec3, Vec3] | None = None,
    ) -> list[Path]:
        """
        Unified animation: render one frame per :class:`CameraPath` keyframe.

        Parameters
        ----------
        scene:
            The scene to render.
        path:
            Pre-computed camera path.
        camera:
            Optional :class:`Camera` to reuse. If not given, a temporary
            camera is created.
        out_dir:
            Output directory for frame images.
        passes:
            Render passes (default ``['beauty']``).
        video:
            If given, encode all frames into an MP4 at this path.
        fps:
            Frame rate for the video (default 24).

        Returns
        -------
        list[Path]
            Rendered frame paths.

        """
        bbox = _bbox or scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
        lo, hi = bbox
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = _diag or max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
        passes = passes or ["beauty"]

        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        self._setup_render(scene, bbox, passes)
        if camera is not None:
            camera._activate(scene.bpy_scene)
            cam_obj = camera.bpy_object
        else:
            cam_obj = self._make_reusable_camera(scene.bpy_scene, diag)

        frame_paths: list[Path] = []
        for i, kf in enumerate(path.frames):
            cam_obj.location = (kf.position.x, kf.position.y, kf.position.z)
            _look_at_bpy(cam_obj, kf.look_at)
            frame_out = out_path / f"frame_{i + 1:03d}.png"
            self._render_frame(scene.bpy_scene, frame_out, passes)
            frame_paths.append(frame_out)

        if video is not None and frame_paths:
            _encode_video(frame_paths, Path(video).resolve(), fps=fps)

        return frame_paths

    # ---- public: debug overlays ----

    def decomposition(
        self,
        scene: Scene,
        out: str | Path = "decomposition.png",
        preset: str = "top",
        *,
        color_by: Literal["entity", "tag"] = "entity",
    ) -> None:
        """
        Render unique flat color per entity (or per tag group).

        Saves original materials, creates temporary emission materials with
        HSV-distributed colours, renders, then restores everything.
        """
        bbox = scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
        objs = [o for o in scene.objects() if o.type == "MESH"]
        if not objs:
            self._do_render(scene, bbox, preset, Path(out), ["beauty"])
            return

        # Assign colour indices
        if color_by == "tag":
            tag_map: dict[str, int] = {}
            idx = 0
            for obj in objs:
                key = ",".join(sorted(scene.tags(obj))) or obj.name
                if key not in tag_map:
                    tag_map[key] = idx
                    idx += 1
            n_colors = max(len(tag_map), 1)
            obj_indices = [
                tag_map.get(",".join(sorted(scene.tags(o))) or o.name, 0) for o in objs
            ]
        else:
            n_colors = len(objs)
            obj_indices = list(range(n_colors))

        # Save originals + create temporary emission materials
        saved: list[tuple[bpy.types.Object, list[bpy.types.Material | None]]] = []
        temp_mats: list[bpy.types.Material] = []
        for i, obj in enumerate(objs):
            orig = [slot.material for slot in obj.material_slots]
            saved.append((obj, orig))

            hue = obj_indices[i] / max(n_colors, 1)
            r, g, b = _hsv_to_rgb(hue, 0.85, 0.95)
            mat = bpy.data.materials.new(f"_mc_decomp_{i}")
            tree = mat.node_tree
            if tree is not None:
                tree.nodes.clear()
                output = tree.nodes.new("ShaderNodeOutputMaterial")
                emit = tree.nodes.new("ShaderNodeEmission")
                emit.inputs["Color"].default_value = (r, g, b, 1.0)  # type: ignore[index]
                emit.inputs["Strength"].default_value = 1.0  # type: ignore[index]
                tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
            temp_mats.append(mat)

            mesh = _mesh_data(obj)
            mesh.materials.clear()
            mesh.materials.append(mat)

        try:
            self._do_render(
                scene,
                bbox,
                preset,
                Path(out),
                ["beauty"],
                force_opaque=True,
                world_color=(0.0, 0.0, 0.0, 1.0),
            )
        finally:
            for obj, orig in saved:
                mesh = _mesh_data(obj)
                mesh.materials.clear()
                for m in orig:
                    mesh.materials.append(m)
            for mat in temp_mats:
                bpy.data.materials.remove(mat)

    def debug_render(
        self,
        scene: Scene,
        out: str | Path = "debug.png",
        preset: str = "top",
        *,
        mode: Literal["height", "slope"] = "height",
    ) -> None:
        """
        Render vertex-colour visualisation (height gradient or slope map).

        Creates temporary vertex colour layers and emission materials,
        renders, then cleans up everything.
        """
        bbox = scene.bbox() or (Vec3(-10, -10, 0), Vec3(10, 10, 5))
        lo, hi = bbox
        z_range = max(hi.z - lo.z, 1e-6)
        objs = [o for o in scene.objects() if o.type == "MESH"]
        if not objs:
            self._do_render(scene, bbox, preset, Path(out), ["beauty"])
            return

        saved: list[tuple[bpy.types.Object, list[bpy.types.Material | None]]] = []
        temp_mats: list[bpy.types.Material] = []
        added_attrs: list[tuple[bpy.types.Mesh, str]] = []

        try:
            for obj in objs:
                mesh = obj.data
                if not isinstance(mesh, bpy.types.Mesh):
                    continue

                orig = [slot.material for slot in obj.material_slots]
                saved.append((obj, orig))

                # Create vertex colour layer
                attr_name = "_mc_debug_vcol"
                vcol = mesh.color_attributes.new(attr_name, "BYTE_COLOR", "CORNER")
                added_attrs.append((mesh, attr_name))

                # Fill colours per loop corner
                mw = obj.matrix_world
                for poly in mesh.polygons:
                    for li in poly.loop_indices:
                        loop = mesh.loops[li]
                        vi = loop.vertex_index
                        world_co = mw @ mesh.vertices[vi].co

                        if mode == "slope":
                            # Slope: angle of face normal vs. up vector
                            normal = poly.normal
                            up_dot = abs(normal.z)
                            t = 1.0 - up_dot  # 0=flat, 1=vertical
                        else:
                            # Height: normalised Z position
                            t = (world_co.z - lo.z) / z_range

                        t = max(0.0, min(1.0, t))
                        r, g, b = _colormap(t)
                        vcol.data[li].color = (r, g, b, 1.0)  # type: ignore[index]

                # Emission material reading vertex colour
                mat = bpy.data.materials.new(f"_mc_debug_{obj.name}")
                mat.use_nodes = True  # type: ignore[attr-defined]
                tree = mat.node_tree
                if tree is not None:
                    tree.nodes.clear()
                    output = tree.nodes.new("ShaderNodeOutputMaterial")
                    emit = tree.nodes.new("ShaderNodeEmission")
                    attr_node = tree.nodes.new("ShaderNodeAttribute")
                    attr_node.attribute_name = attr_name  # type: ignore[attr-defined]
                    tree.links.new(attr_node.outputs["Color"], emit.inputs["Color"])
                    emit.inputs["Strength"].default_value = 1.0  # type: ignore[index]
                    tree.links.new(emit.outputs["Emission"], output.inputs["Surface"])
                temp_mats.append(mat)

                mesh = _mesh_data(obj)
                mesh.materials.clear()
                mesh.materials.append(mat)

            self._do_render(
                scene,
                bbox,
                preset,
                Path(out),
                ["beauty"],
                force_opaque=True,
                world_color=(0.0, 0.0, 0.0, 1.0),
            )
        finally:
            # Restore materials
            for obj, orig in saved:
                mesh = _mesh_data(obj)
                mesh.materials.clear()
                for m in orig:
                    mesh.materials.append(m)
            # Remove temp vertex colour layers
            for mesh, attr_name in added_attrs:
                attr = mesh.color_attributes.get(attr_name)
                if attr is not None:
                    mesh.color_attributes.remove(attr)
            # Remove temp materials
            for mat in temp_mats:
                bpy.data.materials.remove(mat)

    # ---- internal: debug modes ----

    def _do_render_wireframe(
        self,
        scene: Scene,
        bbox: tuple[Vec3, Vec3],
        preset: str,
        out: Path,
    ) -> None:
        """
        Render scene as wireframe via WireframeModifier + emission material.

        Adds a temporary wireframe modifier (``use_replace=True``) and a black
        emission material to every mesh, renders with EEVEE on a white
        background, then removes modifiers and restores original materials.
        """
        # Create wireframe emission material (black lines, no lighting needed)
        wire_mat = bpy.data.materials.new("_mc_wireframe")
        tree = wire_mat.node_tree
        if tree is None:
            msg = "Material has no node tree"
            raise RuntimeError(msg)
        nodes = tree.nodes
        links = tree.links
        nodes.clear()
        output = nodes.new("ShaderNodeOutputMaterial")
        emit = nodes.new("ShaderNodeEmission")
        emit.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)  # type: ignore[index]
        emit.inputs["Strength"].default_value = 1.0  # type: ignore[index]
        links.new(emit.outputs["Emission"], output.inputs["Surface"])

        # Compute wireframe thickness from scene scale
        lo, hi = bbox
        size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
        diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
        thickness = max(0.005, diag * 0.001)  # ~0.1% of scene diagonal, min 5mm

        # Add modifier + swap material on each mesh
        added: list[
            tuple[bpy.types.Object, bpy.types.Modifier, list[bpy.types.Material | None]]
        ] = []
        for obj in scene.objects():
            if obj.type != "MESH":
                continue
            orig_mats = [slot.material for slot in obj.material_slots]
            mod = obj.modifiers.new(name="_mc_wireframe", type="WIREFRAME")
            mod.thickness = thickness  # type: ignore[attr-defined]
            mod.use_replace = True  # type: ignore[attr-defined]
            mod.use_even_offset = True  # type: ignore[attr-defined]
            mesh = _mesh_data(obj)
            mesh.materials.clear()
            mesh.materials.append(wire_mat)
            added.append((obj, mod, orig_mats))

        try:
            self._do_render(
                scene,
                bbox,
                preset,
                out,
                ["beauty"],
                force_opaque=True,
                world_color=(10.0, 10.0, 10.0, 1.0),
            )
        finally:
            for obj, mod, orig_mats in added:
                obj.modifiers.remove(mod)
                mesh = _mesh_data(obj)
                mesh.materials.clear()
                for m in orig_mats:
                    mesh.materials.append(m)
            bpy.data.materials.remove(wire_mat)

    def _do_render_highlight(
        self,
        scene: Scene,
        bbox: tuple[Vec3, Vec3],
        preset: str,
        out: Path,
        passes: list[str],
        highlight: Selection,
        ghost_opacity: float,
    ) -> None:
        """Render with highlighted objects bright, rest dimmed/ghosted."""
        # Build set of highlighted object names for fast lookup
        hl_set: set[int] = set()
        hl_set.update(id(obj) for obj in highlight)

        # Save original materials and apply ghost material to non-highlighted
        saved_mats: list[tuple[bpy.types.Object, list[bpy.types.Material | None]]] = []
        ghost_mat = bpy.data.materials.new("_mc_ghost")
        ghost_mat.blend_method = "BLEND"  # type: ignore[attr-defined]
        bsdf = (
            ghost_mat.node_tree.nodes.get("Principled BSDF")
            if ghost_mat.node_tree is not None
            else None
        )
        if bsdf is not None:
            bsdf.inputs["Base Color"].default_value = _GHOST_COLOR  # type: ignore[index]
            bsdf.inputs["Alpha"].default_value = ghost_opacity  # type: ignore[index]

        for obj in scene.objects():
            if id(obj) not in hl_set and obj.type == "MESH":
                orig = [slot.material for slot in obj.material_slots]
                saved_mats.append((obj, orig))
                mesh = _mesh_data(obj)
                mesh.materials.clear()
                mesh.materials.append(ghost_mat)

        try:
            self._do_render(scene, bbox, preset, out, passes)
        finally:
            # Restore original materials
            for obj, mats in saved_mats:
                mesh = _mesh_data(obj)
                mesh.materials.clear()
                for m in mats:
                    mesh.materials.append(m)
            bpy.data.materials.remove(ghost_mat)

    # ---- internal ----

    def _make_reusable_camera(
        self, bs: bpy.types.Scene, diag: float
    ) -> bpy.types.Object:
        """Create a perspective camera for animated renders."""
        cam = Camera.perspective(clip_end=diag * 10, name="render_cam")
        cam._activate(bs)
        return cam.bpy_object

    def _setup_render(
        self,
        scene: Scene,
        bbox: tuple[Vec3, Vec3],
        passes: list[str],
    ) -> None:
        """Configure render settings, lights, and pass enables (once per animation)."""
        bs = scene.bpy_scene
        rs = bs.render
        if rs is None:
            msg = "Scene has no render settings"
            raise RuntimeError(msg)
        rs.engine = "BLENDER_EEVEE"
        rs.resolution_x, rs.resolution_y = self._resolution
        ims = rs.image_settings
        if ims is None:
            msg = "Scene has no image settings"
            raise RuntimeError(msg)
        ims.file_format = "PNG"
        ims.color_mode = "RGBA"
        _ensure_light(bs)
        _apply_quality_settings(bs, self._quality)

        # Apply HDRI if configured; show sky when HDRI is active
        if self._hdri_path is not None:
            setup_hdri_world(
                bs, self._hdri_path, self._hdri_rotation, self._hdri_strength
            )
        rs.film_transparent = not (self._hdri_path is not None or _world_has_background(bs))

        vl = bs.view_layers[0]
        for p in passes:
            cfg = _PASS_CFG.get(p)
            if cfg:
                setattr(vl, cfg[0], True)

        if "object_id" in passes or "instance_id" in passes:
            for i, obj in enumerate(scene.objects(), start=1):
                obj.pass_index = i

    def _render_frame(self, bs: bpy.types.Scene, out: Path, passes: list[str]) -> None:
        """Render a single frame at the current camera position."""
        rs = bs.render
        if rs is None:
            msg = "Scene has no render settings"
            raise RuntimeError(msg)
        if "beauty" in passes:
            bs.compositing_node_group = None
            rs.filepath = str(out)
            bpy.ops.render.render(write_still=True)
        for p in passes:
            if p == "beauty":
                continue
            pass_out = out.parent / f"{out.stem}_{p}.png"
            _render_pass(bs, p, pass_out)

    def _do_render(
        self,
        scene: Scene,
        bbox: tuple[Vec3, Vec3],
        preset: str,
        out: Path,
        passes: list[str],
        force_opaque: bool = False,
        world_color: tuple[float, float, float, float] | None = None,
        camera: Camera | None = None,
    ) -> None:
        bs = scene.bpy_scene
        out = out.resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        # Render settings
        rs = bs.render
        if rs is None:
            msg = "Scene has no render settings"
            raise RuntimeError(msg)
        rs.engine = "BLENDER_EEVEE"
        rs.resolution_x, rs.resolution_y = self._resolution
        ims = rs.image_settings
        if ims is None:
            msg = "Scene has no image settings"
            raise RuntimeError(msg)
        ims.file_format = "PNG"
        ims.color_mode = "RGBA"

        # Camera + light + quality
        if camera is not None:
            camera._activate(bs)
        else:
            Camera.from_preset(preset, bbox, name="render_cam")._activate(bs)
        _ensure_light(bs)
        _apply_quality_settings(bs, self._quality)

        # Apply HDRI if configured; show sky when HDRI is active
        if self._hdri_path is not None:
            setup_hdri_world(
                bs, self._hdri_path, self._hdri_rotation, self._hdri_strength
            )
        if force_opaque:
            rs.film_transparent = False
        else:
            rs.film_transparent = not (
                self._hdri_path is not None or _world_has_background(bs)
            )

        # Override world background color if requested
        if (
            world_color is not None
            and bs.world is not None
            and bs.world.node_tree is not None
        ):
            bg = bs.world.node_tree.nodes.get("Background")
            if bg is not None:
                bg.inputs["Color"].default_value = world_color  # type: ignore[index]
                bg.inputs["Strength"].default_value = 1.0  # type: ignore[index]

        # Enable passes on view layer
        vl = bs.view_layers[0]
        for p in passes:
            cfg = _PASS_CFG.get(p)
            if cfg:
                setattr(vl, cfg[0], True)

        # Pass indices for object/instance ID
        if "object_id" in passes or "instance_id" in passes:
            for i, obj in enumerate(scene.objects(), start=1):
                obj.pass_index = i

        # Render beauty (no compositor)
        if "beauty" in passes:
            bs.compositing_node_group = None
            rs.filepath = str(out)
            bpy.ops.render.render(write_still=True)

        # Render each extra pass via compositor
        for p in passes:
            if p == "beauty":
                continue
            pass_out = out.parent / f"{out.stem}_{p}.png"
            _render_pass(bs, p, pass_out)


# ---- helpers ----


def _sel_bbox(sel: Selection) -> tuple[Vec3, Vec3]:
    """
    Axis-aligned bounding box of a Selection.

    For EMPTY objects (e.g. collection instances), recurse into children
    to find actual mesh bounds.
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    def _collect(obj: bpy.types.Object) -> None:
        if obj.type == "MESH" and obj.bound_box:
            for c in obj.bound_box:
                w = obj.matrix_world @ mathutils.Vector(c)
                xs.append(w.x)
                ys.append(w.y)
                zs.append(w.z)
        elif obj.type == "EMPTY" and obj.children:
            for child in obj.children:
                _collect(child)
        else:
            loc = obj.matrix_world.translation
            xs.append(loc.x)
            ys.append(loc.y)
            zs.append(loc.z)

    for obj in sel:
        _collect(obj)

    if not xs:
        return Vec3(-10, -10, 0), Vec3(10, 10, 5)
    return Vec3(min(xs), min(ys), min(zs)), Vec3(max(xs), max(ys), max(zs))


# ---- quality presets ----

_QUALITY_PRESETS: dict[str, dict[str, Any]] = {
    "draft": {
        "taa_render_samples": 16,
        "use_compositing": False,
        "use_sequencer": False,
        "shadow_step_count": 2,
        "use_simplify": True,
        "simplify_subdivision_render": 1,
    },
    "preview": {
        "taa_render_samples": 32,
        "use_compositing": False,
        "use_sequencer": False,
        "shadow_step_count": 4,
        "use_simplify": False,
    },
    "final": {
        "taa_render_samples": 64,
        "use_compositing": True,
        "use_sequencer": True,
        "shadow_step_count": 6,
        "use_simplify": False,
    },
}


def _apply_quality_settings(
    bs: bpy.types.Scene,
    quality: Literal["draft", "preview", "final"],
) -> None:
    """Apply EEVEE render quality preset to *bs*."""
    cfg = _QUALITY_PRESETS[quality]

    # EEVEE samples
    eevee = bs.eevee
    if eevee is not None:
        with contextlib.suppress(AttributeError):
            eevee.taa_render_samples = cfg["taa_render_samples"]
        with contextlib.suppress(AttributeError):
            eevee.shadow_step_count = cfg["shadow_step_count"]

    # Compositor / sequencer bypass
    rs = bs.render
    if rs is not None:
        with contextlib.suppress(AttributeError):
            rs.use_compositing = cfg["use_compositing"]
        with contextlib.suppress(AttributeError):
            rs.use_sequencer = cfg["use_sequencer"]

        # Simplify (reduces subdivision, particles)
        with contextlib.suppress(AttributeError):
            rs.use_simplify = cfg.get("use_simplify", False)
        if cfg.get("use_simplify"):
            with contextlib.suppress(AttributeError):
                rs.simplify_subdivision_render = cfg.get(
                    "simplify_subdivision_render",
                    1,
                )


def _ensure_light(bs: bpy.types.Scene) -> None:
    """Add a sun light, fill light, and ambient world light if the scene has none."""
    col = bs.collection
    if col is None:
        msg = "Scene has no root collection"
        raise RuntimeError(msg)
    has_light = any(obj.type == "LIGHT" for obj in col.all_objects)
    if not has_light:
        # Key light (sun) — softer than before
        light_data = bpy.data.lights.new("render_sun", "SUN")
        light_data.energy = 3.0
        light_data.angle = math.radians(5)  # slight softness
        light_obj = bpy.data.objects.new("render_sun", light_data)
        light_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
        col.objects.link(light_obj)

        # Fill light (opposite side, dimmer) to soften shadows
        fill_data = bpy.data.lights.new("render_fill", "SUN")
        fill_data.energy = 1.0
        fill_obj = bpy.data.objects.new("render_fill", fill_data)
        fill_obj.rotation_euler = (math.radians(30), 0, math.radians(-150))
        col.objects.link(fill_obj)

    # Ensure world has ambient light (raised from 0.05 for better shadow fill)
    if bs.world is None:
        bs.world = bpy.data.worlds.new("render_world")
    world = bs.world
    if world is None:
        msg = "Blender scene has no world"
        raise RuntimeError(msg)
    world.use_nodes = True  # type: ignore[attr-defined]
    if world.node_tree is None:
        return
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs[0].default_value = (0.18, 0.20, 0.25, 1.0)  # type: ignore[index]


def _rl_output(rl: bpy.types.Node, name: str) -> bpy.types.NodeSocket:
    """
    Get a Render Layers output by name using index lookup.

    Blender 5.0's Render Layers node has duplicate 'Deprecated' outputs
    which breaks ``bpy_prop_collection`` key lookup.  Fall back to index.
    """
    for i, o in enumerate(rl.outputs):
        if o.name == name:
            return rl.outputs[i]
    msg = f"Render Layers has no output {name!r}"
    raise KeyError(msg)


def _render_pass(
    bs: bpy.types.Scene,
    pass_name: str,
    out: Path,
) -> None:
    """Render a single pass by routing it through a compositor GroupOutput."""
    cfg = _PASS_CFG.get(pass_name)
    if not cfg:
        return
    _, socket_name = cfg

    tree = bpy.data.node_groups.new(f"_pass_{pass_name}", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")

    rl = tree.nodes.new("CompositorNodeRLayers")
    go = tree.nodes.new("NodeGroupOutput")

    src = _rl_output(rl, socket_name)
    if pass_name == "depth":
        # Mask depth by alpha to exclude background, then normalize
        alpha = _rl_output(rl, "Alpha")
        mul = tree.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        tree.links.new(src, mul.inputs[0])
        tree.links.new(alpha, mul.inputs[1])
        norm = tree.nodes.new("CompositorNodeNormalize")
        tree.links.new(mul.outputs[0], norm.inputs[0])
        # Invert so near=white, far=black
        inv = tree.nodes.new("CompositorNodeInvert")
        tree.links.new(norm.outputs[0], inv.inputs["Color"])
        tree.links.new(inv.outputs[0], go.inputs["Image"])
    elif pass_name == "albedo":
        # Combine DiffCol with render alpha so background stays transparent
        alpha = _rl_output(rl, "Alpha")
        set_alpha = tree.nodes.new("CompositorNodeSetAlpha")
        set_alpha.mode = "APPLY"  # type: ignore[attr-defined]
        tree.links.new(src, set_alpha.inputs["Image"])
        tree.links.new(alpha, set_alpha.inputs["Alpha"])
        tree.links.new(set_alpha.outputs["Image"], go.inputs["Image"])
    else:
        tree.links.new(src, go.inputs["Image"])

    bs.compositing_node_group = tree
    rs = bs.render
    if rs is None:
        msg = "Scene has no render settings"
        raise RuntimeError(msg)
    rs.filepath = str(out)
    bpy.ops.render.render(write_still=True)

    # Cleanup
    bs.compositing_node_group = None
    bpy.data.node_groups.remove(tree)


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    """Convert HSV (all 0-1) to RGB (all 0-1)."""
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)


def _colormap(t: float) -> tuple[float, float, float]:
    """Blue (t=0) → red (t=1) gradient via HSV hue rotation."""
    hue = 0.66 * (1.0 - t)  # 0.66=blue, 0.0=red
    return _hsv_to_rgb(hue, 0.85, 0.95)


def _encode_video(frame_paths: list[Path], out: Path, *, fps: int = 6) -> None:
    """Encode a list of PNG frames into an MP4 video using ffmpeg."""
    import subprocess
    import tempfile

    if not frame_paths:
        return

    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        encoding="utf-8", mode="w", suffix=".txt", delete=False, prefix="_mc_concat_"
    ) as f:
        concat_path = Path(f.name)
        for fp in frame_paths:
            f.write(f"file '{fp}'\n")
            f.write(f"duration {1.0 / fps:.6f}\n")

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-r",
            str(fps),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            msg = f"ffmpeg failed (exit {result.returncode}): {result.stderr[:500]}"
            raise RuntimeError(msg)
    finally:
        concat_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Showcase — smart camera helpers
# ---------------------------------------------------------------------------

_GOLDEN_RATIO = (1.0 + math.sqrt(5.0)) / 2.0


@dataclass
class _Viewpoint:
    """Candidate camera position with composite score."""

    pos: Vec3
    score: float = 0.0


@dataclass
class _Keyframe:
    """Resolved showcase keyframe."""

    pos: Vec3
    look_at: Vec3
    hold_frames: int
    label: str
    target_objs: list[Any] = field(default_factory=list)


def _bbox_sample_points(lo: Vec3, hi: Vec3, n: int) -> list[Vec3]:
    """
    Return *n* deterministic sample points inside a bounding box.

    Includes the center plus Fibonacci-distributed points within the box.
    """
    cx = (lo.x + hi.x) * 0.5
    cy = (lo.y + hi.y) * 0.5
    cz = (lo.z + hi.z) * 0.5
    if n <= 1:
        return [Vec3(cx, cy, cz)]
    sx = (hi.x - lo.x) * 0.4  # shrink to inner 80%
    sy = (hi.y - lo.y) * 0.4
    sz = (hi.z - lo.z) * 0.4
    pts = [Vec3(cx, cy, cz)]
    for i in range(1, n):
        # Distribute via golden-ratio offsets in each axis
        fx = ((i * 0.7548776662) % 1.0) * 2.0 - 1.0  # 1/golden_ratio^1
        fy = ((i * 0.5698402910) % 1.0) * 2.0 - 1.0  # 1/golden_ratio^2
        fz = ((i * 0.4256233627) % 1.0) * 2.0 - 1.0  # 1/golden_ratio^3
        pts.append(Vec3(cx + fx * sx, cy + fy * sy, cz + fz * sz))
    return pts


def _generate_candidates(
    center: Vec3, radius: float, elev_deg: float, n: int
) -> list[Vec3]:
    """
    Generate *n* camera candidate positions on a hemisphere above *center*.

    Uses Fibonacci lattice with elevation bias toward *elev_deg*.
    ~80% of samples cluster in the preferred ±20° band, ~20% at higher angles.
    """
    pts: list[Vec3] = []
    n_preferred = max(1, int(n * 0.8))
    n_high = n - n_preferred

    # Preferred band: elev_deg ± 20°
    lo_elev = max(5.0, elev_deg - 20.0)
    hi_elev = min(80.0, elev_deg + 20.0)
    for i in range(n_preferred):
        theta = 2.0 * math.pi * i / _GOLDEN_RATIO
        # Map i to elevation within the preferred band
        frac = (i + 0.5) / n_preferred
        elev = math.radians(lo_elev + frac * (hi_elev - lo_elev))
        pts.append(
            Vec3(
                center.x + radius * math.cos(elev) * math.cos(theta),
                center.y + radius * math.cos(elev) * math.sin(theta),
                center.z + radius * math.sin(elev),
            )
        )

    # High-angle fallback candidates: 60°–80°
    for i in range(n_high):
        theta = 2.0 * math.pi * (i + n_preferred) / _GOLDEN_RATIO
        frac = (i + 0.5) / max(n_high, 1)
        elev = math.radians(60.0 + frac * 20.0)
        pts.append(
            Vec3(
                center.x + radius * math.cos(elev) * math.cos(theta),
                center.y + radius * math.cos(elev) * math.sin(theta),
                center.z + radius * math.sin(elev),
            )
        )

    return pts


def _score_occlusion(
    cam_pos: Vec3,
    sample_pts: list[Vec3],
    target_obj_ids: set[int],
    bpy_scene: bpy.types.Scene,
) -> float:
    """Raycast-based visibility score: fraction of unoccluded rays."""
    if not sample_pts:
        return 1.0
    depsgraph = bpy_scene.view_layers[0].depsgraph
    origin = mathutils.Vector((cam_pos.x, cam_pos.y, cam_pos.z))
    clear = 0
    for pt in sample_pts:
        direction = mathutils.Vector((
            pt.x - cam_pos.x,
            pt.y - cam_pos.y,
            pt.z - cam_pos.z,
        ))
        dist = direction.length
        if dist < 1e-6:
            clear += 1
            continue
        direction.normalize()
        hit, _loc, _norm, _idx, hit_obj, _mat = bpy_scene.ray_cast(
            depsgraph,
            origin,
            direction,
            distance=dist * 1.01,
        )
        if not hit or hit_obj is None or id(hit_obj) in target_obj_ids:
            clear += 1
    return clear / len(sample_pts)


def _score_fill(cam_pos: Vec3, center: Vec3, diag: float) -> float:
    """Frame-fill score: Gaussian peaked at ~40% fill ratio."""
    dx = cam_pos.x - center.x
    dy = cam_pos.y - center.y
    dz = cam_pos.z - center.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    if dist < 1e-6:
        return 0.0
    # Angular diameter as fraction of a ~50° FOV
    angular_size = 2.0 * math.atan2(diag * 0.5, dist)
    fov = math.radians(50.0)
    fill_ratio = angular_size / fov
    # Gaussian centered at 0.4 with σ=0.25
    return math.exp(-(((fill_ratio - 0.4) / 0.25) ** 2))


def _score_above(cam_pos: Vec3, center: Vec3) -> float:
    """Elevation preference: 1.0 for 20°-55°, smooth falloff outside."""
    dx = cam_pos.x - center.x
    dy = cam_pos.y - center.y
    dz = cam_pos.z - center.z
    horiz = math.sqrt(dx * dx + dy * dy)
    elev_deg = math.degrees(math.atan2(dz, horiz)) if horiz > 1e-6 else 90.0
    if 20.0 <= elev_deg <= 55.0:
        return 1.0
    if elev_deg < 20.0:
        # Smooth falloff below 20°
        return max(0.0, math.exp(-(((20.0 - elev_deg) / 15.0) ** 2)))
    # Above 55°
    return max(0.0, math.exp(-(((elev_deg - 55.0) / 20.0) ** 2)))


def _best_viewpoint(
    center: Vec3,
    diag: float,
    radius: float,
    elev_deg: float,
    n_candidates: int,
    occlusion_samples: int,
    target_obj_ids: set[int],
    bpy_scene: bpy.types.Scene,
    bbox: tuple[Vec3, Vec3],
) -> _Viewpoint:
    """Evaluate candidates and return the best-scoring viewpoint."""
    candidates = _generate_candidates(center, radius, elev_deg, n_candidates)
    sample_pts = _bbox_sample_points(bbox[0], bbox[1], occlusion_samples)

    best = _Viewpoint(pos=candidates[0] if candidates else center)
    for pos in candidates:
        occ = _score_occlusion(pos, sample_pts, target_obj_ids, bpy_scene)
        fill = _score_fill(pos, center, diag)
        above = _score_above(pos, center)
        score = 0.5 * occ + 0.3 * fill + 0.2 * above
        if score > best.score:
            best = _Viewpoint(pos=pos, score=score)
    return best


def _make_overview_keyframe(
    scene_bbox: tuple[Vec3, Vec3],
    hold_frames: int,
    elev_deg: float,
    radius_factor: float,
) -> _Keyframe:
    """Create an establishing overview shot keyframe."""
    lo, hi = scene_bbox
    center = Vec3(
        (lo.x + hi.x) * 0.5,
        (lo.y + hi.y) * 0.5,
        (lo.z + hi.z) * 0.5,
    )
    size = Vec3(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z)
    diag = max(1.0, math.sqrt(size.x**2 + size.y**2 + size.z**2))
    radius = diag * radius_factor
    elev = math.radians(elev_deg)
    azim = math.radians(45.0)  # classic 3/4 view
    pos = Vec3(
        center.x + radius * math.cos(elev) * math.cos(azim),
        center.y + radius * math.cos(elev) * math.sin(azim),
        center.z + radius * math.sin(elev),
    )
    return _Keyframe(pos=pos, look_at=center, hold_frames=hold_frames, label="overview")


def _order_keyframes(
    overview: _Keyframe | None,
    target_kfs: list[_Keyframe],
) -> list[_Keyframe]:
    """Order keyframes via greedy nearest-neighbor from *overview*."""
    if not target_kfs:
        return [overview] if overview else []

    def _dist(a: Vec3, b: Vec3) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    ordered: list[_Keyframe] = []
    if overview is not None:
        ordered.append(overview)
        current_pos = overview.pos
    else:
        # Start from the first target, then greedily visit the rest
        ordered.append(target_kfs[0])
        current_pos = target_kfs[0].pos
        target_kfs = target_kfs[1:]

    remaining = list(target_kfs)
    while remaining:
        nearest_idx = 0
        nearest_dist = _dist(current_pos, remaining[0].pos)
        for i in range(1, len(remaining)):
            d = _dist(current_pos, remaining[i].pos)
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = i
        best = remaining.pop(nearest_idx)
        ordered.append(best)
        current_pos = best.pos

    return ordered


def _interpolate_showcase_path(
    keyframes: list[_Keyframe],
    transition_frames: int,
) -> list[tuple[Vec3, Vec3]]:
    """
    Interpolate full frame-by-frame path: list of (cam_pos, look_at).

    Each keyframe gets its hold phase, then a Catmull-Rom + smoothstep
    transition to the next keyframe.
    """
    if not keyframes:
        return []
    if len(keyframes) == 1:
        return [(keyframes[0].pos, keyframes[0].look_at)] * keyframes[0].hold_frames

    path: list[tuple[Vec3, Vec3]] = []
    n = len(keyframes)

    for ki in range(n):
        kf = keyframes[ki]
        # Hold phase
        path.extend((kf.pos, kf.look_at) for _ in range(kf.hold_frames))

        # Transition to next keyframe (skip after last)
        if ki < n - 1:
            kf_next = keyframes[ki + 1]
            # Catmull-Rom needs 4 control points; mirror at endpoints
            if ki == 0:
                p0_pos = Vec3(
                    2.0 * kf.pos.x - kf_next.pos.x,
                    2.0 * kf.pos.y - kf_next.pos.y,
                    2.0 * kf.pos.z - kf_next.pos.z,
                )
                p0_look = Vec3(
                    2.0 * kf.look_at.x - kf_next.look_at.x,
                    2.0 * kf.look_at.y - kf_next.look_at.y,
                    2.0 * kf.look_at.z - kf_next.look_at.z,
                )
            else:
                p0_pos = keyframes[ki - 1].pos
                p0_look = keyframes[ki - 1].look_at

            if ki + 2 < n:
                p3_pos = keyframes[ki + 2].pos
                p3_look = keyframes[ki + 2].look_at
            else:
                p3_pos = Vec3(
                    2.0 * kf_next.pos.x - kf.pos.x,
                    2.0 * kf_next.pos.y - kf.pos.y,
                    2.0 * kf_next.pos.z - kf.pos.z,
                )
                p3_look = Vec3(
                    2.0 * kf_next.look_at.x - kf.look_at.x,
                    2.0 * kf_next.look_at.y - kf.look_at.y,
                    2.0 * kf_next.look_at.z - kf.look_at.z,
                )

            for fi in range(transition_frames):
                raw_t = (fi + 1) / (transition_frames + 1)
                t = _smoothstep(raw_t)
                pos = _cr_eval_vec3(p0_pos, kf.pos, kf_next.pos, p3_pos, t)
                look = _cr_eval_vec3(p0_look, kf.look_at, kf_next.look_at, p3_look, t)
                path.append((pos, look))

    return path


# ---------------------------------------------------------------------------
# Batch-spec constructors
# ---------------------------------------------------------------------------


def still(out: str, preset: str = "top", **opts: Any) -> RenderSpec:
    """Small constructor for ``RenderContext.batch`` still specs."""
    spec = RenderSpec(type="still", out=out, preset=preset)
    spec.update(cast("Any", opts))
    return spec


def focus(
    out: str, preset: str = "iso_close", *, where: str, **opts: Any
) -> RenderSpec:
    """Small constructor for ``RenderContext.batch`` focus specs."""
    spec = RenderSpec(type="focus", out=out, preset=preset, where=where)
    spec.update(cast("Any", opts))
    return spec
