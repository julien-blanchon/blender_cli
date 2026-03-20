"""Render subpackage — camera presets, multi-pass still & animated rendering."""

from blender_cli.render.camera import Camera
from blender_cli.render.camera_path import CameraKeyframe, CameraPath
from blender_cli.render.context import (
    RenderContext,
    setup_hdri_world,
    bundled_hdri,
    focus,
    still,
)
from blender_cli.render.settings import (
    QUALITY_PRESET_MAP,
    RENDER_PRESETS,
    RenderSettings,
    camera_dof_codegen,
    default_dof,
    render_settings_codegen,
    validate_dof,
)
from blender_cli.render.world import (
    WorldSettings,
    validate_world,
    world_codegen,
)

__all__ = [
    "QUALITY_PRESET_MAP",
    "RENDER_PRESETS",
    "Camera",
    "CameraKeyframe",
    "CameraPath",
    "RenderContext",
    "RenderSettings",
    "WorldSettings",
    "bundled_hdri",
    "camera_dof_codegen",
    "default_dof",
    "focus",
    "render_settings_codegen",
    "still",
    "setup_hdri_world",
    "validate_dof",
    "validate_world",
    "world_codegen",
]
