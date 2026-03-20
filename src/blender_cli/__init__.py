"""
maps-creation SDK — agent-friendly procedural map generation.

Core types and primitives are re-exported here for convenience::

    from blender_cli import Scene, plane, box

For less common utilities, import from submodules directly::

    from blender_cli.utils.placement import perimeter_points
    from blender_cli.types import AddResult
"""

from __future__ import annotations

# Lazy imports — heavy dependencies (scipy, cv2, bpy, cma, numpy, etc.)
# are only loaded when the corresponding name is first accessed.
# This keeps `blender_cli.cli:main` startup fast for uvx.

__all__ = [
    "WILDCARD",
    # alignment
    "AlignmentPipelineResult",
    # scene
    "Anchor",
    # blenvy registry
    "BevyRegistry",
    # build
    "BuildContext",
    # render
    "Camera",
    "CameraKeyframe",
    "CameraPath",
    "ComponentInfo",
    "ComposeOptions",
    "ComposeResult",
    "Entity",
    # geometry
    "Field2D",
    "GenerationOptions",
    "GenerationResult",
    "Heightfield",
    "Instances",
    "Mask",
    # assets
    "Material",
    "PointSet",
    "PoseEstimationOptions",
    "PoseEstimationResult",
    # project
    "ProjectFile",
    "Session",
    "RenderContext",
    "Scene",
    "Selection",
    # snap
    "SnapPolicy",
    "SnapSpec",
    "Spline",
    "SplineOp",
    "Transform",
    # types
    "Vec3",
    "apply_bevy_components",
    "as_entity",
    "box",
    "compose_from_directory",
    "cone",
    "cylinder",
    "estimate_pose_from_directory",
    "focus",
    "generate_alignment_assets",
    "integrate_pose_result",
    "load_pose_result",
    "plane",
    "run_alignment_pipeline",
    "snap_points",
    "sphere",
    "still",
    "to_ron",
    "torus",
]

# Map of public name → (module, attribute)
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # alignment
    "AlignmentPipelineResult": ("blender_cli.alignment", "AlignmentPipelineResult"),
    "ComposeOptions": ("blender_cli.alignment", "ComposeOptions"),
    "ComposeResult": ("blender_cli.alignment", "ComposeResult"),
    "GenerationOptions": ("blender_cli.alignment", "GenerationOptions"),
    "GenerationResult": ("blender_cli.alignment", "GenerationResult"),
    "PoseEstimationOptions": ("blender_cli.alignment", "PoseEstimationOptions"),
    "PoseEstimationResult": ("blender_cli.alignment", "PoseEstimationResult"),
    "compose_from_directory": ("blender_cli.alignment", "compose_from_directory"),
    "estimate_pose_from_directory": ("blender_cli.alignment", "estimate_pose_from_directory"),
    "generate_alignment_assets": ("blender_cli.alignment", "generate_alignment_assets"),
    "integrate_pose_result": ("blender_cli.alignment", "integrate_pose_result"),
    "load_pose_result": ("blender_cli.alignment", "load_pose_result"),
    "run_alignment_pipeline": ("blender_cli.alignment", "run_alignment_pipeline"),
    # assets
    "Material": ("blender_cli.assets", "Material"),
    # build
    "BuildContext": ("blender_cli.build", "BuildContext"),
    # geometry
    "WILDCARD": ("blender_cli.geometry", "WILDCARD"),
    "Field2D": ("blender_cli.geometry", "Field2D"),
    "Heightfield": ("blender_cli.geometry", "Heightfield"),
    "Mask": ("blender_cli.geometry", "Mask"),
    "PointSet": ("blender_cli.geometry", "PointSet"),
    "Spline": ("blender_cli.geometry", "Spline"),
    "SplineOp": ("blender_cli.geometry", "SplineOp"),
    # project
    "ProjectFile": ("blender_cli.project", "ProjectFile"),
    "Session": ("blender_cli.project", "Session"),
    # render
    "Camera": ("blender_cli.render", "Camera"),
    "CameraKeyframe": ("blender_cli.render", "CameraKeyframe"),
    "CameraPath": ("blender_cli.render", "CameraPath"),
    "RenderContext": ("blender_cli.render", "RenderContext"),
    "focus": ("blender_cli.render", "focus"),
    "still": ("blender_cli.render", "still"),
    # scene
    "Anchor": ("blender_cli.scene", "Anchor"),
    "Entity": ("blender_cli.scene", "Entity"),
    "Instances": ("blender_cli.scene", "Instances"),
    "Scene": ("blender_cli.scene", "Scene"),
    "Selection": ("blender_cli.scene", "Selection"),
    "SnapSpec": ("blender_cli.scene", "SnapSpec"),
    "Transform": ("blender_cli.scene", "Transform"),
    "as_entity": ("blender_cli.scene", "as_entity"),
    "box": ("blender_cli.scene", "box"),
    "cone": ("blender_cli.scene", "cone"),
    "cylinder": ("blender_cli.scene", "cylinder"),
    "plane": ("blender_cli.scene", "plane"),
    "sphere": ("blender_cli.scene", "sphere"),
    "torus": ("blender_cli.scene", "torus"),
    # blenvy
    "apply_bevy_components": ("blender_cli.blenvy", "apply_bevy_components"),
    "to_ron": ("blender_cli.blenvy", "to_ron"),
    "BevyRegistry": ("blender_cli.blenvy_registry", "BevyRegistry"),
    "ComponentInfo": ("blender_cli.blenvy_registry", "ComponentInfo"),
    # snap
    "SnapPolicy": ("blender_cli.snap", "SnapPolicy"),
    "snap_points": ("blender_cli.snap", "snap"),
    # types
    "Vec3": ("blender_cli.types", "Vec3"),
}


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path)
        value = getattr(module, attr)
        # Cache on the module so __getattr__ isn't called again
        globals()[name] = value
        return value
    raise AttributeError(f"module 'blender_cli' has no attribute {name!r}")
