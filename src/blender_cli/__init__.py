"""
maps-creation SDK — agent-friendly procedural map generation.

Core types and primitives are re-exported here for convenience::

    from blender_cli import Scene, plane, box

For less common utilities, import from submodules directly::

    from blender_cli.utils.placement import perimeter_points
    from blender_cli.types import AddResult
"""

# Scene & entity primitives — the most-used API surface.
# Alignment
from blender_cli.alignment import (
    AlignmentPipelineResult,
    ComposeOptions,
    ComposeResult,
    GenerationOptions,
    GenerationResult,
    PoseEstimationOptions,
    PoseEstimationResult,
    compose_from_directory,
    estimate_pose_from_directory,
    generate_alignment_assets,
    integrate_pose_result,
    load_pose_result,
    run_alignment_pipeline,
)

# Assets
from blender_cli.assets import Material

# Build
from blender_cli.build import BuildContext

# Geometry
from blender_cli.geometry import (
    WILDCARD,
    Field2D,
    Heightfield,
    Mask,
    PointSet,
    Spline,
    SplineOp,
)

# Project
from blender_cli.project import ProjectFile, Session

# Render
from blender_cli.render import (
    Camera,
    CameraKeyframe,
    CameraPath,
    RenderContext,
    focus,
    still,
)
from blender_cli.scene import (
    Anchor,
    Entity,
    Instances,
    Scene,
    Selection,
    SnapSpec,
    Transform,
    as_entity,
    box,
    cone,
    cylinder,
    plane,
    sphere,
    torus,
)

# Blenvy
from blender_cli.blenvy import apply_bevy_components, to_ron
from blender_cli.blenvy_registry import BevyRegistry, ComponentInfo

# Snap
from blender_cli.snap import SnapPolicy
from blender_cli.snap import snap as snap_points

# Types
from blender_cli.types import Vec3

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
