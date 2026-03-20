"""Animation system — keyframes, validation, and project integration."""

from blender_cli.animation.codegen import keyframe_codegen
from blender_cli.animation.keyframes import (
    ANIMATABLE_PROPERTIES,
    INTERPOLATION_MODES,
    Animation,
)

__all__ = [
    "ANIMATABLE_PROPERTIES",
    "INTERPOLATION_MODES",
    "Animation",
    "keyframe_codegen",
]
