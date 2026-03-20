"""Modifier system — registry, validation, and project integration."""

from blender_cli.modifiers.codegen import modifier_codegen
from blender_cli.modifiers.modifier import Modifier
from blender_cli.modifiers.registry import (
    MODIFIER_REGISTRY,
    ModifierRegistry,
)

__all__ = [
    "MODIFIER_REGISTRY",
    "Modifier",
    "ModifierRegistry",
    "modifier_codegen",
]
