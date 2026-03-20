"""Modifier registry — defines all supported Blender modifiers with parameter specs."""

from __future__ import annotations

import math
from typing import Any

# ---------------------------------------------------------------------------
# Parameter spec: each param is a dict with name, type, default, and optional
# min/max/enum constraints.
# ---------------------------------------------------------------------------


_P = dict[str, Any]  # shorthand


def _param(
    name: str,
    type: str,  # noqa: A002 — shadows builtin intentionally for readability
    default: Any,
    *,
    min: Any = None,  # noqa: A002
    max: Any = None,  # noqa: A002
    enum: list[str] | None = None,
    description: str = "",
) -> _P:
    p: _P = {"name": name, "type": type, "default": default, "description": description}
    if min is not None:
        p["min"] = min
    if max is not None:
        p["max"] = max
    if enum is not None:
        p["enum"] = enum
    return p


# ---------------------------------------------------------------------------
# Registry: 8 modifier types
# ---------------------------------------------------------------------------

MODIFIER_REGISTRY: dict[str, dict[str, Any]] = {
    "subdivision_surface": {
        "name": "Subdivision Surface",
        "category": "generate",
        "bpy_type": "SUBSURF",
        "description": "Subdivide mesh faces for smoother geometry.",
        "params": [
            _param(
                "levels",
                "int",
                1,
                min=0,
                max=6,
                description="Viewport subdivision levels",
            ),
            _param(
                "render_levels",
                "int",
                2,
                min=0,
                max=6,
                description="Render subdivision levels",
            ),
            _param("use_creases", "bool", True, description="Use edge creases"),
            _param(
                "uv_smooth",
                "str",
                "PRESERVE_BOUNDARIES",
                enum=["NONE", "PRESERVE_CORNERS", "PRESERVE_BOUNDARIES", "SMOOTH_ALL"],
                description="UV smoothing mode",
            ),
        ],
    },
    "mirror": {
        "name": "Mirror",
        "category": "generate",
        "bpy_type": "MIRROR",
        "description": "Mirror mesh across one or more axes.",
        "params": [
            _param("use_axis_x", "bool", True, description="Mirror on X axis"),
            _param("use_axis_y", "bool", False, description="Mirror on Y axis"),
            _param("use_axis_z", "bool", False, description="Mirror on Z axis"),
            _param(
                "use_clip",
                "bool",
                True,
                description="Prevent vertices from crossing mirror plane",
            ),
            _param(
                "merge_threshold",
                "float",
                0.001,
                min=0.0,
                max=1.0,
                description="Merge distance for mirrored vertices",
            ),
        ],
    },
    "array": {
        "name": "Array",
        "category": "generate",
        "bpy_type": "ARRAY",
        "description": "Create copies arranged in a line or pattern.",
        "params": [
            _param("count", "int", 2, min=1, max=1000, description="Number of copies"),
            _param(
                "use_relative_offset", "bool", True, description="Use relative offset"
            ),
            _param(
                "relative_offset_displace_x",
                "float",
                1.0,
                min=-100.0,
                max=100.0,
                description="Relative offset X",
            ),
            _param(
                "relative_offset_displace_y",
                "float",
                0.0,
                min=-100.0,
                max=100.0,
                description="Relative offset Y",
            ),
            _param(
                "relative_offset_displace_z",
                "float",
                0.0,
                min=-100.0,
                max=100.0,
                description="Relative offset Z",
            ),
            _param(
                "use_constant_offset", "bool", False, description="Use constant offset"
            ),
            _param(
                "constant_offset_displace_x",
                "float",
                0.0,
                description="Constant offset X (metres)",
            ),
            _param(
                "constant_offset_displace_y",
                "float",
                0.0,
                description="Constant offset Y (metres)",
            ),
            _param(
                "constant_offset_displace_z",
                "float",
                0.0,
                description="Constant offset Z (metres)",
            ),
        ],
    },
    "bevel": {
        "name": "Bevel",
        "category": "generate",
        "bpy_type": "BEVEL",
        "description": "Bevel edges for smoother hard-surface transitions.",
        "params": [
            _param(
                "width", "float", 0.1, min=0.0, max=100.0, description="Bevel width"
            ),
            _param(
                "segments",
                "int",
                1,
                min=1,
                max=100,
                description="Number of bevel segments",
            ),
            _param(
                "affect",
                "str",
                "EDGES",
                enum=["VERTICES", "EDGES"],
                description="What to bevel",
            ),
            _param(
                "limit_method",
                "str",
                "NONE",
                enum=["NONE", "ANGLE", "WEIGHT", "VGROUP"],
                description="Limit bevel by method",
            ),
            _param(
                "angle_limit",
                "float",
                0.523599,
                min=0.0,
                max=math.pi,
                description="Angle limit in radians (for ANGLE method)",
            ),
            _param(
                "clamp_overlap",
                "bool",
                True,
                description="Clamp bevel to prevent overlap",
            ),
        ],
    },
    "solidify": {
        "name": "Solidify",
        "category": "generate",
        "bpy_type": "SOLIDIFY",
        "description": "Add thickness to surface meshes.",
        "params": [
            _param(
                "thickness",
                "float",
                0.01,
                min=-10.0,
                max=10.0,
                description="Thickness of solidified shell",
            ),
            _param(
                "offset",
                "float",
                -1.0,
                min=-1.0,
                max=1.0,
                description="Offset direction (-1=outward, 1=inward)",
            ),
            _param(
                "use_even_offset", "bool", True, description="Maintain even thickness"
            ),
            _param("use_rim", "bool", True, description="Create rim faces"),
        ],
    },
    "decimate": {
        "name": "Decimate",
        "category": "generate",
        "bpy_type": "DECIMATE",
        "description": "Reduce polygon count while preserving shape.",
        "params": [
            _param(
                "decimate_type",
                "str",
                "COLLAPSE",
                enum=["COLLAPSE", "UNSUBDIV", "DISSOLVE"],
                description="Decimation algorithm",
            ),
            _param(
                "ratio",
                "float",
                0.5,
                min=0.0,
                max=1.0,
                description="Ratio of faces to keep (COLLAPSE mode)",
            ),
            _param(
                "iterations",
                "int",
                0,
                min=0,
                max=10,
                description="Un-subdivision iterations (UNSUBDIV mode)",
            ),
            _param(
                "angle_limit",
                "float",
                0.087266,
                min=0.0,
                max=math.pi,
                description="Dissolve angle limit in radians (DISSOLVE mode)",
            ),
            _param("use_symmetry", "bool", False, description="Maintain symmetry"),
        ],
    },
    "boolean": {
        "name": "Boolean",
        "category": "generate",
        "bpy_type": "BOOLEAN",
        "description": "Combine meshes with boolean operations.",
        "params": [
            _param(
                "operation",
                "str",
                "DIFFERENCE",
                enum=["INTERSECT", "UNION", "DIFFERENCE"],
                description="Boolean operation type",
            ),
            _param(
                "operand_object", "str", "", description="Name of the operand object"
            ),
            _param(
                "solver",
                "str",
                "EXACT",
                enum=["FAST", "EXACT"],
                description="Boolean solver algorithm",
            ),
            _param("use_self", "bool", False, description="Allow self-intersection"),
        ],
    },
    "smooth": {
        "name": "Smooth",
        "category": "deform",
        "bpy_type": "SMOOTH",
        "description": "Smooth mesh geometry by averaging vertex positions.",
        "params": [
            _param(
                "factor", "float", 0.5, min=0.0, max=1.0, description="Smoothing factor"
            ),
            _param(
                "iterations",
                "int",
                1,
                min=0,
                max=100,
                description="Number of smoothing passes",
            ),
            _param("use_x", "bool", True, description="Smooth along X axis"),
            _param("use_y", "bool", True, description="Smooth along Y axis"),
            _param("use_z", "bool", True, description="Smooth along Z axis"),
        ],
    },
}


# ---------------------------------------------------------------------------
# ModifierRegistry — lookup helpers
# ---------------------------------------------------------------------------


class ModifierRegistry:
    """Read-only access to the modifier registry."""

    @staticmethod
    def available(category: str | None = None) -> list[dict[str, Any]]:
        """List available modifiers, optionally filtered by category."""
        results = []
        for key, spec in sorted(MODIFIER_REGISTRY.items()):
            if category and spec["category"] != category:
                continue
            results.append({"type": key, **spec})
        return results

    @staticmethod
    def info(name: str) -> dict[str, Any]:
        """Get full info for a modifier type. Raises KeyError if not found."""
        if name not in MODIFIER_REGISTRY:
            msg = f"Unknown modifier type {name!r}. Available: {sorted(MODIFIER_REGISTRY)}"
            raise KeyError(msg)
        return {"type": name, **MODIFIER_REGISTRY[name]}
