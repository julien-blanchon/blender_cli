"""Blender Python (bpy) code generation for modifiers."""

from __future__ import annotations

from typing import Any


def _repr_value(value: Any) -> str:
    """Format a value for embedding in generated Python code."""
    return repr(value)


# ---------------------------------------------------------------------------
# Per-modifier-type property mapping
# ---------------------------------------------------------------------------
# Maps modifier param names to bpy modifier attribute names.
# Most are 1:1, but a few need special handling (e.g. array offsets).

_ARRAY_OFFSET_AXES = {
    "relative_offset_displace_x": ("relative_offset_displace", 0),
    "relative_offset_displace_y": ("relative_offset_displace", 1),
    "relative_offset_displace_z": ("relative_offset_displace", 2),
    "constant_offset_displace_x": ("constant_offset_displace", 0),
    "constant_offset_displace_y": ("constant_offset_displace", 1),
    "constant_offset_displace_z": ("constant_offset_displace", 2),
}


def _codegen_single_modifier(
    obj_var: str,
    mod: dict[str, Any],
    mod_var: str = "mod",
) -> list[str]:
    """
    Generate bpy lines for a single modifier.

    Returns a list of Python source lines (no trailing newlines).
    """
    mod_type = mod["type"]
    bpy_type = mod["bpy_type"]
    params = mod.get("params", {})
    lines: list[str] = []

    lines.append(
        f"{mod_var} = {obj_var}.modifiers.new(name={mod['name']!r}, type={bpy_type!r})"
    )

    for pname, pval in params.items():
        # Special handling: boolean operand_object
        if mod_type == "boolean" and pname == "operand_object":
            if pval:
                lines.append(
                    f"{mod_var}.object = bpy.data.objects.get({_repr_value(pval)})"
                )
            continue

        # Special handling: array vector offsets
        if mod_type == "array" and pname in _ARRAY_OFFSET_AXES:
            attr, idx = _ARRAY_OFFSET_AXES[pname]
            lines.append(f"{mod_var}.{attr}[{idx}] = {_repr_value(pval)}")
            continue

        # Default: direct attribute assignment
        lines.append(f"{mod_var}.{pname} = {_repr_value(pval)}")

    return lines


def modifier_codegen(
    obj_var: str,
    modifiers: list[dict[str, Any]],
) -> str:
    """
    Generate bpy code string that applies all modifiers to an object.

    Args:
        obj_var: Python variable name referencing the bpy object (e.g. "obj").
        modifiers: List of modifier dicts from project JSON.

    Returns:
        Multi-line Python code string.

    """
    if not modifiers:
        return ""

    all_lines: list[str] = []
    for i, mod in enumerate(modifiers):
        mod_var = f"mod_{i}"
        lines = _codegen_single_modifier(obj_var, mod, mod_var)
        all_lines.extend(lines)
        all_lines.append("")  # blank separator

    return "\n".join(all_lines).rstrip()
