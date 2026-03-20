"""Blender Python (bpy) code generation for keyframe animations."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Property -> bpy data_path mapping
# ---------------------------------------------------------------------------

_PROPERTY_MAP: dict[str, dict[str, Any]] = {
    "location": {
        "data_path": "location",
        "is_array": True,
        "components": 3,
    },
    "rotation": {
        "data_path": "rotation_euler",
        "is_array": True,
        "components": 3,
    },
    "scale": {
        "data_path": "scale",
        "is_array": True,
        "components": 3,
    },
    "visible": {
        "data_path": "hide_render",
        "is_array": False,
        "invert": True,  # visible=True -> hide_render=False
    },
    "material.color": {
        "data_path": "active_material.diffuse_color",
        "is_array": True,
        "components": 4,
        "material": True,
    },
    "material.metallic": {
        "data_path": "active_material.metallic",
        "is_array": False,
        "material": True,
    },
    "material.roughness": {
        "data_path": "active_material.roughness",
        "is_array": False,
        "material": True,
    },
    "material.alpha": {
        "data_path": "active_material.alpha",
        "is_array": False,
        "material": True,
    },
    "material.emission_strength": {
        "data_path": "active_material.emission_strength",
        "is_array": False,
        "material": True,
    },
}

_VALID_INTERP = {"CONSTANT", "LINEAR", "BEZIER"}


def _repr_value(value: Any) -> str:
    """Format a value for embedding in generated Python code."""
    if isinstance(value, (list, tuple)):
        return repr(list(value))
    return repr(value)


def keyframe_codegen(
    obj_var: str,
    keyframes: list[dict[str, Any]],
) -> str:
    """
    Generate bpy code that sets keyframes on an object.

    Args:
        obj_var: Python variable name referencing the bpy object (e.g. "obj").
        keyframes: List of keyframe dicts from project JSON.

    Returns:
        Multi-line Python code string.

    """
    if not keyframes:
        return ""

    lines: list[str] = []

    for kf in keyframes:
        frame = kf["frame"]
        prop = kf["property"]
        value = kf["value"]
        interp = kf.get("interpolation", "BEZIER")

        mapping = _PROPERTY_MAP.get(prop)
        if mapping is None:
            lines.append(f"# Unknown property: {prop}")
            continue

        data_path = mapping["data_path"]
        is_array = mapping.get("is_array", False)

        # Set the value
        if mapping.get("invert"):
            lines.extend((
                f"{obj_var}.hide_render = {_repr_value(not value)}",
                f"{obj_var}.hide_viewport = {_repr_value(not value)}",
            ))
        else:
            lines.append(f"{obj_var}.{data_path} = {_repr_value(value)}")

        # Insert keyframe
        if is_array:
            components = mapping.get("components", 3)
            lines.extend(
                f"{obj_var}.keyframe_insert(data_path={data_path!r}, index={idx}, frame={frame})"
                for idx in range(components)
            )
        elif mapping.get("invert"):
            lines.extend((
                f"{obj_var}.keyframe_insert(data_path='hide_render', frame={frame})",
                f"{obj_var}.keyframe_insert(data_path='hide_viewport', frame={frame})",
            ))
        else:
            lines.append(
                f"{obj_var}.keyframe_insert(data_path={data_path!r}, frame={frame})"
            )

        # Set interpolation
        bpy_interp = interp if interp in _VALID_INTERP else "BEZIER"
        lines.extend((f"# interpolation: {bpy_interp}", ""))

    return "\n".join(lines).rstrip()
