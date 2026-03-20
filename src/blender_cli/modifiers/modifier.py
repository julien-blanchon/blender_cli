"""Modifier operations — add, remove, set, list modifiers on project objects."""

from __future__ import annotations

from typing import Any

from blender_cli.modifiers.registry import MODIFIER_REGISTRY
from blender_cli.project.project_file import resolve_object

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_TYPE_CHECKERS: dict[str, type | tuple[type, ...]] = {
    "float": (int, float),
    "int": (int,),
    "bool": (bool,),
    "str": (str,),
}


def validate_param(spec: dict[str, Any], value: Any) -> str | None:
    """
    Validate a single parameter value against its spec.

    Returns an error message string if invalid, None if valid.
    """
    pname = spec["name"]
    ptype = spec["type"]

    # Enum check first (before type, since enums are strings)
    if "enum" in spec and value not in spec["enum"]:
        return f"Parameter {pname!r}: {value!r} not in {spec['enum']}"

    # bool is subclass of int — reject bools for int/float params before type check
    if ptype in {"int", "float"} and isinstance(value, bool):
        return f"Parameter {pname!r}: expected {ptype}, got bool"

    # Type check
    expected = _TYPE_CHECKERS.get(ptype)
    if expected and not isinstance(value, expected):
        return f"Parameter {pname!r}: expected {ptype}, got {type(value).__name__}"

    # Range check
    if "min" in spec and value < spec["min"]:
        return f"Parameter {pname!r}: {value} < min {spec['min']}"
    if "max" in spec and value > spec["max"]:
        return f"Parameter {pname!r}: {value} > max {spec['max']}"

    return None


def validate_modifier_params(mod_type: str, params: dict[str, Any]) -> list[str]:
    """Validate all params for a modifier type. Returns list of error messages."""
    if mod_type not in MODIFIER_REGISTRY:
        return [
            f"Unknown modifier type {mod_type!r}. Available: {sorted(MODIFIER_REGISTRY)}"
        ]

    spec_list = MODIFIER_REGISTRY[mod_type]["params"]
    spec_by_name = {s["name"]: s for s in spec_list}

    # Check for unknown params
    errors: list[str] = [
        f"Unknown parameter {k!r} for {mod_type}. Valid: {sorted(spec_by_name)}"
        for k in params
        if k not in spec_by_name
    ]

    # Validate provided params
    for k, v in params.items():
        spec = spec_by_name.get(k)
        if spec:
            err = validate_param(spec, v)
            if err:
                errors.append(err)

    return errors




# ---------------------------------------------------------------------------
# Modifier class — operations on project objects
# ---------------------------------------------------------------------------


class Modifier:
    """Add, remove, list, and update modifiers on project objects."""

    @staticmethod
    def add(
        project_data: dict[str, Any],
        object_ref: str | int,
        mod_type: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a modifier to an object. Returns the created modifier dict."""
        if mod_type not in MODIFIER_REGISTRY:
            msg = f"Unknown modifier type {mod_type!r}. Available: {sorted(MODIFIER_REGISTRY)}"
            raise ValueError(msg)

        spec = MODIFIER_REGISTRY[mod_type]
        params = params or {}

        # Fill defaults for unspecified params
        full_params: dict[str, Any] = {}
        for p in spec["params"]:
            full_params[p["name"]] = params.get(p["name"], p["default"])

        # Validate
        errors = validate_modifier_params(mod_type, full_params)
        if errors:
            msg = f"Invalid modifier params: {'; '.join(errors)}"
            raise ValueError(msg)

        _, obj = resolve_object(project_data, object_ref)

        if "modifiers" not in obj:
            obj["modifiers"] = []

        modifier = {
            "type": mod_type,
            "name": spec["name"],
            "bpy_type": spec["bpy_type"],
            "params": full_params,
        }
        obj["modifiers"].append(modifier)
        return modifier

    @staticmethod
    def remove(
        project_data: dict[str, Any],
        object_ref: str | int,
        index: int,
    ) -> dict[str, Any]:
        """Remove a modifier by index. Returns the removed modifier dict."""
        _, obj = resolve_object(project_data, object_ref)
        mods = obj.get("modifiers", [])
        if index < 0 or index >= len(mods):
            raise IndexError(
                f"Modifier index {index} out of range [0, {len(mods) - 1}]"
                if mods
                else "Object has no modifiers"
            )
        return mods.pop(index)

    @staticmethod
    def set(
        project_data: dict[str, Any],
        object_ref: str | int,
        index: int,
        param: str,
        value: Any,
    ) -> dict[str, Any]:
        """Update a modifier parameter. Returns the updated modifier dict."""
        _, obj = resolve_object(project_data, object_ref)
        mods = obj.get("modifiers", [])
        if index < 0 or index >= len(mods):
            msg = f"Modifier index {index} out of range"
            raise IndexError(msg)
        mod = mods[index]

        # Validate the param against the registry spec
        spec_list = MODIFIER_REGISTRY[mod["type"]]["params"]
        spec_by_name = {s["name"]: s for s in spec_list}
        if param not in spec_by_name:
            msg = f"Unknown parameter {param!r} for {mod['type']}. Valid: {sorted(spec_by_name)}"
            raise ValueError(msg)

        err = validate_param(spec_by_name[param], value)
        if err:
            raise ValueError(err)

        mod["params"][param] = value
        return mod

    @staticmethod
    def list(
        project_data: dict[str, Any],
        object_ref: str | int,
    ) -> list[dict[str, Any]]:
        """List modifiers on an object."""
        _, obj = resolve_object(project_data, object_ref)
        return obj.get("modifiers", [])
