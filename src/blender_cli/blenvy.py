"""Blenvy component serialization for GLTF extras.

Converts Python values to RON (Rusty Object Notation) strings that Blenvy's
``ronstring_to_reflect_component`` can deserialize into Bevy components.

Each Bevy component is stored as a GLTF node extra with:
  - key   = component type name (e.g. ``"RigidBody"``, ``"BlueprintInfo"``)
  - value = RON-formatted string (e.g. ``"Dynamic"``, ``"(name: \\"Wall\\")"``

Usage from the SDK::

    entity = Entity(mesh).component("RigidBody", "Dynamic")
    entity.component("Health", {"max": 100, "current": 100})
    entity.component("SpawnBlueprint")  # unit component

The raw RON string is written as a Blender custom property on the object.
Blender's glTF exporter (with ``export_extras=True``) serialises it into the
GLTF node extras, which Blenvy picks up on the Bevy side.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from blender_cli.blenvy_registry import BevyRegistry

# Module-level active registry (optional — set via set_registry()).
_active_registry: BevyRegistry | None = None


def set_registry(registry: BevyRegistry | None) -> None:
    """Set the module-level Bevy component registry for validation."""
    global _active_registry  # noqa: PLW0603
    _active_registry = registry


def get_registry() -> BevyRegistry | None:
    """Return the current module-level registry, or ``None``."""
    return _active_registry

# Type alias for a single Bevy component value.
# Can be: a raw RON string, a dict (struct), a list, a scalar, or None (unit).
type BevyComponentValue = str | dict[str, Any] | list[Any] | int | float | bool | None


def to_ron(value: BevyComponentValue) -> str:
    """Convert a Python value to a RON string for Blenvy.

    This is the top-level serializer for component values. Strings that look
    like RON constructs (enum variants, structs, lists) are passed through;
    plain strings are quoted.

    Examples::

        >>> to_ron(None)
        '()'
        >>> to_ron("Dynamic")
        'Dynamic'
        >>> to_ron({"max": 100, "current": 100})
        '(max: 100, current: 100)'
        >>> to_ron(True)
        'true'
        >>> to_ron([1.0, 2.0, 3.0])
        '[1.0, 2.0, 3.0]'
        >>> to_ron({"name": "Wall", "path": "blueprints/Wall.glb"})
        '(name: "Wall", path: "blueprints/Wall.glb")'
    """
    return _to_ron_impl(value, top_level=True)


def _to_ron_impl(value: Any, *, top_level: bool = False) -> str:
    """Internal RON serializer.

    Args:
        top_level: When True, strings starting with uppercase are treated
            as raw RON (enum variants). When False (struct field values),
            strings are always quoted unless they start with ( or [.
    """
    if value is None:
        return "()"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _format_float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if _is_raw_ron(stripped, top_level=top_level):
            return stripped
        return f'"{_escape_ron_string(value)}"'
    if isinstance(value, dict):
        return _dict_to_ron_struct(value)
    if isinstance(value, (list, tuple)):
        return _list_to_ron(value)
    msg = f"Cannot convert {type(value).__name__} to RON"
    raise TypeError(msg)


def _format_float(v: float) -> str:
    """Format float, ensuring it always has a decimal point."""
    s = repr(v)
    if "." not in s and "e" not in s and "E" not in s:
        s += ".0"
    return s


def _escape_ron_string(s: str) -> str:
    """Escape a string for RON (same as Rust string literals)."""
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _is_raw_ron(s: str, *, top_level: bool = False) -> bool:
    """Heuristic: is this string already a valid RON value?

    At top-level (component value position), strings that look like enum
    variants (``Dynamic``, ``Ball(0.3)``, ``Mesh``) are passed through.

    At field level (inside a dict/struct), only structural markers like
    ``(...)`` and ``[...]`` are passed through — regular words are quoted
    to avoid ambiguity with RON identifiers.
    """
    if not s:
        return False
    # Starts with ( or [ — struct/tuple/list — always raw
    if s[0] in ("(", "["):
        return True
    # Boolean keywords
    if s in ("true", "false"):
        return True
    # Number
    try:
        float(s)
        return True
    except ValueError:
        pass
    if s[0].isupper():
        # At top level: any uppercase start → enum variant (Dynamic, Ball(0.3), Mesh)
        if top_level:
            return True
        # At field level: uppercase + parens → enum with data (Srgba(...), Vec3(...))
        if "(" in s:
            return True
    return False


def _dict_to_ron_struct(d: dict[str, Any]) -> str:
    """Convert a dict to a RON struct: ``(key1: val1, key2: val2)``."""
    parts = []
    for key, val in d.items():
        parts.append(f"{key}: {_to_ron_impl(val, top_level=False)}")
    return "(" + ", ".join(parts) + ")"


def _list_to_ron(lst: list[Any] | tuple[Any, ...]) -> str:
    """Convert a list/tuple to a RON list: ``[val1, val2, ...]``."""
    parts = [_to_ron_impl(v, top_level=False) for v in lst]
    return "[" + ", ".join(parts) + "]"


def apply_bevy_components(
    obj: object,
    components: dict[str, BevyComponentValue],
) -> None:
    """Write Bevy components as Blender custom properties on *obj*.

    Components with ``::`` in their name (full type paths like
    ``avian3d::dynamics::rigid_body::RigidBody``) are packed into a single
    ``bevy_components`` JSON property — this is how Blenvy resolves them
    by full type path, avoiding short-name ambiguity.

    Simple short names (e.g. ``Health``, ``Pickable``) are written as
    individual custom properties.
    """
    import json as _json

    # Warn on unknown components if a registry is loaded (never block).
    reg = _active_registry
    if reg is not None:
        for comp_name in components:
            info = reg.find(comp_name)
            if info is None:
                suggestions = reg.suggest(comp_name)
                hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                warnings.warn(
                    f"Bevy component {comp_name!r} not found in registry.{hint}",
                    stacklevel=2,
                )

    full_path_comps: dict[str, str] = {}
    short_name_comps: dict[str, str] = {}

    for comp_name, comp_value in components.items():
        ron_str = to_ron(comp_value)
        if "::" in comp_name:
            full_path_comps[comp_name] = ron_str
        else:
            short_name_comps[comp_name] = ron_str

    # Short names → individual custom properties
    for name, ron_str in short_name_comps.items():
        obj[name] = ron_str  # type: ignore[index]

    # Full paths → single bevy_components JSON property
    if full_path_comps:
        existing = obj.get("bevy_components")  # type: ignore[union-attr]
        if existing and isinstance(existing, str):
            merged = _json.loads(existing)
            merged.update(full_path_comps)
        else:
            merged = full_path_comps
        obj["bevy_components"] = _json.dumps(merged)  # type: ignore[index]
