"""Bevy component registry loader and validator.

Parses the ``registry.json`` that Bevy exports (via ``bevy_reflect``) and
provides lookup, fuzzy matching, and lightweight value validation. This is
the same JSON Schema file that the Blenvy Blender add-on uses.

Usage::

    from blender_cli.blenvy_registry import BevyRegistry

    reg = BevyRegistry.load("assets/registry.json")
    print(reg.find("RigidBody"))
    print(reg.resolve("RigdBody"))  # raises with suggestions
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class UnknownComponentError(KeyError):
    """Raised when a component name cannot be resolved in the registry."""

    def __init__(self, name: str, suggestions: list[str] | None = None) -> None:
        self.name = name
        self.suggestions = suggestions or []
        parts = [f"Unknown component {name!r}"]
        if self.suggestions:
            parts.append(f"Did you mean: {', '.join(self.suggestions)}?")
        super().__init__(". ".join(parts))


class AmbiguousComponentError(KeyError):
    """Raised when a short name matches multiple full type paths."""

    def __init__(self, short_name: str, candidates: list[str]) -> None:
        self.short_name = short_name
        self.candidates = candidates
        super().__init__(
            f"Ambiguous short name {short_name!r} — matches: {', '.join(candidates)}"
        )


@dataclass(frozen=True, slots=True)
class ComponentInfo:
    """Metadata for a single Bevy ECS component."""

    long_name: str
    short_name: str
    schema: dict[str, Any]

    @property
    def type_info(self) -> str:
        return self.schema.get("typeInfo", "Unknown")

    @property
    def is_resource(self) -> bool:
        return self.schema.get("isResource", False)

    @property
    def fields(self) -> dict[str, Any]:
        return self.schema.get("properties", {})

    @property
    def required_fields(self) -> list[str]:
        return self.schema.get("required", [])

    @property
    def variants(self) -> list[str | dict]:
        return self.schema.get("oneOf", [])

    @property
    def variant_names(self) -> list[str]:
        """Variant names as plain strings (handles both string and dict forms)."""
        result: list[str] = []
        for v in self.variants:
            if isinstance(v, str):
                result.append(v)
            elif isinstance(v, dict):
                result.append(v.get("short_name", v.get("long_name", "")))
        return result

    def __repr__(self) -> str:
        return f"ComponentInfo({self.short_name!r}, path={self.long_name!r}, type={self.type_info})"


@dataclass
class BevyRegistry:
    """Bevy component registry loaded from ``registry.json``.

    Indexes all ``$defs`` entries where ``isComponent: true`` and provides
    lookup by short name or full Rust type path.
    """

    _defs: dict[str, Any] = field(repr=False)
    _components: dict[str, ComponentInfo] = field(default_factory=dict)
    _short_to_long: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> BevyRegistry:
        """Load a registry from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        defs = data.get("$defs", {})
        reg = cls(_defs=defs)
        reg._index()
        return reg

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BevyRegistry:
        """Build a registry from an already-parsed dict."""
        defs = data.get("$defs", {})
        reg = cls(_defs=defs)
        reg._index()
        return reg

    def _index(self) -> None:
        """Build component and short-name indexes."""
        self._components.clear()
        self._short_to_long.clear()
        for long_name, schema in self._defs.items():
            if not isinstance(schema, dict):
                continue
            if not schema.get("isComponent", False):
                continue
            short = schema.get("short_name", long_name.rsplit("::", 1)[-1])
            info = ComponentInfo(long_name=long_name, short_name=short, schema=schema)
            self._components[long_name] = info
            self._short_to_long.setdefault(short, []).append(long_name)

    def components(self, *, prefix: str | None = None) -> list[ComponentInfo]:
        """List all components, optionally filtered by path prefix."""
        result = list(self._components.values())
        if prefix:
            result = [c for c in result if c.long_name.startswith(prefix)]
        result.sort(key=lambda c: c.long_name)
        return result

    def find(self, name: str) -> ComponentInfo | None:
        """Look up by short name or full path. Returns ``None`` if not found."""
        # Full path lookup
        if name in self._components:
            return self._components[name]
        # Short name lookup
        longs = self._short_to_long.get(name)
        if longs and len(longs) == 1:
            return self._components[longs[0]]
        return None

    def resolve(self, name: str) -> str:
        """Resolve *name* to a full Rust type path.

        Raises :class:`AmbiguousComponentError` if the short name matches
        multiple paths, or :class:`UnknownComponentError` with fuzzy
        suggestions if not found at all.
        """
        # Full path
        if name in self._components:
            return name
        # Short name
        longs = self._short_to_long.get(name)
        if longs:
            if len(longs) == 1:
                return longs[0]
            raise AmbiguousComponentError(name, longs)
        raise UnknownComponentError(name, self.suggest(name))

    def suggest(self, partial: str, *, n: int = 5) -> list[str]:
        """Return up to *n* close matches for *partial* among short and long names."""
        candidates = list(self._short_to_long.keys()) + list(self._components.keys())
        return difflib.get_close_matches(partial, candidates, n=n, cutoff=0.5)

    def validate_value(self, component_path: str, value: Any) -> list[str]:
        """Validate a component value against the registry schema.

        Returns a list of warning strings (empty = valid). This is a
        lightweight check — it validates enum variants, struct field names,
        and tuple arity, not deep type-checking.
        """
        info = self.find(component_path)
        if info is None:
            return [f"Unknown component {component_path!r}"]
        return self._validate_against_schema(info.schema, value, component_path)

    def _validate_against_schema(
        self, schema: dict[str, Any], value: Any, ctx: str
    ) -> list[str]:
        """Check *value* against a single schema entry."""
        warnings: list[str] = []
        type_info = schema.get("typeInfo", "")

        if type_info == "Enum" and "oneOf" in schema:
            raw_variants = schema["oneOf"]
            # Variants can be strings ("Dynamic") or dicts with short_name
            variant_names: list[str] = []
            for v in raw_variants:
                if isinstance(v, str):
                    variant_names.append(v)
                elif isinstance(v, dict):
                    variant_names.append(v.get("short_name", v.get("long_name", "")))
            if isinstance(value, str):
                # Check bare variant name (e.g. "Cuboid" from "Cuboid(x: 1)")
                bare = value.split("(")[0].strip()
                if variant_names and bare not in variant_names:
                    warnings.append(
                        f"{ctx}: unknown variant {bare!r}, expected one of {variant_names}"
                    )

        elif type_info == "Struct" and "properties" in schema:
            if isinstance(value, dict):
                valid_fields = set(schema["properties"].keys())
                for k in value:
                    if k not in valid_fields:
                        warnings.append(
                            f"{ctx}: unknown field {k!r}, valid fields: {sorted(valid_fields)}"
                        )
                for req in schema.get("required", []):
                    if req not in value:
                        warnings.append(f"{ctx}: missing required field {req!r}")

        elif type_info == "TupleStruct" and "prefixItems" in schema:
            if isinstance(value, (list, tuple)):
                expected = len(schema["prefixItems"])
                if len(value) != expected:
                    warnings.append(
                        f"{ctx}: expected {expected} items, got {len(value)}"
                    )

        return warnings

    def schema(self, name: str) -> dict[str, Any] | None:
        """Return the raw JSON schema for a component (by short or full name)."""
        info = self.find(name)
        return info.schema if info else None
