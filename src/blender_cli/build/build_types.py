"""Typed configuration objects for scene builders."""

from __future__ import annotations

from typing import TypeAlias

# JSON-like values used in props payloads.
JsonScalar: TypeAlias = str | int | float | bool | None
PropValue: TypeAlias = JsonScalar | list[JsonScalar]
