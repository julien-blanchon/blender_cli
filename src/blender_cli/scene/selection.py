"""Selection & Query System — filter, transform, and inspect scene objects."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import bpy

from blender_cli.core.metadata import decode_dict, decode_set

if TYPE_CHECKING:
    from blender_cli.types import PropValue

# ---------------------------------------------------------------------------
# Metadata helpers (standalone to avoid circular import with Scene)
# ---------------------------------------------------------------------------

_Predicate = Callable[[bpy.types.Object], bool]


def _get_tags(obj: bpy.types.Object) -> set[str]:
    return decode_set(obj.get("_tags"))


def _get_annotations(obj: bpy.types.Object) -> set[str]:
    return decode_set(obj.get("_annotations"))


def _get_props(obj: bpy.types.Object) -> dict[str, object]:
    return decode_dict(obj.get("_props"))


# ---------------------------------------------------------------------------
# Query DSL — tokenizer + recursive-descent parser
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        '([^']*)'           |   # group 1: single-quoted string
        (==)                |   # group 2: equality
        ([.()&|!])          |   # group 3: single-char ops
        (-?\d+(?:\.\d+)?)  |   # group 4: number
        ([a-zA-Z_]\w*)          # group 5: identifier
    )
    """,
    re.VERBOSE,
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    """
    Tokenize *expr* into ``(kind, value)`` pairs.

    Kinds: ``s`` string, ``n`` number, ``o`` operator, ``i`` identifier.
    """
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(expr):
        if m.group(1) is not None:
            tokens.append(("s", m.group(1)))
        elif m.group(2):
            tokens.append(("o", "=="))
        elif m.group(3):
            tokens.append(("o", m.group(3)))
        elif m.group(4) is not None:
            tokens.append(("n", m.group(4)))
        elif m.group(5):
            tokens.append(("i", m.group(5)))
    return tokens


def parse_query(expr: str) -> _Predicate:
    """
    Parse a query DSL expression into a predicate function.

    Grammar (precedence: ``!`` > ``&`` > ``|``)::

        expr     → or_expr
        or_expr  → and_expr ('|' and_expr)*
        and_expr → not_expr ('&' not_expr)*
        not_expr → '!' not_expr | atom
        atom     → tags.has('…')
                  | annotations.has('…')
                  | props.<name> == '…' | <number>
                  | '(' expr ')'
    """
    tokens = _tokenize(expr)
    pos = [0]

    def peek() -> tuple[str, str] | None:
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def advance() -> tuple[str, str]:
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def expect(kind: str, val: str) -> None:
        t = advance()
        if t != (kind, val):
            msg = f"Expected ('{kind}', '{val}'), got {t}"
            raise ValueError(msg)

    def parse_or() -> _Predicate:
        left: _Predicate = parse_and()
        while peek() == ("o", "|"):
            advance()
            right = parse_and()
            a, b = left, right
            left = lambda obj, _a=a, _b=b: _a(obj) or _b(obj)  # noqa: E731
        return left

    def parse_and() -> _Predicate:
        left: _Predicate = parse_not()
        while peek() == ("o", "&"):
            advance()
            right = parse_not()
            a, b = left, right
            left = lambda obj, _a=a, _b=b: _a(obj) and _b(obj)  # noqa: E731
        return left

    def parse_not() -> _Predicate:
        if peek() == ("o", "!"):
            advance()
            inner = parse_not()
            return lambda obj, _i=inner: not _i(obj)
        return parse_atom()

    def parse_atom() -> _Predicate:
        if peek() == ("o", "("):
            advance()
            result = parse_or()
            expect("o", ")")
            return result

        kind, val = advance()
        if kind != "i":
            msg = f"Expected identifier, got ('{kind}', '{val}')"
            raise ValueError(msg)

        if val == "tags":
            expect("o", ".")
            expect("i", "has")
            expect("o", "(")
            _, tag = advance()
            expect("o", ")")
            return lambda obj, _t=tag: _t in _get_tags(obj)

        if val == "annotations":
            expect("o", ".")
            expect("i", "has")
            expect("o", "(")
            _, ann = advance()
            expect("o", ")")
            return lambda obj, _a=ann: _a in _get_annotations(obj)

        if val == "props":
            expect("o", ".")
            _, prop_name = advance()
            expect("o", "==")
            vkind, vval = advance()
            pval: PropValue = (
                (float(vval) if "." in vval else int(vval)) if vkind == "n" else vval
            )
            return lambda obj, _n=prop_name, _v=pval: _get_props(obj).get(_n) == _v

        msg = f"Unknown query target: {val!r}"
        raise ValueError(msg)

    result = parse_or()
    if pos[0] < len(tokens):
        msg = f"Unexpected token: {tokens[pos[0]]}"
        raise ValueError(msg)
    return result


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


class Selection:
    """A set of scene objects matching a query. Iterable and countable."""

    __slots__ = ("_objects",)

    def __init__(self, objects: list[bpy.types.Object]) -> None:
        self._objects = list(objects)

    def __iter__(self) -> Iterator[bpy.types.Object]:
        return iter(self._objects)

    def __len__(self) -> int:
        return len(self._objects)

    def count(self) -> int:
        """Number of matched objects."""
        return len(self._objects)

    def first(self) -> bpy.types.Object | None:
        """First matched object, or ``None``."""
        return self._objects[0] if self._objects else None

    def uids(self) -> list[str]:
        """UIDs of all matched objects."""
        out: list[str] = []
        for obj in self._objects:
            v = obj.get("_uid")
            if v is not None:
                out.append(str(v))
        return out


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


class Transform:
    """Chainable transform operations on a selection of objects."""

    __slots__ = ("_objects",)

    def __init__(self, objects: list[bpy.types.Object]) -> None:
        self._objects = objects

    def move(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> Transform:
        """Translate all objects by *(dx, dy, dz)*."""
        for obj in self._objects:
            obj.location.x += dx
            obj.location.y += dy
            obj.location.z += dz
        return self

    def rotate(
        self,
        *,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
    ) -> Transform:
        """
        Rotate all objects by the given Euler angles (degrees).

        Parameters
        ----------
            yaw_deg: Rotation around Z axis.
            pitch_deg: Rotation around X axis.
            roll_deg: Rotation around Y axis.

        """
        rz = math.radians(yaw_deg)
        rx = math.radians(pitch_deg)
        ry = math.radians(roll_deg)
        for obj in self._objects:
            if rx:
                obj.rotation_euler.x += rx
            if ry:
                obj.rotation_euler.y += ry
            if rz:
                obj.rotation_euler.z += rz
        return self

    def scale(self, sx: float = 1.0, sy: float = 1.0, sz: float = 1.0) -> Transform:
        """Scale all objects by *(sx, sy, sz)*."""
        for obj in self._objects:
            obj.scale.x *= sx
            obj.scale.y *= sy
            obj.scale.z *= sz
        return self
