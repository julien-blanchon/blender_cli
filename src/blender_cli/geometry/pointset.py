"""PointSet — scatter, filter, snap, and randomize point placements."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

from blender_cli.core.diagnostics import logger
from blender_cli.snap import SnapResult, SnapSummary
from blender_cli.snap import snap as _snap
from blender_cli.types import RandomChoiceSpec, Vec3

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from blender_cli.geometry.mask import Mask
    from blender_cli.geometry.spline import Spline
    from blender_cli.scene.scene import Scene
    from blender_cli.snap import FilteredScene


#: Sentinel value used in :meth:`PointSet.from_coords` to mark a
#: coordinate axis as "don't care — let snap resolve it".
WILDCARD = "*"


# -- Per-point randomization result ------------------------------------


@dataclass(frozen=True, slots=True)
class PointAttrs:
    """Per-point attributes: scale, yaw, choice, and custom attrs."""

    scale: float = 1.0
    yaw: float = 0.0
    choice: int = 0


# -- PointSet ----------------------------------------------------------


class PointSet:
    """
    Immutable set of 3D points with per-point attributes and snap provenance.

    All mutation methods return new PointSet instances.
    Use :meth:`poisson` or :meth:`grid_jitter` to create, then chain
    *filter*, *snap*, *randomize*, *partition*.
    """

    __slots__ = (
        "_attrs",
        "_points",
        "_resolved_axes",
        "_snap_results",
        "_snap_summary",
        "_wildcards",
    )

    def __init__(
        self,
        points: list[Vec3],
        snap_results: list[SnapResult] | None = None,
        attrs: dict[str, Sequence[float | int | str]] | None = None,
        snap_summary: SnapSummary | None = None,
        *,
        wildcards: list[frozenset[int]] | None = None,
        resolved_axes: frozenset[int] = frozenset(),
    ) -> None:
        self._points = list(points)
        self._snap_results = list(snap_results) if snap_results else None
        self._snap_summary = snap_summary
        self._attrs: dict[str, Sequence[float | int | str]] = (
            dict(attrs) if attrs else {}
        )
        self._wildcards = wildcards
        self._resolved_axes = resolved_axes

    # -- factories -----------------------------------------------------

    @staticmethod
    def from_coords(coords: Sequence[tuple]) -> PointSet:
        """
        Create a PointSet from coordinate tuples, supporting ``"*"`` wildcards.

        Use ``"*"`` for axes that snap will resolve::

            # Single wildcard (axis-aligned snap)
            pts = PointSet.from_coords([(10, "*", 5)])
            pts.snap(scene, axis="-Y")

            # Multiple wildcards (arbitrary direction snap)
            pts = PointSet.from_coords([("*", "*", 50)])
            pts.with_x(100).with_y(200).snap(scene, axis=(1, 1, 0))

        Override wildcards with :meth:`with_x` / :meth:`with_y` / :meth:`with_z`
        for multi-layer scenes or to set specific ray start positions.
        """
        points: list[Vec3] = []
        wildcards: list[frozenset[int]] = []
        has_any = False

        for coord in coords:
            if len(coord) != 3:
                msg = f"Expected 3-element tuple, got {len(coord)}: {coord!r}"
                raise ValueError(msg)

            wild: set[int] = set()
            vals: list[float] = []
            for i, v in enumerate(coord):
                if v is ... or v == WILDCARD:
                    wild.add(i)
                    vals.append(0.0)
                else:
                    vals.append(float(v))

            if wild:
                has_any = True
            wildcards.append(frozenset(wild))
            points.append(Vec3(vals[0], vals[1], vals[2]))

        return PointSet(
            points,
            wildcards=wildcards if has_any else None,
        )

    @staticmethod
    def poisson(
        mask: Mask,
        density: float,
        seed: int = 0,
        radius: float | None = None,
        rng: random.Random | None = None,
    ) -> PointSet:
        """
        Generate blue-noise points via Bridson's algorithm within mask regions.

        *density* is points per square metre.  *radius* overrides the
        minimum distance between samples (default: derived from density).
        Points are only kept where ``mask.sample(x, y) > 0``.
        """
        if radius is None:
            radius = 1.0 / math.sqrt(density) if density > 0 else 1.0

        w_m = mask.width * mask.meters_per_px
        h_m = mask.height * mask.meters_per_px
        seed_used = seed if rng is None else rng.randint(0, 2**31 - 1)
        pts = _bridson_2d(w_m, h_m, radius, seed_used)

        # Filter by mask
        kept = [p for p in pts if mask.sample(p.x, p.y) > 0.0]
        return PointSet(kept)

    @staticmethod
    def grid_jitter(
        mask: Mask,
        step: float,
        jitter: float,
        seed: int,
        rng: random.Random | None = None,
    ) -> PointSet:
        """
        Generate jittered grid points within mask regions.

        *step* is the grid spacing in metres, *jitter* is the max random
        offset per axis (in metres).
        """
        rand = rng if rng is not None else random.Random(seed)
        w_m = mask.width * mask.meters_per_px
        h_m = mask.height * mask.meters_per_px

        pts: list[Vec3] = []
        x = step * 0.5
        while x < w_m:
            y = step * 0.5
            while y < h_m:
                jx = x + rand.uniform(-jitter, jitter)
                jy = y + rand.uniform(-jitter, jitter)
                if 0 <= jx < w_m and 0 <= jy < h_m and mask.sample(jx, jy) > 0.0:
                    pts.append(Vec3(jx, jy, 0.0))
                y += step
            x += step
        return PointSet(pts)

    # -- properties ----------------------------------------------------

    @property
    def points(self) -> list[Vec3]:
        """Copy of point positions."""
        return list(self._points)

    @property
    def count(self) -> int:
        return len(self._points)

    @property
    def snap_results(self) -> list[SnapResult] | None:
        """Per-point snap provenance (None if snap hasn't been called)."""
        return list(self._snap_results) if self._snap_results else None

    @property
    def snap_summary(self) -> SnapSummary | None:
        """Aggregated snap summary (None if snap hasn't been called)."""
        return self._snap_summary

    @property
    def wildcards(self) -> list[frozenset[int]] | None:
        """Per-point unresolved wildcard axes (``None`` if no wildcards)."""
        return self._wildcards

    @property
    def resolved_axes(self) -> frozenset[int]:
        """Axes resolved via :meth:`with_x` / :meth:`with_y` / :meth:`with_z`."""
        return self._resolved_axes

    @property
    def filter(self) -> _FilterProxy:
        """Access the filter chain."""
        return _FilterProxy(self)

    # -- iteration / slicing -------------------------------------------

    def __iter__(self):  # type: ignore[override]
        return iter(self._points)

    def __len__(self) -> int:
        return len(self._points)

    def take(self, n: int) -> PointSet:
        """Return a new PointSet with the first *n* points."""
        return self._slice(list(range(min(n, len(self._points)))))

    # -- transforms (all return new PointSet) --------------------------

    def with_x(self, x_value: float) -> PointSet:
        """Return a copy with all X set to *x_value*, clearing X wildcards."""
        new_pts = [Vec3(x_value, p.y, p.z) for p in self._points]
        new_wc = _clear_wildcard_axis(self._wildcards, 0)
        new_resolved = (
            self._resolved_axes | {0} if self._wildcards else self._resolved_axes
        )
        return self._copy(points=new_pts, wildcards=new_wc, resolved_axes=new_resolved)

    def with_y(self, y_value: float) -> PointSet:
        """Return a copy with all Y set to *y_value*, clearing Y wildcards."""
        new_pts = [Vec3(p.x, y_value, p.z) for p in self._points]
        new_wc = _clear_wildcard_axis(self._wildcards, 1)
        new_resolved = (
            self._resolved_axes | {1} if self._wildcards else self._resolved_axes
        )
        return self._copy(points=new_pts, wildcards=new_wc, resolved_axes=new_resolved)

    def with_z(self, z_value: float) -> PointSet:
        """Return a copy with all Z set to *z_value*, clearing Z wildcards."""
        new_pts = [Vec3(p.x, p.y, z_value) for p in self._points]
        new_wc = _clear_wildcard_axis(self._wildcards, 2)
        new_resolved = (
            self._resolved_axes | {2} if self._wildcards else self._resolved_axes
        )
        return self._copy(points=new_pts, wildcards=new_wc, resolved_axes=new_resolved)

    def snap(
        self,
        scene: Scene | FilteredScene,
        axis: str | tuple[float, float, float] = "-Z",
    ) -> PointSet:
        """
        Snap all points onto scene geometry and store provenance.

        Wildcards from :meth:`from_coords` are validated and resolved here,
        then cleared on the returned PointSet.
        """
        snap_results = _snap(
            self._points,
            scene,
            axis,
            _wildcards=self._wildcards,
            _resolved_axes=self._resolved_axes,
        )
        summary = snap_results.summary
        new_pts = [r.hit_pos for r in snap_results]
        msg = (
            f"[snap] {summary.hits}/{summary.total} hits, "
            f"{summary.misses} misses, "
            f"z_range=({summary.z_range[0]:.1f}, {summary.z_range[1]:.1f})"
        )
        if summary.warnings:
            msg += f" warnings={summary.warnings}"
        logger.debug(msg)
        return PointSet(new_pts, list(snap_results), self._attrs, summary)

    def randomize(
        self,
        scale: tuple[float, float] = (1.0, 1.0),
        yaw: tuple[float, float] = (0.0, 360.0),
        choice: RandomChoiceSpec | None = None,
        seed: int = 0,
        rng: random.Random | None = None,
    ) -> PointSet:
        """
        Assign random per-point scale, yaw, and variant choice.

        *scale* = (min, max) uniform range.
        *yaw* = (min, max) degrees.
        *choice* = {"variant": [...], "weights": [...]} for weighted choice.
        """
        rand = rng if rng is not None else random.Random(seed)
        n = len(self._points)

        scales = [rand.uniform(scale[0], scale[1]) for _ in range(n)]
        yaws = [rand.uniform(yaw[0], yaw[1]) for _ in range(n)]

        if choice is not None:
            variants = choice.get("variant", [0])
            weights = choice.get("weights", None)
            choices = rand.choices(variants, weights=weights, k=n)
        else:
            choices = [0] * n

        new_attrs = dict(self._attrs)
        new_attrs["scale"] = scales
        new_attrs["yaw"] = yaws
        new_attrs["choice"] = choices
        return self._copy(attrs=new_attrs)

    def set_attr(
        self,
        name: str,
        values_or_callable: Sequence[float | int | str]
        | Callable[[Vec3, int], float | int | str],
    ) -> PointSet:
        """
        Set a named attribute on all points.

        *values_or_callable* is either a list of values (len == count)
        or a callable ``(point, index) -> value``.
        """
        if callable(values_or_callable):
            values = [values_or_callable(p, i) for i, p in enumerate(self._points)]
        else:
            values = list(values_or_callable)
            if len(values) != len(self._points):
                msg = f"values length {len(values)} != point count {len(self._points)}"
                raise ValueError(msg)
        new_attrs = dict(self._attrs)
        new_attrs[name] = values
        return self._copy(attrs=new_attrs)

    def partition(self, cell_size: float) -> PointSet:
        """
        Split points into spatial cells by adding a ``cell_id`` attribute.

        Cell IDs are computed as ``(floor(x/cell_size), floor(y/cell_size))``
        encoded as a string ``"cx_cy"``.
        """
        cell_ids: list[str] = []
        for p in self._points:
            cx = math.floor(p.x / cell_size)
            cy = math.floor(p.y / cell_size)
            cell_ids.append(f"{cx}_{cy}")
        new_attrs = dict(self._attrs)
        new_attrs["cell_id"] = cell_ids
        return self._copy(attrs=new_attrs)

    def attr(self, name: str) -> list[float | int | str]:
        """Retrieve a named attribute list."""
        if name not in self._attrs:
            msg = f"Attribute '{name}' not set"
            raise KeyError(msg)
        return list(self._attrs[name])

    # -- internal helpers ----------------------------------------------

    def _copy(
        self,
        *,
        points: list[Vec3] | None = None,
        snap_results: list[SnapResult] | None = ...,  # type: ignore[assignment]
        attrs: dict[str, Sequence[float | int | str]] | None = ...,  # type: ignore[assignment]
        snap_summary: SnapSummary | None = ...,  # type: ignore[assignment]
        wildcards: list[frozenset[int]] | None = ...,  # type: ignore[assignment]
        resolved_axes: frozenset[int] | None = None,
    ) -> PointSet:
        """
        Return a shallow copy, overriding only the specified fields.

        Uses ``...`` (Ellipsis) as the default sentinel so that ``None``
        can be passed explicitly to clear a field.
        """
        return PointSet(
            points if points is not None else self._points,
            self._snap_results if snap_results is ... else snap_results,
            self._attrs if attrs is ... else attrs,
            self._snap_summary if snap_summary is ... else snap_summary,
            wildcards=self._wildcards if wildcards is ... else wildcards,
            resolved_axes=resolved_axes
            if resolved_axes is not None
            else self._resolved_axes,
        )

    def _slice(self, indices: list[int]) -> PointSet:
        """Return a subset by index list, preserving all metadata."""
        pts = [self._points[i] for i in indices]
        sr = [self._snap_results[i] for i in indices] if self._snap_results else None
        attrs: dict[str, Sequence[float | int | str]] = {
            k: [v[i] for i in indices] for k, v in self._attrs.items()
        }
        wc = [self._wildcards[i] for i in indices] if self._wildcards else None
        return PointSet(
            pts,
            sr,
            attrs,
            self._snap_summary,
            wildcards=wc,
            resolved_axes=self._resolved_axes,
        )


# -- Filter proxy ------------------------------------------------------


class _FilterProxy:
    """Chainable filter operations accessed via ``pts.filter``."""

    __slots__ = ("_ps",)

    def __init__(self, ps: PointSet) -> None:
        self._ps = ps

    def by_mask(self, mask: Mask, min_value: float = 0.5) -> PointSet:
        """Keep points where ``mask.sample(x, y) >= min_value``."""
        keep = [
            i
            for i, p in enumerate(self._ps._points)
            if mask.sample(p.x, p.y) >= min_value
        ]
        return self._ps._slice(keep)

    def by_surface_angle(self, max_deg: float) -> PointSet:
        """
        Keep points where surface normal angle from vertical <= *max_deg*.

        Requires :meth:`PointSet.snap` to have been called first.
        """
        if self._ps._snap_results is None:
            msg = (
                "by_surface_angle requires snap() to be called first "
                "(need hit_normal provenance)"
            )
            raise RuntimeError(msg)
        up = Vec3(0.0, 0.0, 1.0)
        cos_limit = math.cos(math.radians(max_deg))
        keep: list[int] = []
        for i, sr in enumerate(self._ps._snap_results):
            if not sr.hit or sr.hit_normal is None:
                continue
            n = sr.hit_normal.normalized()
            cos_angle = abs(n.dot(up))
            if cos_angle >= cos_limit:
                keep.append(i)
        return self._ps._slice(keep)

    def distance_to_spline(
        self, spline: Spline, min_d: float, max_d: float
    ) -> PointSet:
        """Keep points within [min_d, max_d] XY distance of *spline*."""
        # Pre-sample spline for fast distance queries.
        n_samples = max(64, int(spline.length() / 1.0))
        spine_pts = [spline.sample(t / n_samples) for t in range(n_samples + 1)]

        keep: list[int] = []
        for i, p in enumerate(self._ps._points):
            d = _min_xy_distance(p, spine_pts)
            if min_d <= d <= max_d:
                keep.append(i)
        return self._ps._slice(keep)


# -- Wildcard helpers --------------------------------------------------


def _clear_wildcard_axis(
    wildcards: list[frozenset[int]] | None, axis_idx: int
) -> list[frozenset[int]] | None:
    """
    Remove *axis_idx* from every per-point wildcard set.

    Returns ``None`` if *wildcards* is ``None`` or no wildcards remain.
    """
    if wildcards is None:
        return None
    cleared = [wc - {axis_idx} for wc in wildcards]
    if any(cleared):
        return cleared
    return None


# -- Bridson's Poisson disk sampling -----------------------------------


def _bridson_2d(
    width: float, height: float, radius: float, seed: int, k: int = 30
) -> list[Vec3]:
    """Bridson's algorithm for O(n) blue-noise point generation in 2D."""
    rng = random.Random(seed)
    cell = radius / math.sqrt(2)
    cols = math.ceil(width / cell)
    rows = math.ceil(height / cell)

    grid: list[int | None] = [None] * (cols * rows)
    points: list[Vec3] = []
    active: list[int] = []

    def _grid_idx(x: float, y: float) -> int:
        return int(x / cell) + int(y / cell) * cols

    # Seed point
    sx, sy = rng.uniform(0, width), rng.uniform(0, height)
    p0 = Vec3(sx, sy, 0.0)
    points.append(p0)
    active.append(0)
    gi = _grid_idx(sx, sy)
    if 0 <= gi < len(grid):
        grid[gi] = 0

    while active:
        idx = rng.randrange(len(active))
        pi = active[idx]
        base = points[pi]
        found = False

        for _ in range(k):
            angle = rng.uniform(0, 2 * math.pi)
            dist = rng.uniform(radius, 2 * radius)
            nx = base.x + dist * math.cos(angle)
            ny = base.y + dist * math.sin(angle)

            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue

            gc = int(nx / cell)
            gr = int(ny / cell)

            # Check neighbours in a 5x5 grid window
            ok = True
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    rr, cc = gr + dr, gc + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        ni = grid[cc + rr * cols]
                        if ni is not None:
                            dx = points[ni].x - nx
                            dy = points[ni].y - ny
                            if dx * dx + dy * dy < radius * radius:
                                ok = False
                                break
                    if not ok:
                        break

            if ok:
                new_idx = len(points)
                points.append(Vec3(nx, ny, 0.0))
                active.append(new_idx)
                gi2 = gc + gr * cols
                if 0 <= gi2 < len(grid):
                    grid[gi2] = new_idx
                found = True
                break

        if not found:
            active.pop(idx)

    return points


# -- Helpers -----------------------------------------------------------


def _min_xy_distance(p: Vec3, spine_pts: list[Vec3]) -> float:
    """Minimum XY (2D) distance from *p* to a polyline defined by *spine_pts*."""
    best = float("inf")
    px, py = p.x, p.y
    for i in range(len(spine_pts) - 1):
        a = spine_pts[i]
        b = spine_pts[i + 1]
        d = _point_segment_dist_2d(px, py, a.x, a.y, b.x, b.y)
        best = min(best, d)
    return best


def _point_segment_dist_2d(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Distance from point (px,py) to line segment (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.sqrt((px - ax) ** 2 + (py - ay) ** 2)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    cx, cy = ax + t * dx, ay + t * dy
    return math.sqrt((px - cx) ** 2 + (py - cy) ** 2)
