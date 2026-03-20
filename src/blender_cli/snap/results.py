"""Snap result types — SnapResult, SnapSummary, SnapResults, SnapPolicy, SnapObjectResult."""

from __future__ import annotations

from collections import Counter, UserList
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from blender_cli.types import Vec3

# -- SnapResult --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapResult:
    """Result of snapping a single point to scene geometry."""

    point: Vec3
    """Original input point."""
    hit: bool
    """True if the ray hit geometry."""
    hit_pos: Vec3
    """Snapped position (original point if miss)."""
    hit_normal: Vec3 | None
    """Surface normal at hit (None if miss)."""
    hit_uid: str | None
    """UID of the hit scene object (None if miss or object untracked)."""
    hit_distance: float
    """Distance from ray origin to hit point (-1 if miss)."""
    snap_axis: str | tuple[float, float, float]
    """Axis or direction used for snapping."""
    ray_origin: Vec3
    """Ray origin point."""


# -- SnapSummary -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapSummary:
    """
    Aggregated feedback from a batch of snap results.

    Built via :meth:`from_results` — gives the agent a quick overview
    of what happened: hit/miss counts, which surfaces were hit, Z range,
    and actionable warnings.
    """

    total: int
    """Total points snapped."""
    hits: int
    """Number of rays that hit geometry."""
    misses: int
    """Number of rays that missed."""
    hit_surfaces: dict[str, int]
    """Map of hit object UID -> hit count."""
    z_range: tuple[float, float]
    """``(min_z, max_z)`` of successful hit positions."""
    warnings: list[str] = field(default_factory=list)
    """Actionable warnings for the agent."""

    @classmethod
    def from_results(cls, results: list[SnapResult]) -> SnapSummary:
        """Build a summary from a list of :class:`SnapResult`."""
        total = len(results)
        hit_results = [r for r in results if r.hit]
        hits = len(hit_results)
        misses = total - hits

        # Surface hit counts (skip None UIDs — untracked objects).
        surface_counts: dict[str, int] = dict(
            Counter(r.hit_uid for r in hit_results if r.hit_uid is not None)
        )

        # Z range of hits.
        if hit_results:
            z_vals = [r.hit_pos.z for r in hit_results]
            z_range = (min(z_vals), max(z_vals))
        else:
            z_range = (0.0, 0.0)

        # Warnings.
        warn_list: list[str] = []
        if total > 0:
            miss_pct = misses / total * 100
            if miss_pct > 20:
                warn_list.append(
                    f"High miss rate ({miss_pct:.0f}%): "
                    f"check ray origins and scene geometry"
                )
        if hit_results:
            z_spread = z_range[1] - z_range[0]
            if z_spread > 50.0:
                warn_list.append(
                    f"Large Z spread ({z_spread:.0f}m): "
                    f"objects may be on different terrain levels"
                )

        return cls(
            total=total,
            hits=hits,
            misses=misses,
            hit_surfaces=surface_counts,
            z_range=z_range,
            warnings=warn_list,
        )


class SnapResults(UserList[SnapResult]):
    """
    List of :class:`SnapResult` with an attached :class:`SnapSummary`.

    Behaves exactly like ``list[SnapResult]`` (indexing, iteration, len)
    so existing code is unaffected.  New code can access ``.summary``.
    """

    summary: SnapSummary

    def __init__(self, results: list[SnapResult], summary: SnapSummary) -> None:
        super().__init__(results)
        self.summary = summary


# -- SnapPolicy --------------------------------------------------------


class SnapPolicy(Enum):
    """
    How to resolve height when snapping a mesh object to scene geometry.

    All policies cast rays from the object's actual mesh vertices — no grid
    approximation.
    """

    FIRST = "first"
    """Per-vertex rays, first contact — object rests on the highest terrain
    point under any vertex (minimum drop distance)."""

    LAST = "last"
    """Per-vertex rays, through-cast — object sinks to the deepest contact
    across all vertices (maximum drop distance)."""

    HIGHEST = "highest"
    """Per-vertex rays — object origin Z set to max hit Z."""

    LOWEST = "lowest"
    """Per-vertex rays — object origin Z set to min hit Z."""

    AVERAGE = "average"
    """Per-vertex rays — object origin Z set to mean hit Z."""

    ORIENT = "orient"
    """Per-vertex rays — object rests on terrain (min travel, like FIRST)
    plus rotation from mean surface normal.  Prevents clipping while
    orienting the mesh to match the slope."""


# -- SnapObjectResult --------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapObjectResult:
    """Result of snapping an object to scene geometry via mesh vertices."""

    position: Vec3
    """Final snapped position for the object origin."""
    rotation: tuple[float, float, float] | None
    """Euler XYZ radians (ORIENT only, None otherwise)."""
    policy: SnapPolicy
    """Policy used."""
    vertex_hits: int
    """How many vertices hit geometry."""
    vertex_total: int
    """Total vertices cast."""
    z_min: float
    """Min Z across all vertex hits."""
    z_max: float
    """Max Z across all vertex hits."""
    z_mean: float
    """Mean Z across all vertex hits."""
    z_spread: float
    """z_max - z_min."""
    drop_distance: float
    """How far the object was moved along the snap axis."""
    penetration_depth: float = 0.0
    """Max depth any vertex penetrates below terrain surface (>0 means clipping)."""
    penetrating_vertices: int = 0
    """Count of vertices that end up below the terrain surface."""
