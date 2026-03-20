"""
CameraPath — pre-computed camera animation paths.

A :class:`CameraPath` is a pure data structure: a list of
:class:`CameraKeyframe` entries (position + look_at), one per frame.
All interpolation is computed at construction time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from blender_cli.types import Vec3

if TYPE_CHECKING:
    from blender_cli.geometry.spline import Spline


# ---------------------------------------------------------------------------
# Keyframe
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraKeyframe:
    """Single camera keyframe: position and look-at target."""

    position: Vec3
    look_at: Vec3


# ---------------------------------------------------------------------------
# Interpolation helpers (single source of truth)
# ---------------------------------------------------------------------------


def _smoothstep(t: float) -> float:
    """Hermite ease-in-out: 3t² − 2t³, clamped to [0, 1]."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _cr_eval_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: float) -> Vec3:
    """Evaluate Catmull-Rom spline at *t* ∈ [0, 1] between *p1* and *p2*."""
    t2 = t * t
    t3 = t2 * t
    x = 0.5 * (
        (2.0 * p1.x)
        + (-p0.x + p2.x) * t
        + (2.0 * p0.x - 5.0 * p1.x + 4.0 * p2.x - p3.x) * t2
        + (-p0.x + 3.0 * p1.x - 3.0 * p2.x + p3.x) * t3
    )
    y = 0.5 * (
        (2.0 * p1.y)
        + (-p0.y + p2.y) * t
        + (2.0 * p0.y - 5.0 * p1.y + 4.0 * p2.y - p3.y) * t2
        + (-p0.y + 3.0 * p1.y - 3.0 * p2.y + p3.y) * t3
    )
    z = 0.5 * (
        (2.0 * p1.z)
        + (-p0.z + p2.z) * t
        + (2.0 * p0.z - 5.0 * p1.z + 4.0 * p2.z - p3.z) * t2
        + (-p0.z + 3.0 * p1.z - 3.0 * p2.z + p3.z) * t3
    )
    return Vec3(x, y, z)


# ---------------------------------------------------------------------------
# CameraPath
# ---------------------------------------------------------------------------


class CameraPath:
    """
    Pre-computed camera animation path.

    Each instance holds a list of :class:`CameraKeyframe` entries — one per
    frame.  Build via factory methods, then pass to
    :meth:`RenderContext.animate`.
    """

    __slots__ = ("_frames",)

    def __init__(self, frames: list[CameraKeyframe]) -> None:
        if not frames:
            msg = "CameraPath requires at least one frame"
            raise ValueError(msg)
        self._frames = list(frames)

    # -- Factory methods ------------------------------------------------------

    @classmethod
    def orbit(
        cls,
        center: Vec3,
        radius: float,
        elevation: float = 35.0,
        frames: int = 24,
    ) -> CameraPath:
        """
        Create a 360° orbit around *center*.

        Parameters
        ----------
        center:
            World-space point to orbit around.
        radius:
            Orbit radius in scene units.
        elevation:
            Camera elevation in degrees above the horizon.
        frames:
            Number of frames in the full orbit.

        """
        elev = math.radians(elevation)
        keyframes: list[CameraKeyframe] = []
        for i in range(frames):
            azim = 2 * math.pi * i / frames
            pos = Vec3(
                center.x + radius * math.cos(elev) * math.cos(azim),
                center.y - radius * math.cos(elev) * math.sin(azim),
                center.z + radius * math.sin(elev),
            )
            keyframes.append(CameraKeyframe(position=pos, look_at=center))
        return cls(keyframes)

    @classmethod
    def orbit_grid(
        cls,
        center: Vec3,
        radius: float,
        elevations: list[float] | None = None,
        azimuths: int = 8,
    ) -> CameraPath:
        """
        Create a grid of views at multiple elevations and azimuths.

        Parameters
        ----------
        center:
            World-space point to orbit around.
        radius:
            Orbit radius.
        elevations:
            Elevation angles in degrees (default ``[15, 35, 60]``).
        azimuths:
            Number of azimuth steps around full 360°.

        """
        if elevations is None:
            elevations = [15.0, 35.0, 60.0]

        keyframes: list[CameraKeyframe] = []
        for elev_deg in elevations:
            elev = math.radians(elev_deg)
            for ai in range(azimuths):
                azim = 2 * math.pi * ai / azimuths
                pos = Vec3(
                    center.x + radius * math.cos(elev) * math.cos(azim),
                    center.y - radius * math.cos(elev) * math.sin(azim),
                    center.z + radius * math.sin(elev),
                )
                keyframes.append(CameraKeyframe(position=pos, look_at=center))
        return cls(keyframes)

    @classmethod
    def from_spline(
        cls,
        spline: Spline,
        frames: int = 24,
        look: Literal["ahead", "target"] = "ahead",
        look_target: Vec3 | None = None,
    ) -> CameraPath:
        """
        Create a path following a :class:`Spline`.

        Parameters
        ----------
        spline:
            The spline to follow.
        frames:
            Number of frames to sample.
        look:
            ``"ahead"`` points along the spline tangent;
            ``"target"`` points at *look_target*.
        look_target:
            Target position for ``look="target"`` mode.

        """
        keyframes: list[CameraKeyframe] = []
        for i in range(frames):
            t = i / max(frames - 1, 1)
            pos = spline.sample(t)
            if look == "ahead":
                tang = spline.tangent(t)
                la = Vec3(pos.x + tang.x, pos.y + tang.y, pos.z + tang.z)
            else:
                if look_target is None:
                    msg = "look_target required for look='target' mode"
                    raise ValueError(msg)
                la = look_target
            keyframes.append(CameraKeyframe(position=pos, look_at=la))
        return cls(keyframes)

    @classmethod
    def from_keyframes(
        cls,
        positions: list[Vec3],
        look_ats: list[Vec3],
        hold_frames: int = 1,
        transition_frames: int = 0,
    ) -> CameraPath:
        """
        Create a path from explicit keyframe positions and look-at targets.

        Each keyframe is held for *hold_frames*, with Catmull-Rom +
        smoothstep interpolation of *transition_frames* between them.
        """
        if len(positions) != len(look_ats):
            msg = f"positions ({len(positions)}) and look_ats ({len(look_ats)}) must match"
            raise ValueError(msg)
        if not positions:
            msg = "Need at least one keyframe"
            raise ValueError(msg)

        if len(positions) == 1:
            return cls(
                [CameraKeyframe(position=positions[0], look_at=look_ats[0])]
                * hold_frames
            )

        frames: list[CameraKeyframe] = []
        n = len(positions)

        for ki in range(n):
            # Hold phase
            frames.extend(
                CameraKeyframe(position=positions[ki], look_at=look_ats[ki])
                for _ in range(hold_frames)
            )

            # Transition to next (skip after last)
            if ki < n - 1 and transition_frames > 0:
                # Catmull-Rom control points (mirror at endpoints)
                if ki == 0:
                    p0_pos = Vec3(
                        2.0 * positions[0].x - positions[1].x,
                        2.0 * positions[0].y - positions[1].y,
                        2.0 * positions[0].z - positions[1].z,
                    )
                    p0_look = Vec3(
                        2.0 * look_ats[0].x - look_ats[1].x,
                        2.0 * look_ats[0].y - look_ats[1].y,
                        2.0 * look_ats[0].z - look_ats[1].z,
                    )
                else:
                    p0_pos = positions[ki - 1]
                    p0_look = look_ats[ki - 1]

                if ki + 2 < n:
                    p3_pos = positions[ki + 2]
                    p3_look = look_ats[ki + 2]
                else:
                    p3_pos = Vec3(
                        2.0 * positions[ki + 1].x - positions[ki].x,
                        2.0 * positions[ki + 1].y - positions[ki].y,
                        2.0 * positions[ki + 1].z - positions[ki].z,
                    )
                    p3_look = Vec3(
                        2.0 * look_ats[ki + 1].x - look_ats[ki].x,
                        2.0 * look_ats[ki + 1].y - look_ats[ki].y,
                        2.0 * look_ats[ki + 1].z - look_ats[ki].z,
                    )

                for fi in range(transition_frames):
                    raw_t = (fi + 1) / (transition_frames + 1)
                    t = _smoothstep(raw_t)
                    pos = _cr_eval_vec3(
                        p0_pos, positions[ki], positions[ki + 1], p3_pos, t
                    )
                    look = _cr_eval_vec3(
                        p0_look, look_ats[ki], look_ats[ki + 1], p3_look, t
                    )
                    frames.append(CameraKeyframe(position=pos, look_at=look))

        return cls(frames)

    # -- Composition ----------------------------------------------------------

    def chain(self, other: CameraPath) -> CameraPath:
        """Concatenate this path with *other*."""
        return CameraPath(self._frames + other._frames)

    def reversed(self) -> CameraPath:
        """Return a new path with frames in reverse order."""
        return CameraPath(list(reversed(self._frames)))

    def subsample(self, n: int) -> CameraPath:
        """Keep every *n*-th frame."""
        if n < 1:
            msg = "subsample step must be >= 1"
            raise ValueError(msg)
        return CameraPath(self._frames[::n])

    # -- Access ---------------------------------------------------------------

    @property
    def frames(self) -> list[CameraKeyframe]:
        """All keyframes in order."""
        return list(self._frames)

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, index: int) -> CameraKeyframe:
        return self._frames[index]

    def __repr__(self) -> str:
        return f"CameraPath({len(self._frames)} frames)"
