"""Animation keyframes — add, remove, list keyframes on project objects."""

from __future__ import annotations

import operator
from typing import Any

from blender_cli.project.project_file import resolve_object

# ---------------------------------------------------------------------------
# Animatable properties: name -> expected value type
# ---------------------------------------------------------------------------

ANIMATABLE_PROPERTIES: dict[str, str] = {
    "location": "vec3",  # [x, y, z]
    "rotation": "vec3",  # [x, y, z] (radians)
    "scale": "vec3",  # [x, y, z]
    "visible": "bool",
    "material.color": "color",  # [r, g, b, a]
    "material.metallic": "float",
    "material.roughness": "float",
    "material.alpha": "float",
    "material.emission_strength": "float",
}

INTERPOLATION_MODES = {"CONSTANT", "LINEAR", "BEZIER"}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_value(prop: str, value: Any) -> str | None:
    """Validate a keyframe value against its property type. Returns error or None."""
    vtype = ANIMATABLE_PROPERTIES.get(prop)
    if vtype is None:
        return f"Unknown property {prop!r}. Valid: {sorted(ANIMATABLE_PROPERTIES)}"

    if vtype == "vec3":
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return f"Property {prop!r} expects [x, y, z] (3 numbers), got {value!r}"
    elif vtype == "color":
        if not (
            isinstance(value, (list, tuple))
            and len(value) == 4
            and all(isinstance(v, (int, float)) for v in value)
        ):
            return f"Property {prop!r} expects [r, g, b, a] (4 numbers), got {value!r}"
    elif vtype == "float":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"Property {prop!r} expects float, got {type(value).__name__}"
    elif vtype == "bool" and not isinstance(value, bool):
        return f"Property {prop!r} expects bool, got {type(value).__name__}"

    return None




# ---------------------------------------------------------------------------
# Animation class — operations on project objects
# ---------------------------------------------------------------------------


class Animation:
    """Add, remove, and list keyframes on project objects."""

    @staticmethod
    def keyframe(
        project_data: dict[str, Any],
        object_ref: str | int,
        frame: int,
        prop: str,
        value: Any,
        interpolation: str = "BEZIER",
    ) -> dict[str, Any]:
        """Set a keyframe. Updates existing at same frame+prop, or adds new."""
        # Validate property
        if prop not in ANIMATABLE_PROPERTIES:
            msg = f"Unknown property {prop!r}. Valid: {sorted(ANIMATABLE_PROPERTIES)}"
            raise ValueError(msg)

        # Validate interpolation
        if interpolation not in INTERPOLATION_MODES:
            msg = f"Unknown interpolation {interpolation!r}. Valid: {sorted(INTERPOLATION_MODES)}"
            raise ValueError(msg)

        # Validate frame
        scene = project_data.get("scene", {})
        frame_start = scene.get("frame_start", 1)
        frame_end = scene.get("frame_end", 250)
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 1:
            msg = f"Frame must be a positive integer, got {frame!r}"
            raise ValueError(msg)
        if frame < frame_start or frame > frame_end:
            msg = f"Frame {frame} outside range [{frame_start}, {frame_end}]"
            raise ValueError(msg)

        # Validate value
        err = _validate_value(prop, value)
        if err:
            raise ValueError(err)

        _, obj = resolve_object(project_data, object_ref)

        if "keyframes" not in obj:
            obj["keyframes"] = []

        # Normalize value to list for vec3/color
        if isinstance(value, tuple):
            value = list(value)

        kf_entry = {
            "frame": frame,
            "property": prop,
            "value": value,
            "interpolation": interpolation,
        }

        # Conflict resolution: update existing at same frame+property
        for i, existing in enumerate(obj["keyframes"]):
            if existing["frame"] == frame and existing["property"] == prop:
                obj["keyframes"][i] = kf_entry
                _sort_keyframes(obj)
                return kf_entry

        obj["keyframes"].append(kf_entry)
        _sort_keyframes(obj)
        return kf_entry

    @staticmethod
    def remove(
        project_data: dict[str, Any],
        object_ref: str | int,
        frame: int,
        prop: str | None = None,
    ) -> list[dict[str, Any]]:
        """Remove keyframe(s) at a frame. If prop is None, removes all at that frame."""
        _, obj = resolve_object(project_data, object_ref)
        kfs = obj.get("keyframes", [])

        removed = []
        remaining = []
        for kf in kfs:
            if kf["frame"] == frame and (prop is None or kf["property"] == prop):
                removed.append(kf)
            else:
                remaining.append(kf)

        if not removed:
            msg = f"No keyframes at frame {frame}" + (
                f" for property {prop!r}" if prop else ""
            )
            raise KeyError(msg)

        obj["keyframes"] = remaining
        return removed

    @staticmethod
    def list(
        project_data: dict[str, Any],
        object_ref: str | int,
        prop: str | None = None,
    ) -> list[dict[str, Any]]:
        """List keyframes on an object, optionally filtered by property."""
        _, obj = resolve_object(project_data, object_ref)
        kfs = obj.get("keyframes", [])
        if prop is not None:
            kfs = [kf for kf in kfs if kf["property"] == prop]
        return kfs

    @staticmethod
    def set_frame_range(
        project_data: dict[str, Any],
        start: int,
        end: int,
    ) -> dict[str, int]:
        """Set the scene frame range."""
        if not isinstance(start, int) or isinstance(start, bool) or start < 1:
            msg = f"frame_start must be positive int, got {start!r}"
            raise ValueError(msg)
        if not isinstance(end, int) or isinstance(end, bool) or end < 1:
            msg = f"frame_end must be positive int, got {end!r}"
            raise ValueError(msg)
        if end < start:
            msg = f"frame_end ({end}) must be >= frame_start ({start})"
            raise ValueError(msg)
        project_data["scene"]["frame_start"] = start
        project_data["scene"]["frame_end"] = end
        return {"frame_start": start, "frame_end": end}

    @staticmethod
    def set_fps(
        project_data: dict[str, Any],
        fps: int,
    ) -> int:
        """Set the scene FPS."""
        if not isinstance(fps, int) or isinstance(fps, bool) or fps < 1:
            msg = f"FPS must be int >= 1, got {fps!r}"
            raise ValueError(msg)
        project_data["scene"]["fps"] = fps
        return fps


def _sort_keyframes(obj: dict[str, Any]) -> None:
    """Sort keyframes by frame number."""
    obj["keyframes"].sort(key=operator.itemgetter("frame", "property"))
