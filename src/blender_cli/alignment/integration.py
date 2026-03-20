"""Compose estimated object pose into a scene GLB."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .types import ComposeOptions, ComposeResult
from .viz import save_combined_glb


def load_pose_result(pose_result: str | Path | dict[str, Any]) -> dict[str, Any]:
    """Load pose_result payload from file path or return dict unchanged."""
    if isinstance(pose_result, dict):
        return pose_result
    pose_path = Path(pose_result).resolve()
    if not pose_path.is_file():
        msg = f"Pose result file not found: {pose_path}"
        raise FileNotFoundError(msg)
    return json.loads(pose_path.read_text(encoding="utf-8"))


def integrate_pose_result(
    *,
    scene_path: str | Path,
    object_path: str | Path,
    pose_result: str | Path | dict[str, Any],
    output_path: str | Path,
) -> ComposeResult:
    """Create a combined GLB by applying `T_world_obj` from pose result."""
    resolved_scene = Path(scene_path).resolve()
    resolved_object = Path(object_path).resolve()
    payload = load_pose_result(pose_result)
    if "T_world_obj" not in payload:
        msg = "pose_result is missing required key `T_world_obj`"
        raise KeyError(msg)

    transform = np.asarray(payload["T_world_obj"], dtype=np.float64)
    if transform.shape != (4, 4):
        msg = f"T_world_obj must be a 4x4 matrix, got shape {transform.shape}"
        raise ValueError(msg)

    resolved_output = Path(output_path).resolve()
    save_combined_glb(
        room_path=resolved_scene,
        statue_path=resolved_object,
        T_scene_obj=transform,
        out=resolved_output,
    )
    pose_result_path = (
        Path(pose_result).resolve()
        if isinstance(pose_result, (str, Path))
        else Path("<inline-pose-result>")
    )
    return ComposeResult(
        output_scene_path=resolved_output,
        pose_result_path=pose_result_path,
        scene_path=resolved_scene,
        object_path=resolved_object,
        transform=transform.tolist(),
    )


def compose_from_directory(
    directory: str | Path,
    options: ComposeOptions | None = None,
) -> ComposeResult:
    """Compose using conventional filenames from an example directory."""
    opts = options or ComposeOptions()
    base_dir = Path(directory).resolve()
    output_dir = base_dir / opts.output_dir_name
    return integrate_pose_result(
        scene_path=base_dir / opts.scene_name,
        object_path=base_dir / opts.object_name,
        pose_result=output_dir / opts.pose_result_name,
        output_path=output_dir / opts.output_name,
    )
