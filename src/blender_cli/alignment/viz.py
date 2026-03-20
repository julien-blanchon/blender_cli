"""Alignment visual helpers (composition-focused)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Scene convention (Y-forward, Z-up) -> GLB convention (Y-up, -Z-forward)
M_SCENE_TO_GLB = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, -1, 0, 0],
        [0, 0, 0, 1],
    ],
    dtype=np.float64,
)


def save_combined_glb(
    room_path: str | Path,
    statue_path: str | Path,
    T_scene_obj: np.ndarray,
    out: str | Path,
) -> Path:
    """Export a combined GLB with room + object at a predicted transform."""
    import trimesh

    out = Path(out)
    T_glb_obj = M_SCENE_TO_GLB @ T_scene_obj

    room = trimesh.load(str(room_path))
    statue = trimesh.load(str(statue_path))

    if isinstance(statue, trimesh.Scene):
        for node_name in statue.graph.nodes_geometry:
            transform, geometry_name = statue.graph[node_name]
            geom = statue.geometry[geometry_name]
            geom.apply_transform(T_glb_obj @ transform)
    elif isinstance(statue, trimesh.Trimesh):
        statue.apply_transform(T_glb_obj)

    if isinstance(room, trimesh.Scene):
        combined = room
    else:
        combined = trimesh.Scene()
        combined.add_geometry(room, node_name="room")

    if isinstance(statue, trimesh.Scene):
        for name, geom in statue.geometry.items():
            combined.add_geometry(geom, node_name=f"object_{name}")
    else:
        combined.add_geometry(statue, node_name="object")

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out), file_type="glb")
    logger.info("Saved combined GLB -> %s", out)
    return out
