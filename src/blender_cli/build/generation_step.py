"""GenerationStep helper for consistent step I/O, rendering, and determinism."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import bpy

from blender_cli.build.context import BuildContext
from blender_cli.scene.scene import Scene


def _step_index_from_path(script_path: Path) -> int:
    m = re.search(r"(\d+)", script_path.stem)
    return int(m.group(1)) if m else 0


class GenerationStep:
    """Convenience wrapper around :class:`BuildContext` + :class:`Scene`."""

    __slots__ = (
        "_assets_dir",
        "_ctx",
        "_out_dir",
        "_project_dir",
        "_script_path",
        "_step_index",
    )

    def __init__(
        self,
        script_file: str | Path,
        *,
        assets_dir: str = "assets",
        out_dir: str = "output",
        seed: int = 0,
    ) -> None:
        script = Path(script_file).resolve()
        project_dir = script.parent
        step_idx = _step_index_from_path(script)
        self._script_path = script
        self._ctx = BuildContext(project_dir, step_idx, seed=seed)
        self._assets_dir = Path(assets_dir)
        self._out_dir = Path(out_dir)
        self._project_dir = project_dir
        self._step_index = step_idx

    @property
    def ctx(self) -> BuildContext:
        return self._ctx

    @property
    def script_path(self) -> Path:
        return self._script_path

    def _resolve_dir(self, rel_or_abs: Path) -> Path:
        if rel_or_abs.is_absolute():
            return rel_or_abs
        return (self._project_dir / rel_or_abs).resolve()

    def out_path(self, step_index: int | None = None) -> Path:
        """Resolve the canonical GLB path for a generation step."""
        idx = self._step_index if step_index is None else int(step_index)
        return self._resolve_dir(self._out_dir) / f"generation_{idx:03d}.glb"

    def asset(self, relative_path: str | Path) -> Path:
        """Resolve an asset path under the configured assets directory."""
        return self._resolve_dir(self._assets_dir) / relative_path

    def scene(self, load_prev: bool = True) -> Scene:
        """Open previous step scene (or start fresh) and wrap it."""
        bpy_scene = self._ctx.new_scene()
        if load_prev and self._step_index > 0:
            prev = self.out_path(self._step_index - 1)
            if prev.exists():
                bpy.ops.import_scene.gltf(filepath=str(prev))
        return Scene(bpy_scene)

    def render(
        self, rc, scene: Scene, *, preset: str = "iso", out: str, **opts: Any
    ) -> Path:  # type: ignore[no-untyped-def]
        """Render a debug still into this step's debug directory."""
        out_path = self._ctx.debug_dir() / out
        rc.still(scene, preset=preset, out=out_path, **opts)
        return out_path

    def save(self, scene: Scene, filename: str | None = None) -> Path:
        """Save scene GLB and manifest to the step output directory."""
        if filename is None:
            out = self.out_path()
        else:
            out = self._resolve_dir(self._out_dir) / filename
            out.parent.mkdir(parents=True, exist_ok=True)
        scene.save(out)
        return out
