"""BuildContext — step management, paths, and deterministic RNG."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bpy

if TYPE_CHECKING:
    from blender_cli.render import RenderContext


class DeterministicRNG:
    """
    Deterministic RNG seeded by step index + stream name.

    Same (seed, step_index, stream) always produces the same sequence.
    Different streams or steps produce different sequences.
    """

    __slots__ = ("_base_seed", "_rng", "_step_index", "_stream")

    def __init__(self, step_index: int, stream: str, seed: int = 0) -> None:
        self._base_seed = seed
        self._step_index = step_index
        self._stream = stream
        self._rng = random.Random(f"{seed}:{step_index}:{stream}")

    def int(self, a: int = 0, b: int = 2**31 - 1) -> int:
        """Random integer in [a, b]."""
        return self._rng.randint(a, b)

    def float(self) -> float:
        """Random float in [0.0, 1.0)."""
        return self._rng.random()

    def range(self, min_val: float, max_val: float) -> float:
        """Random float in [min_val, max_val]."""
        return self._rng.uniform(min_val, max_val)

    def metadata(self) -> dict[str, Any]:
        """Determinism metadata for manifest recording."""
        return {
            "stream": self._stream,
            "base_seed": self._base_seed,
            "step_index": self._step_index,
            "derived_seed": f"{self._base_seed}:{self._step_index}:{self._stream}",
        }


class BuildContext:
    """
    Entry point for each build step.

    Manages step sequencing, output paths, debug directories,
    and deterministic RNG streams.
    """

    __slots__ = ("_project_dir", "_render", "_rng_streams", "_seed", "_step_index")

    def __init__(self, project_dir: str | Path, step_index: int, seed: int = 0) -> None:
        self._project_dir = Path(project_dir)
        self._step_index = step_index
        self._seed = seed
        self._render: RenderContext | None = None
        self._rng_streams: dict[str, dict[str, Any]] = {}

    @property
    def step_index(self) -> int:
        """Zero-based index of this build step."""
        return self._step_index

    @property
    def step_id(self) -> str:
        """Human-readable step identifier (e.g. ``generation_003``)."""
        return f"generation_{self._step_index:03d}"

    def out_path(self) -> Path:
        """GLB output path for this step (e.g. ``output/generation_003.glb``)."""
        return self._project_dir / "output" / f"{self.step_id}.glb"

    def debug_dir(self) -> Path:
        """Debug output directory for this step (auto-created)."""
        d = self._project_dir / "debug" / self.step_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def open_prev(self) -> bpy.types.Scene:
        """Load the previous step's GLB, or return an empty scene if step 0."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if self._step_index > 0:
            prev = (
                self._project_dir
                / "output"
                / f"generation_{self._step_index - 1:03d}.glb"
            )
            bpy.ops.import_scene.gltf(filepath=str(prev))
        assert bpy.context.scene is not None, "Scene is not set"
        self._sync_rng_streams()
        return bpy.context.scene

    def new_scene(self) -> bpy.types.Scene:
        """Return a fresh empty scene."""
        bpy.ops.wm.read_factory_settings(use_empty=True)
        assert bpy.context.scene is not None, "Scene is not set"
        self._sync_rng_streams()
        return bpy.context.scene

    @property
    def render(self) -> RenderContext:
        """Lazy-initialized :class:`RenderContext` for rendering operations."""
        if self._render is None:
            from blender_cli.render import RenderContext

            self._render = RenderContext()
        return self._render

    def asset(self, relative_path: str | Path) -> Path:
        """
        Resolve an asset path relative to ``<project_dir>/assets/``.

        Eliminates the common ``PROJECT = Path(__file__).resolve().parent…``
        boilerplate in example scripts.
        """
        return self._project_dir / "assets" / relative_path

    def prev_glb_path(self, step_index: int | None = None) -> Path:
        """Path to the GLB for the given step (default: previous step)."""
        idx = step_index if step_index is not None else self._step_index - 1
        return self._project_dir / "output" / f"generation_{idx:03d}.glb"

    def load_manifest(self) -> dict[str, Any]:
        """
        Load the manifest from the previous step's GLB (embedded extras).

        Returns the parsed JSON dict so the agent can inspect anchor
        positions, object counts, spatial extent, and generation history
        without building the full scene.

        Raises ``FileNotFoundError`` if no previous GLB exists
        (e.g. when ``step_index`` is 0).
        """
        import bpy

        glb = self.prev_glb_path()
        if not glb.exists():
            msg = f"Previous GLB not found: {glb}"
            raise FileNotFoundError(msg)
        bpy.ops.wm.read_factory_settings(use_empty=True)
        bpy.ops.import_scene.gltf(filepath=str(glb))
        manifest_str = bpy.context.scene.get("_scene_manifest")
        if not manifest_str:
            msg = f"No manifest in GLB: {glb}"
            raise FileNotFoundError(msg)
        return dict(json.loads(manifest_str))

    def rng(self, stream: str) -> DeterministicRNG:
        """
        Deterministic RNG for the given stream name.

        Same (seed, step_index, stream) always produces the same sequence.
        """
        rng = DeterministicRNG(self._step_index, stream, self._seed)
        self._rng_streams[stream] = rng.metadata()
        self._sync_rng_streams()
        return rng

    def _sync_rng_streams(self) -> None:
        """Mirror RNG stream metadata into scene props for manifest export."""
        if bpy.context.scene is None:
            return
        payload = sorted(
            self._rng_streams.values(),
            key=lambda item: str(item.get("stream", "")),
        )
        bpy.context.scene["_mc_rng_streams"] = json.dumps(payload, sort_keys=True)
