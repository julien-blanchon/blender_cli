"""Scene-bound asset/material registries with caching and usage tracking."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from blender_cli.assets.material import Material
from blender_cli.assets.prefab import Prefab
from blender_cli.core.metadata import decode_dict, encode_dict

if TYPE_CHECKING:
    from blender_cli.scene.scene import Scene


class _RegistryUsage(TypedDict):
    prefabs: list[dict[str, str]]
    materials: list[str]


def _read_usage(scene: Scene) -> _RegistryUsage:
    raw = scene.bpy_scene.get("_mc_registry_usage")
    parsed = decode_dict(raw)
    parsed.setdefault("prefabs", [])
    parsed.setdefault("materials", [])
    if not isinstance(parsed["prefabs"], list):
        parsed["prefabs"] = []
    if not isinstance(parsed["materials"], list):
        parsed["materials"] = []
    return _RegistryUsage(prefabs=parsed["prefabs"], materials=parsed["materials"])


def _write_usage(scene: Scene, usage: _RegistryUsage) -> None:
    scene.bpy_scene["_mc_registry_usage"] = encode_dict(usage, sort_keys=True)


def _default_asset_roots() -> list[Path]:
    cwd = Path.cwd()
    return [
        cwd / "assets",
        cwd / "examples",
    ]


class AssetRegistry:
    """Prefab resolver/cache by logical id or filename."""

    __slots__ = ("_prefab_cache", "_prefab_index", "_roots", "_scene")

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._roots = _default_asset_roots()
        self._prefab_cache: dict[Path, Prefab] = {}
        self._prefab_index: dict[str, Path] = {}

    def add_root(self, root: str | Path) -> "AssetRegistry":
        p = Path(root).resolve()
        if p not in self._roots:
            self._roots.append(p)
        return self

    def load_index(self, path: str | Path) -> "AssetRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        models = data.get("models", {})
        if isinstance(models, dict):
            for key, value in models.items():
                self._prefab_index[str(key)] = Path(str(value)).resolve()
        return self

    def index_prefab(self, asset_id: str, path: str | Path) -> "AssetRegistry":
        self._prefab_index[asset_id] = Path(path).resolve()
        return self

    def prefab(self, key: str) -> Prefab:
        resolved = self._resolve_prefab_path(key)
        if resolved in self._prefab_cache:
            prefab = self._prefab_cache[resolved]
        else:
            prefab = Prefab(resolved)
            self._prefab_cache[resolved] = prefab

        usage = _read_usage(self._scene)
        entry = {"id": key, "path": str(resolved)}
        if entry not in usage["prefabs"]:
            usage["prefabs"].append(entry)
            _write_usage(self._scene, usage)
        return prefab

    def _resolve_prefab_path(self, key: str) -> Path:
        direct = Path(key).expanduser()
        if direct.is_file():
            return direct.resolve()
        if key in self._prefab_index:
            return self._prefab_index[key]

        candidates: list[Path] = []
        for root in self._roots:
            if not root.exists():
                continue
            for pat in (f"**/{key}", f"**/{key}.glb", f"**/{key}.gltf"):
                candidates.extend(root.glob(pat))

        files = [p.resolve() for p in candidates if p.is_file()]
        if not files:
            msg = f"Prefab {key!r} not found. Checked roots: {[str(r) for r in self._roots]}"
            raise FileNotFoundError(msg)
        files.sort(key=lambda p: len(str(p)))
        return files[0]


class MaterialRegistry:
    """Material resolver/cache by logical id."""

    __slots__ = ("_cache", "_material_index", "_scene")

    _INDEX_KEY = "_mc_material_index"

    def __init__(self, scene: Scene) -> None:
        self._scene = scene
        self._material_index: dict[str, Path] = {}
        self._cache: dict[tuple[str, float], Material] = {}
        self._restore_index()

    def _persist_index(self) -> None:
        """Write the material_id → folder mapping into the Blender scene so it survives GLB round-trips."""
        payload = {k: str(v) for k, v in self._material_index.items()}
        self._scene.bpy_scene[self._INDEX_KEY] = encode_dict(payload, sort_keys=True)

    def _restore_index(self) -> None:
        """Rebuild _material_index from the scene custom property (if present)."""
        raw = self._scene.bpy_scene.get(self._INDEX_KEY)
        if raw is None:
            return
        parsed = decode_dict(raw)
        for k, v in parsed.items():
            p = Path(str(v))
            if p.is_dir():
                self._material_index.setdefault(str(k), p)

    def load_index(self, path: str | Path) -> "MaterialRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        materials = data.get("materials", {})
        if isinstance(materials, dict):
            for key, value in materials.items():
                self._material_index[str(key)] = Path(str(value)).resolve()
        return self

    def index_material(self, material_id: str, path: str | Path) -> "MaterialRegistry":
        self._material_index[material_id] = Path(path).resolve()
        self._persist_index()
        return self

    def pbr(self, material_id: str, tile: float = 1.0) -> Material:
        key = (material_id, float(tile))
        if key in self._cache:
            return self._cache[key]

        folder = self._resolve_material_folder(material_id)
        safe = material_id.replace("/", "_").replace(" ", "_")
        mat_name = f"{safe}__tile_{str(tile).replace('.', '_')}"
        mat = Material.from_pbr_folder(mat_name, folder, tile_scale=tile)
        self._cache[key] = mat

        usage = _read_usage(self._scene)
        if material_id not in usage["materials"]:
            usage["materials"].append(material_id)
            _write_usage(self._scene, usage)
        return mat

    def _resolve_material_folder(self, material_id: str) -> Path:
        if material_id in self._material_index:
            return self._material_index[material_id]

        direct = Path(material_id).expanduser()
        if direct.is_dir():
            return direct.resolve()

        roots = self._scene.assets._roots  # noqa: SLF001
        candidates: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            candidates.extend(p for p in root.glob(f"**/{material_id}") if p.is_dir())
        if not candidates:
            msg = f"Material id {material_id!r} not found in indexed roots"
            raise FileNotFoundError(msg)
        candidates.sort(key=lambda p: len(str(p)))
        return candidates[0].resolve()
