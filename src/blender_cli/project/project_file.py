"""ProjectFile — persistent JSON project for incremental scene building."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Scene profiles
# ---------------------------------------------------------------------------

PROFILES: dict[str, dict[str, Any]] = {
    "default": {
        "render": {
            "engine": "EEVEE",
            "resolution": [1920, 1080],
            "samples": 64,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "preview": {
        "render": {
            "engine": "EEVEE",
            "resolution": [960, 540],
            "samples": 16,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "hd720p": {
        "render": {
            "engine": "EEVEE",
            "resolution": [1280, 720],
            "samples": 64,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "hd1080p": {
        "render": {
            "engine": "CYCLES",
            "resolution": [1920, 1080],
            "samples": 128,
            "denoising": True,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "4k": {
        "render": {
            "engine": "CYCLES",
            "resolution": [3840, 2160],
            "samples": 256,
            "denoising": True,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "instagram_square": {
        "render": {
            "engine": "EEVEE",
            "resolution": [1080, 1080],
            "samples": 64,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 30},
    },
    "youtube_short": {
        "render": {
            "engine": "EEVEE",
            "resolution": [1080, 1920],
            "samples": 64,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 30},
    },
    "product_render": {
        "render": {
            "engine": "CYCLES",
            "resolution": [2048, 2048],
            "samples": 512,
            "denoising": True,
            "film_transparent": True,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
    "animation_preview": {
        "render": {
            "engine": "EEVEE",
            "resolution": [1280, 720],
            "samples": 16,
            "denoising": False,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 30},
    },
    "print_a4_300dpi": {
        "render": {
            "engine": "CYCLES",
            "resolution": [3508, 2480],
            "samples": 256,
            "denoising": True,
            "film_transparent": False,
            "output_format": "PNG",
        },
        "scene": {"fps": 24},
    },
}

# ---------------------------------------------------------------------------
# Schema keys for validation
# ---------------------------------------------------------------------------

_REQUIRED_TOP_KEYS = {
    "version",
    "name",
    "scene",
    "render",
    "world",
    "terrain",
    "anchors",
    "objects",
    "instances",
    "materials",
    "cameras",
    "lights",
    "metadata",
}

_SCENE_KEYS = {"unit_system", "unit_scale", "frame_start", "frame_end", "fps"}
_RENDER_KEYS = {
    "engine",
    "resolution",
    "samples",
    "denoising",
    "film_transparent",
    "output_format",
    "output_path",
    "resolution_pct",
}
_WORLD_KEYS = {
    "background_color",
    "use_hdri",
    "hdri_path",
    "hdri_strength",
    "hdri_rotation",
}
_METADATA_KEYS = {"created", "modified", "software"}

_TERRAIN_OPS = {
    "flat", "noise", "smooth", "terrace", "clamp", "stamp",
    "erode", "radial_falloff", "remap_curve",
}
_PRIMITIVE_TYPES = {"box", "plane", "cylinder", "sphere", "cone", "torus"}
_CAMERA_PATH_TYPES = {"orbit", "dolly", "flyover"}


def _validate_terrain(terrain: Any) -> list[str]:
    """Validate terrain field (None or dict with operations)."""
    if terrain is None:
        return []
    errors: list[str] = []
    if not isinstance(terrain, dict):
        errors.append("'terrain' must be null or a dict")
        return errors
    for key in ("width", "height"):
        if key not in terrain:
            errors.append(f"terrain.{key} is required")
        elif not isinstance(terrain[key], int) or terrain[key] < 1:
            errors.append(f"terrain.{key} must be int >= 1")
    if "meters_per_px" in terrain:
        mpp = terrain["meters_per_px"]
        if not isinstance(mpp, (int, float)) or mpp <= 0:
            errors.append("terrain.meters_per_px must be > 0")
    ops = terrain.get("operations", [])
    if not isinstance(ops, list):
        errors.append("terrain.operations must be a list")
    else:
        for i, op in enumerate(ops):
            if not isinstance(op, dict):
                errors.append(f"terrain.operations[{i}] must be a dict")
            elif "op" not in op:
                errors.append(f"terrain.operations[{i}] missing 'op' key")
            elif op["op"] not in _TERRAIN_OPS:
                errors.append(
                    f"terrain.operations[{i}].op={op['op']!r} not in {sorted(_TERRAIN_OPS)}"
                )
    mesh = terrain.get("mesh")
    if mesh is not None and not isinstance(mesh, dict):
        errors.append("terrain.mesh must be null or a dict")
    return errors


def _validate_anchors(anchors: Any) -> list[str]:
    """Validate anchors list."""
    if not isinstance(anchors, list):
        return ["'anchors' must be a list"]
    errors: list[str] = []
    for i, a in enumerate(anchors):
        if not isinstance(a, dict):
            errors.append(f"anchors[{i}] must be a dict")
            continue
        if "name" not in a or not isinstance(a["name"], str):
            errors.append(f"anchors[{i}].name must be a string")
        pos = a.get("position")
        if pos is None or not (isinstance(pos, list) and len(pos) == 3):
            errors.append(f"anchors[{i}].position must be [x, y, z]")
    return errors


def _validate_instances(instances: Any) -> list[str]:
    """Validate instances list."""
    if not isinstance(instances, list):
        return ["'instances' must be a list"]
    errors: list[str] = []
    for i, inst in enumerate(instances):
        if not isinstance(inst, dict):
            errors.append(f"instances[{i}] must be a dict")
            continue
        if "prefab" not in inst or not isinstance(inst["prefab"], str):
            errors.append(f"instances[{i}].prefab must be a string path")
        pts = inst.get("points", [])
        if not isinstance(pts, list):
            errors.append(f"instances[{i}].points must be a list")
    return errors


def _validate_materials(materials: Any) -> list[str]:
    """Validate materials list."""
    if not isinstance(materials, list):
        return ["'materials' must be a list"]
    errors: list[str] = []
    names_seen: set[str] = set()
    for i, mat in enumerate(materials):
        if not isinstance(mat, dict):
            errors.append(f"materials[{i}] must be a dict")
            continue
        name = mat.get("name")
        if not name or not isinstance(name, str):
            errors.append(f"materials[{i}].name must be a non-empty string")
        elif name in names_seen:
            errors.append(f"materials[{i}].name={name!r} is duplicated")
        else:
            names_seen.add(name)
        # Must have either pbr_folder or color (or both are absent for a basic mat)
        color = mat.get("color")
        if color is not None and not (isinstance(color, list) and len(color) in (3, 4)):
            errors.append(f"materials[{i}].color must be [R,G,B] or [R,G,B,A]")
        tiling = mat.get("tiling")
        if tiling is not None and not (isinstance(tiling, list) and len(tiling) == 2):
            errors.append(f"materials[{i}].tiling must be [u, v]")
    return errors


def validate_project(data: dict[str, Any]) -> list[str]:
    """Validate project dict against schema. Returns list of error messages (empty = valid)."""
    errors: list[str] = []
    missing = _REQUIRED_TOP_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing top-level keys: {sorted(missing)}")
        return errors  # can't validate further

    if not isinstance(data["version"], (int, str)):
        errors.append("'version' must be int or str")
    if not isinstance(data["name"], str) or not data["name"]:
        errors.append("'name' must be a non-empty string")

    # scene
    scene = data["scene"]
    if not isinstance(scene, dict):
        errors.append("'scene' must be a dict")
    else:
        s_missing = _SCENE_KEYS - set(scene.keys())
        if s_missing:
            errors.append(f"Missing scene keys: {sorted(s_missing)}")
        if "fps" in scene and (not isinstance(scene["fps"], int) or scene["fps"] < 1):
            errors.append("scene.fps must be int >= 1")
        if "frame_start" in scene and "frame_end" in scene:
            if scene["frame_end"] < scene["frame_start"]:
                errors.append("scene.frame_end must be >= scene.frame_start")

    # render
    rnd = data["render"]
    if not isinstance(rnd, dict):
        errors.append("'render' must be a dict")
    else:
        r_missing = (_RENDER_KEYS - {"output_path", "resolution_pct"}) - set(rnd.keys())
        if r_missing:
            errors.append(f"Missing render keys: {sorted(r_missing)}")
        if "engine" in rnd and rnd["engine"] not in {"CYCLES", "EEVEE", "WORKBENCH"}:
            errors.append(
                f"render.engine must be CYCLES|EEVEE|WORKBENCH, got {rnd['engine']}"
            )
        if "resolution" in rnd:
            res = rnd["resolution"]
            if not (
                isinstance(res, list)
                and len(res) == 2
                and all(isinstance(v, int) and v > 0 for v in res)
            ):
                errors.append(
                    "render.resolution must be [width, height] with positive ints"
                )
        if "samples" in rnd and (
            not isinstance(rnd["samples"], int) or rnd["samples"] < 1
        ):
            errors.append("render.samples must be int >= 1")
        if "output_format" in rnd and rnd["output_format"] not in {
            "PNG",
            "JPEG",
            "TIFF",
            "OPEN_EXR",
            "HDR",
        }:
            errors.append(
                f"render.output_format must be PNG|JPEG|TIFF|OPEN_EXR|HDR, got {rnd['output_format']}"
            )
        if "resolution_pct" in rnd:
            pct = rnd["resolution_pct"]
            if not isinstance(pct, int) or not (1 <= pct <= 100):
                errors.append("render.resolution_pct must be int 1..100")

    # world
    world = data["world"]
    if not isinstance(world, dict):
        errors.append("'world' must be a dict")
    else:
        from blender_cli.render.world import validate_world

        errors.extend(validate_world(world))

    # terrain (nullable)
    errors.extend(_validate_terrain(data["terrain"]))

    # anchors
    errors.extend(_validate_anchors(data["anchors"]))

    # instances
    errors.extend(_validate_instances(data["instances"]))

    # materials
    errors.extend(_validate_materials(data["materials"]))

    # arrays
    errors.extend(
        f"'{key}' must be a list"
        for key in ("objects", "cameras", "lights")
        if not isinstance(data[key], list)
    )

    # metadata
    meta = data["metadata"]
    if not isinstance(meta, dict):
        errors.append("'metadata' must be a dict")
    else:
        m_missing = _METADATA_KEYS - set(meta.keys())
        if m_missing:
            errors.append(f"Missing metadata keys: {sorted(m_missing)}")

    return errors


# ---------------------------------------------------------------------------
# Shared project-data helpers
# ---------------------------------------------------------------------------


def resolve_object(
    project_data: dict[str, Any], ref: str | int
) -> tuple[int, dict[str, Any]]:
    """Resolve object by index (int) or name (str). Returns (index, object_dict)."""
    objects = project_data["objects"]
    if not objects:
        msg = "Project has no objects"
        raise ValueError(msg)

    if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
        idx = int(ref)
        if idx < 0 or idx >= len(objects):
            msg = f"Object index {idx} out of range [0, {len(objects) - 1}]"
            raise IndexError(msg)
        return idx, objects[idx]

    for i, obj in enumerate(objects):
        if obj.get("name") == ref:
            return i, obj
    msg = f"No object named {ref!r}"
    raise KeyError(msg)


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------


def _atomic_write(path: Path, data: bytes) -> None:
    """Write *data* to *path* atomically via temp file + rename."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=parent, suffix=".tmp")
    try:
        os.write(fd, data)
        os.fsync(fd)
        os.close(fd)
        Path(tmp).replace(path)
    except BaseException:
        os.close(fd) if not _is_closed(fd) else None
        with _suppress():
            Path(tmp).unlink()
        raise


def _is_closed(fd: int) -> bool:
    try:
        os.fstat(fd)
        return False
    except OSError:
        return True


class _suppress:
    def __enter__(self) -> None:
        pass

    def __exit__(self, *_: object) -> bool:
        return True


# ---------------------------------------------------------------------------
# ProjectFile
# ---------------------------------------------------------------------------


class ProjectFile:
    """Persistent JSON project file for incremental scene building."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self.data = data
        self.path = path

    # -- Factories -----------------------------------------------------------

    @classmethod
    def new(cls, name: str, profile: str = "default") -> ProjectFile:
        """Create a new project with the given name and profile."""
        if profile not in PROFILES:
            msg = f"Unknown profile {profile!r}. Available: {sorted(PROFILES)}"
            raise ValueError(msg)

        p = PROFILES[profile]
        now = datetime.now(UTC).isoformat()

        data: dict[str, Any] = {
            "version": 2,
            "name": name,
            "scene": {
                "unit_system": "METRIC",
                "unit_scale": 1.0,
                "frame_start": 1,
                "frame_end": 250,
                "fps": p["scene"]["fps"],
            },
            "render": {
                "engine": p["render"]["engine"],
                "resolution": list(p["render"]["resolution"]),
                "samples": p["render"]["samples"],
                "denoising": p["render"]["denoising"],
                "film_transparent": p["render"]["film_transparent"],
                "output_format": p["render"]["output_format"],
                "output_path": "//render/",
                "resolution_pct": 100,
            },
            "world": {
                "background_color": [0.05, 0.05, 0.05],
                "use_hdri": False,
                "hdri_path": None,
                "hdri_strength": 1.0,
                "hdri_rotation": 0.0,
            },
            "terrain": None,
            "anchors": [],
            "objects": [],
            "instances": [],
            "materials": [],
            "cameras": [],
            "lights": [],
            "metadata": {
                "created": now,
                "modified": now,
                "software": "maps-creation",
            },
        }
        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> ProjectFile:
        """Load a project from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        errors = validate_project(data)
        if errors:
            msg = f"Invalid project file: {'; '.join(errors)}"
            raise ValueError(msg)
        return cls(data, path=path)

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """Persist project with atomic write. Returns the saved path."""
        target = Path(path) if path else self.path
        if target is None:
            msg = "No path specified and project has no associated path"
            raise ValueError(msg)
        target = Path(target)
        self.data["metadata"]["modified"] = datetime.now(UTC).isoformat()
        blob = json.dumps(self.data, indent=2, ensure_ascii=False).encode("utf-8")
        _atomic_write(target, blob)
        self.path = target
        return target

    # -- Blenvy registry -----------------------------------------------------

    def _load_registry(self) -> Any:
        """Try to load the Bevy registry from project config or fallback path.

        Returns a :class:`BevyRegistry` or ``None``. Never raises.
        """
        from blender_cli.blenvy_registry import BevyRegistry

        rpath = self.data.get("blenvy", {}).get("registry_path")
        if rpath:
            p = Path(rpath)
            if not p.is_absolute() and self.path:
                p = self.path.parent / p
            if p.exists():
                return BevyRegistry.load(p)
        # Fallback: assets/registry.json next to project file
        if self.path:
            fallback = self.path.parent / "assets" / "registry.json"
            if fallback.exists():
                return BevyRegistry.load(fallback)
        return None

    def set_registry_path(self, registry_path: str) -> None:
        """Store the Blenvy registry path in the project."""
        if "blenvy" not in self.data:
            self.data["blenvy"] = {}
        self.data["blenvy"]["registry_path"] = registry_path

    # -- Info ----------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.data["name"]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for display."""
        r = self.data["render"]
        s = self.data["scene"]
        terrain = self.data.get("terrain")
        return {
            "name": self.data["name"],
            "version": self.data["version"],
            "frame_range": [s["frame_start"], s["frame_end"]],
            "fps": s["fps"],
            "engine": r["engine"],
            "resolution": r["resolution"],
            "has_terrain": terrain is not None,
            "terrain_ops": len(terrain["operations"]) if terrain else 0,
            "anchors": len(self.data.get("anchors", [])),
            "objects": len(self.data["objects"]),
            "instances": len(self.data.get("instances", [])),
            "materials": len(self.data["materials"]),
            "cameras": len(self.data["cameras"]),
            "lights": len(self.data["lights"]),
        }

    # -- Spatial awareness (pure Python, no bpy) ----------------------------

    def bbox(self) -> dict[str, list[float]] | None:
        """Compute approximate scene bounding box from stored object locations.

        Returns ``{"min": [x, y, z], "max": [x, y, z]}`` or None if no
        objects exist.  Uses object locations + primitive half-extents.
        """
        mins = [float("inf")] * 3
        maxs = [float("-inf")] * 3
        has_any = False

        for obj in self.data["objects"]:
            loc = obj.get("location", [0, 0, 0])
            hx, hy, hz = self._half_extents(obj.get("primitive"), obj.get("scale"))
            for i, h in enumerate((hx, hy, hz)):
                mins[i] = min(mins[i], loc[i] - h)
                maxs[i] = max(maxs[i], loc[i] + h)
            has_any = True

        # Include anchor positions
        for anchor in self.data.get("anchors", []):
            pos = anchor.get("position", [0, 0, 0])
            for i in range(3):
                mins[i] = min(mins[i], pos[i])
                maxs[i] = max(maxs[i], pos[i])
            has_any = True

        if not has_any:
            return None
        return {
            "min": [round(v, 4) for v in mins],
            "max": [round(v, 4) for v in maxs],
        }

    def nearby_objects(
        self,
        position: list[float],
        radius: float = 20.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find objects near a position (pure Python, no bpy).

        Returns list of ``{"name", "uid", "distance", "tags"}`` sorted by
        distance, up to *limit* entries within *radius* metres.
        """
        import math

        px, py, pz = position
        results: list[tuple[float, dict]] = []
        for obj in self.data["objects"]:
            loc = obj.get("location", [0, 0, 0])
            dx, dy, dz = loc[0] - px, loc[1] - py, loc[2] - pz
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist <= radius:
                results.append((dist, {
                    "name": obj.get("name", ""),
                    "uid": obj.get("uid", ""),
                    "distance": round(dist, 2),
                    "tags": obj.get("tags", []),
                }))
        results.sort(key=lambda t: t[0])
        return [entry for _, entry in results[:limit]]

    @staticmethod
    def _half_extents(
        primitive: dict[str, Any] | None,
        scale: list[float] | None = None,
    ) -> tuple[float, float, float]:
        """Estimate half-extents (hx, hy, hz) from a primitive definition."""
        hx = hy = hz = 0.5  # default for empties / assets
        if primitive:
            pt = primitive.get("type")
            if pt == "box":
                s = primitive.get("size", [1, 1, 1])
                hx, hy, hz = s[0] / 2, s[1] / 2, s[2] / 2
            elif pt == "plane":
                s = primitive.get("size", [1, 1])
                hx, hy, hz = s[0] / 2, s[1] / 2, 0.01
            elif pt in ("cylinder", "cone"):
                r = primitive.get("radius", primitive.get("radius1", 0.5))
                h = primitive.get("height", primitive.get("depth", 1.0))
                hx = hy = r
                hz = h / 2
            elif pt == "sphere":
                r = primitive.get("radius", 0.5)
                hx = hy = hz = r
            elif pt == "torus":
                mr = primitive.get("major_radius", 1.0) + primitive.get("minor_radius", 0.25)
                hx = hy = mr
                hz = primitive.get("minor_radius", 0.25)
        if scale:
            hx *= scale[0]
            hy *= scale[1]
            hz *= scale[2]
        return hx, hy, hz

    def placement_warnings(
        self,
        name: str,
        location: list[float],
        primitive: dict[str, Any] | None = None,
        scale: list[float] | None = None,
    ) -> list[str]:
        """Check for potential placement issues (pure Python, no bpy).

        Performs full 3D AABB overlap detection, proximity checks, and
        terrain-bounds validation. Returns list of warning strings.
        """
        import math

        warnings: list[str] = []
        px, py, pz = location
        hx, hy, hz = self._half_extents(primitive, scale)

        for obj in self.data["objects"]:
            if obj.get("name") == name:
                continue
            oloc = obj.get("location", [0, 0, 0])
            dx = abs(px - oloc[0])
            dy = abs(py - oloc[1])
            dz = abs(pz - oloc[2])

            ohx, ohy, ohz = self._half_extents(
                obj.get("primitive"), obj.get("scale"),
            )

            # Full 3D AABB overlap
            overlap_x = (hx + ohx) - dx
            overlap_y = (hy + ohy) - dy
            overlap_z = (hz + ohz) - dz
            if overlap_x > 0.001 and overlap_y > 0.001 and overlap_z > 0.001:
                overlap = min(overlap_x, overlap_y, overlap_z)
                oname = obj.get("name", "?")
                warnings.append(
                    f"Object '{name}' overlaps '{oname}' "
                    f"(~{overlap:.2f}m penetration). "
                    f"Consider moving apart."
                )
            else:
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < 0.5:
                    warnings.append(
                        f"Object '{name}' very close to '{obj.get('name', '?')}' "
                        f"({dist:.2f}m)."
                    )

        # Check physics collider overlap with ground surfaces.
        # Objects with RigidBody sitting exactly at ground level will overlap
        # the ground's collider at spawn — avian3d warns about "explosive behavior".
        if primitive is not None:
            has_rb = False
            for obj in self.data["objects"]:
                if obj.get("name") == name:
                    continue
                oprim = obj.get("primitive")
                if oprim and oprim.get("type") == "plane" and obj.get("bevy_components"):
                    # This is a ground plane with physics — check if new object
                    # sits at or below ground level (z <= half_extent of new obj)
                    ground_z = obj.get("location", [0, 0, 0])[2]
                    if (pz - hz - ground_z) < 0.02:
                        warnings.append(
                            f"Object '{name}' at z={pz:.2f} sits exactly on "
                            f"ground plane '{obj.get('name', '?')}' (z={ground_z:.1f}). "
                            f"Physics colliders will overlap at spawn. "
                            f"Lift by ~0.05m to avoid explosive behavior."
                        )

        # Check terrain bounds
        terrain = self.data.get("terrain")
        if terrain:
            tw = terrain["width"] * terrain.get("meters_per_px", 1.0)
            th = terrain["height"] * terrain.get("meters_per_px", 1.0)
            if px - hx < 0 or px + hx > tw or py - hy < 0 or py + hy > th:
                warnings.append(
                    f"Object '{name}' at ({px:.1f}, {py:.1f}) extends outside "
                    f"terrain (0..{tw:.0f}, 0..{th:.0f})."
                )

        return warnings

    def describe(self) -> dict[str, Any]:
        """Return a structured scene description for agentic inspection.

        Includes: object list with locations, anchors, spatial extent,
        material list, terrain summary, camera list, light list.
        """
        result: dict[str, Any] = {
            "name": self.data["name"],
            "version": self.data["version"],
        }

        # Terrain
        terrain = self.data.get("terrain")
        if terrain:
            mpp = terrain.get("meters_per_px", 1.0)
            result["terrain"] = {
                "size_px": [terrain["width"], terrain["height"]],
                "size_m": [terrain["width"] * mpp, terrain["height"] * mpp],
                "operations": len(terrain.get("operations", [])),
                "material": terrain.get("material"),
            }
        else:
            result["terrain"] = None

        # Objects
        result["objects"] = [
            {
                "name": o.get("name"),
                "uid": o.get("uid"),
                "type": o.get("type"),
                "location": o.get("location"),
                "tags": o.get("tags", []),
                "material": o.get("material"),
                "primitive": o.get("primitive", {}).get("type") if o.get("primitive") else None,
                "asset_path": o.get("asset_path"),
                "visible": o.get("visible", True),
            }
            for o in self.data["objects"]
        ]

        # Anchors
        result["anchors"] = self.data.get("anchors", [])

        # Instances
        result["instances"] = [
            {
                "name": i.get("name"),
                "prefab": i.get("prefab"),
                "count": len(i.get("points", [])),
                "tags": i.get("tags", []),
            }
            for i in self.data.get("instances", [])
        ]

        # Materials, cameras, lights (summaries)
        result["materials"] = [m.get("name") for m in self.data.get("materials", [])]
        result["cameras"] = [
            {"name": c.get("name"), "type": c.get("type"), "has_path": "path" in c}
            for c in self.data.get("cameras", [])
        ]
        result["lights"] = [
            {"name": l.get("name"), "type": l.get("type"), "energy": l.get("energy")}
            for l in self.data.get("lights", [])
        ]

        # Spatial extent
        result["bbox"] = self.bbox()

        # Tag histogram
        tag_counts: dict[str, int] = {}
        for o in self.data["objects"]:
            for t in o.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1
        result["by_tag"] = tag_counts

        return result

    # -- World settings convenience methods ---------------------------------

    def set_world_hdri(
        self, hdri_path: str, *, strength: float = 1.0, rotation: float = 0.0
    ) -> None:
        """Set world background to an HDRI environment map."""
        w = self.data["world"]
        w["use_hdri"] = True
        w["hdri_path"] = hdri_path
        w["hdri_strength"] = strength
        w["hdri_rotation"] = rotation

    def set_world_background(self, color: list[float]) -> None:
        """Set world background to a solid color [R, G, B]."""
        w = self.data["world"]
        w["use_hdri"] = False
        w["hdri_path"] = None
        w["background_color"] = list(color)

    def clear_hdri(self) -> None:
        """Disable HDRI and revert to solid background."""
        w = self.data["world"]
        w["use_hdri"] = False
        w["hdri_path"] = None

    # -- Render settings convenience methods --------------------------------

    def set_render(self, **kwargs: Any) -> None:
        """Update render settings. Accepts any render key (engine, resolution, samples, etc.)."""
        rnd = self.data["render"]
        for key, value in kwargs.items():
            if key not in {"engine", "resolution", "samples", "denoising",
                           "film_transparent", "output_format", "output_path",
                           "resolution_pct"}:
                msg = f"Unknown render key {key!r}"
                raise ValueError(msg)
            rnd[key] = value

    # -- Anchor-relative position helpers -----------------------------------

    def anchor_pos(
        self,
        name: str,
        offset: list[float] | None = None,
    ) -> list[float]:
        """Resolve an anchor position, optionally with an offset.

        Returns ``[x, y, z]``.  Raises ``KeyError`` if anchor not found.
        """
        anchor = self.find_anchor(name)
        if anchor is None:
            msg = f"No anchor named {name!r}"
            raise KeyError(msg)
        pos = list(anchor["position"])
        if offset:
            pos[0] += offset[0]
            pos[1] += offset[1]
            pos[2] += offset[2]
        return pos

    # -- Object management ---------------------------------------------------

    def _find_object(self, index_or_name: str | int) -> tuple[int, dict[str, Any]]:
        """Find an object by integer index or name. Returns (index, obj_dict)."""
        objects = self.data["objects"]
        if isinstance(index_or_name, int) or (
            isinstance(index_or_name, str) and index_or_name.isdigit()
        ):
            idx = int(index_or_name)
            if idx < 0 or idx >= len(objects):
                msg = f"Object index {idx} out of range (0..{len(objects) - 1})"
                raise IndexError(msg)
            return idx, objects[idx]
        # Search by name
        for i, obj in enumerate(objects):
            if obj["name"] == index_or_name:
                return i, obj
        msg = f"No object named {index_or_name!r}"
        raise KeyError(msg)

    def duplicate_object(self, index_or_name: str | int) -> dict[str, Any]:
        """Deep-copy an object with a new UID and auto-incremented name."""
        _, obj = self._find_object(index_or_name)
        new_obj = copy.deepcopy(obj)
        new_obj["uid"] = str(uuid.uuid4())

        # Auto-increment name: "Cube" -> "Cube.001", "Cube.001" -> "Cube.002"
        base = obj["name"]
        existing_names = {o["name"] for o in self.data["objects"]}
        counter = 1
        while True:
            candidate = f"{base}.{counter:03d}"
            if candidate not in existing_names:
                new_obj["name"] = candidate
                break
            counter += 1

        self.data["objects"].append(new_obj)
        return new_obj

    def set_parent(self, child: str | int, parent: str | int) -> None:
        """Set parent-child relationship. Stores parent UID on child."""
        _, child_obj = self._find_object(child)
        _, parent_obj = self._find_object(parent)
        child_obj["parent_id"] = parent_obj["uid"]

    def set_visible(self, index_or_name: str | int, visible: bool) -> None:
        """Toggle persistent visibility on an object."""
        _, obj = self._find_object(index_or_name)
        obj["visible"] = visible

    def remove_object(self, index_or_name: str | int, cascade: bool = False) -> None:
        """Remove an object. With cascade=True, also removes children."""
        idx, obj = self._find_object(index_or_name)
        uid = obj["uid"]

        if cascade:
            # Remove children recursively
            children = [
                i
                for i, o in enumerate(self.data["objects"])
                if o.get("parent_id") == uid
            ]
            # Remove in reverse order to preserve indices
            for child_idx in sorted(children, reverse=True):
                self.remove_object(child_idx, cascade=True)
            # Re-find after removals
            idx, _ = self._find_object(index_or_name)
        else:
            # Unparent children
            for o in self.data["objects"]:
                if o.get("parent_id") == uid:
                    o.pop("parent_id", None)

        self.data["objects"].pop(idx)

    # -- Terrain management --------------------------------------------------

    def set_terrain(
        self,
        width: int,
        height: int,
        meters_per_px: float = 1.0,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Initialise (or reset) the terrain recipe."""
        terrain: dict[str, Any] = {
            "width": width,
            "height": height,
            "meters_per_px": meters_per_px,
            "operations": [],
            "mesh": {"lod": 2, "skirts": 5.0, "tile_scale": 5.0},
            "material": None,
        }
        if seed is not None:
            terrain["seed"] = seed
        self.data["terrain"] = terrain
        return terrain

    def terrain_op(self, op: str, **params: Any) -> dict[str, Any]:
        """Append a terrain operation. Raises if terrain not initialised."""
        terrain = self.data.get("terrain")
        if terrain is None:
            msg = "Terrain not initialised — call set_terrain() first"
            raise ValueError(msg)
        if op not in _TERRAIN_OPS:
            msg = f"Unknown terrain op {op!r}. Valid: {sorted(_TERRAIN_OPS)}"
            raise ValueError(msg)
        entry: dict[str, Any] = {"op": op, **params}
        terrain["operations"].append(entry)
        return entry

    def set_terrain_mesh(self, **params: Any) -> None:
        """Update terrain mesh generation params (lod, skirts, tile_scale)."""
        terrain = self.data.get("terrain")
        if terrain is None:
            msg = "Terrain not initialised — call set_terrain() first"
            raise ValueError(msg)
        terrain.setdefault("mesh", {}).update(params)

    def set_terrain_material(self, material_name: str) -> None:
        """Assign a named material to the terrain mesh."""
        terrain = self.data.get("terrain")
        if terrain is None:
            msg = "Terrain not initialised — call set_terrain() first"
            raise ValueError(msg)
        terrain["material"] = material_name

    def clear_terrain(self) -> None:
        """Remove the terrain recipe."""
        self.data["terrain"] = None

    # -- Anchor management ---------------------------------------------------

    def add_anchor(
        self,
        name: str,
        position: list[float],
        annotation: str | None = None,
    ) -> dict[str, Any]:
        """Add a named anchor point."""
        for a in self.data["anchors"]:
            if a["name"] == name:
                msg = f"Anchor {name!r} already exists"
                raise ValueError(msg)
        anchor: dict[str, Any] = {"name": name, "position": list(position)}
        if annotation:
            anchor["annotation"] = annotation
        self.data["anchors"].append(anchor)
        return anchor

    def remove_anchor(self, name: str) -> dict[str, Any]:
        """Remove an anchor by name. Returns the removed anchor."""
        for i, a in enumerate(self.data["anchors"]):
            if a["name"] == name:
                return self.data["anchors"].pop(i)
        msg = f"No anchor named {name!r}"
        raise KeyError(msg)

    def find_anchor(self, name: str) -> dict[str, Any] | None:
        """Find an anchor by name. Returns None if not found."""
        for a in self.data["anchors"]:
            if a["name"] == name:
                return a
        return None

    # -- Material management -------------------------------------------------

    def add_material(
        self,
        name: str,
        *,
        pbr_folder: str | None = None,
        tiling: list[float] | None = None,
        color: list[float] | None = None,
        roughness: float | None = None,
        metallic: float | None = None,
    ) -> dict[str, Any]:
        """Add a named material definition."""
        for m in self.data["materials"]:
            if m["name"] == name:
                msg = f"Material {name!r} already exists"
                raise ValueError(msg)
        mat: dict[str, Any] = {"name": name}
        if pbr_folder is not None:
            mat["pbr_folder"] = pbr_folder
        if tiling is not None:
            mat["tiling"] = list(tiling)
        if color is not None:
            mat["color"] = list(color)
        if roughness is not None:
            mat["roughness"] = roughness
        if metallic is not None:
            mat["metallic"] = metallic
        self.data["materials"].append(mat)
        return mat

    def remove_material(self, name: str) -> dict[str, Any]:
        """Remove a material by name. Returns the removed material."""
        for i, m in enumerate(self.data["materials"]):
            if m["name"] == name:
                return self.data["materials"].pop(i)
        msg = f"No material named {name!r}"
        raise KeyError(msg)

    def find_material(self, name: str) -> dict[str, Any] | None:
        """Find a material by name. Returns None if not found."""
        for m in self.data["materials"]:
            if m["name"] == name:
                return m
        return None

    # -- Object management (extended) ----------------------------------------

    def add_object(
        self,
        name: str,
        *,
        primitive: dict[str, Any] | None = None,
        asset_path: str | None = None,
        location: list[float] | None = None,
        rotation: list[float] | None = None,
        scale: list[float] | None = None,
        material: str | None = None,
        tags: list[str] | None = None,
        annotations: list[str] | None = None,
        props: dict[str, Any] | None = None,
        bevy_components: dict[str, Any] | None = None,
        parent_id: str | None = None,
        visible: bool = True,
        snap: dict[str, Any] | bool | None = None,
    ) -> dict[str, Any]:
        """Add a new object to the project. Returns the object dict.

        *snap* controls export-time raycasting onto scene geometry:

        - ``True`` or ``{"axis": "-Z"}`` — snap down onto terrain (default)
        - ``{"axis": "+Z"}`` — snap upward (ceiling mount)
        - ``{"axis": "-X"}`` — snap onto +X-facing wall
        - ``{"axis": "-Z", "policy": "ORIENT"}`` — snap + align to surface
        - ``{"axis": "-Z", "exclude_tags": ["vegetation"]}`` — skip veg
        - ``{"axis": "-Z", "target_tags": ["floor"]}`` — snap only to floor

        Supported axes: ``-Z +Z -X +X -Y +Y``.
        Policies: ``FIRST LAST HIGHEST LOWEST AVERAGE ORIENT``.

        Placement warnings are emitted to the ``blender_cli`` logger
        at WARNING level.  Nearby objects are logged at INFO level.
        """
        import logging

        if primitive is not None:
            ptype = primitive.get("type")
            if ptype not in _PRIMITIVE_TYPES:
                msg = f"Unknown primitive type {ptype!r}. Valid: {sorted(_PRIMITIVE_TYPES)}"
                raise ValueError(msg)
        loc = list(location or [0.0, 0.0, 0.0])
        scl = list(scale or [1.0, 1.0, 1.0])
        obj: dict[str, Any] = {
            "uid": str(uuid.uuid4()),
            "name": name,
            "type": "MESH" if (primitive or asset_path) else "EMPTY",
            "location": loc,
            "rotation": list(rotation or [0.0, 0.0, 0.0]),
            "scale": scl,
            "tags": list(tags or []),
            "annotations": list(annotations or []),
            "props": dict(props or {}),
            "parent_id": parent_id,
            "modifiers": [],
            "keyframes": [],
            "visible": visible,
        }
        if primitive is not None:
            obj["primitive"] = primitive
        if asset_path is not None:
            obj["asset_path"] = asset_path
        if material is not None:
            obj["material"] = material
        if bevy_components is not None:
            obj["bevy_components"] = dict(bevy_components)
        if snap is not None:
            if snap is True:
                obj["snap"] = {"axis": "-Z"}
            elif isinstance(snap, dict):
                obj["snap"] = snap

        # Emit placement feedback before appending
        log = logging.getLogger("blender_cli")
        warns = self.placement_warnings(name, loc, primitive, scl)
        for w in warns:
            log.warning(w)
        nearby = self.nearby_objects(loc, radius=20.0, limit=5)
        if nearby:
            names = ", ".join(f"{n['name']} ({n['distance']}m)" for n in nearby)
            log.info("add_object(%s): nearby [%s]", name, names)

        # Validate Bevy components against registry (warn only, never block).
        if bevy_components:
            reg = self._load_registry()
            if reg is not None:
                for comp_name, comp_value in bevy_components.items():
                    info = reg.find(comp_name)
                    if info is None:
                        suggestions = reg.suggest(comp_name)
                        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
                        log.warning("add_object(%s): unknown Bevy component %r.%s", name, comp_name, hint)
                    else:
                        for w in reg.validate_value(comp_name, comp_value):
                            log.warning("add_object(%s): %s", name, w)

        self.data["objects"].append(obj)
        return obj

    # -- Instance management -------------------------------------------------

    def add_instance(
        self,
        name: str,
        prefab: str,
        points: list[list[float]],
        *,
        rotations: list[float] | None = None,
        scales: list[float] | None = None,
        align: str = "bottom",
        material: str | None = None,
        tags: list[str] | None = None,
        annotations: list[str] | None = None,
        props: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a GPU-instanced object group."""
        inst: dict[str, Any] = {
            "uid": str(uuid.uuid4()),
            "name": name,
            "prefab": prefab,
            "points": [list(p) for p in points],
            "rotations": list(rotations) if rotations else [0.0] * len(points),
            "scales": list(scales) if scales else [1.0] * len(points),
            "align": align,
            "tags": list(tags or []),
            "annotations": list(annotations or []),
            "props": dict(props or {}),
        }
        if material is not None:
            inst["material"] = material
        self.data["instances"].append(inst)
        return inst

    def remove_instance(self, name_or_index: str | int) -> dict[str, Any]:
        """Remove an instance group by name or index."""
        instances = self.data["instances"]
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(instances):
                msg = f"Instance index {name_or_index} out of range"
                raise IndexError(msg)
            return instances.pop(name_or_index)
        for i, inst in enumerate(instances):
            if inst["name"] == name_or_index:
                return instances.pop(i)
        msg = f"No instance named {name_or_index!r}"
        raise KeyError(msg)

    # -- Camera management ---------------------------------------------------

    def add_camera(
        self,
        name: str,
        *,
        location: list[float] | None = None,
        rotation: list[float] | None = None,
        look_at: list[float] | None = None,
        lens: float = 50.0,
        clip_start: float = 0.1,
        clip_end: float = 1000.0,
        camera_type: str = "PERSP",
        path: dict[str, Any] | None = None,
        ghost: bool = False,
        bevy_components: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a camera (static or with an animation path).

        If *look_at* is set, *rotation* is computed to point at that position.
        If *ghost* is True, the camera is stored in project.json but not
        exported into the GLB — useful for debug/render-only viewpoints.
        """
        import math

        if path is not None:
            ptype = path.get("type")
            if ptype not in _CAMERA_PATH_TYPES:
                msg = f"Unknown camera path type {ptype!r}. Valid: {sorted(_CAMERA_PATH_TYPES)}"
                raise ValueError(msg)

        loc = list(location or [0.0, 0.0, 0.0])

        # Compute rotation from look_at target
        if look_at is not None and rotation is None:
            dx = look_at[0] - loc[0]
            dy = look_at[1] - loc[1]
            dz = look_at[2] - loc[2]
            dist_xy = math.sqrt(dx * dx + dy * dy)
            rotation = [
                math.atan2(dist_xy, -dz),       # pitch
                0.0,                              # roll
                math.atan2(dy, dx) - math.pi / 2,  # yaw
            ]

        cam: dict[str, Any] = {
            "uid": str(uuid.uuid4()),
            "name": name,
            "type": camera_type,
            "location": loc,
            "rotation": list(rotation or [0.0, 0.0, 0.0]),
            "lens": lens,
            "clip_start": clip_start,
            "clip_end": clip_end,
        }
        if path is not None:
            cam["path"] = path
        if look_at is not None:
            cam["look_at"] = list(look_at)
        if ghost:
            cam["ghost"] = True
        if bevy_components is not None:
            cam["bevy_components"] = dict(bevy_components)
        self.data["cameras"].append(cam)
        return cam

    def remove_camera(self, name_or_index: str | int) -> dict[str, Any]:
        """Remove a camera by name or index."""
        cameras = self.data["cameras"]
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(cameras):
                msg = f"Camera index {name_or_index} out of range"
                raise IndexError(msg)
            return cameras.pop(name_or_index)
        for i, cam in enumerate(cameras):
            if cam["name"] == name_or_index:
                return cameras.pop(i)
        msg = f"No camera named {name_or_index!r}"
        raise KeyError(msg)

    # -- Light management ----------------------------------------------------

    def add_light(
        self,
        name: str,
        light_type: str = "SUN",
        *,
        location: list[float] | None = None,
        rotation: list[float] | None = None,
        energy: float = 1.0,
        color: list[float] | None = None,
        bevy_components: dict[str, Any] | None = None,
        ghost: bool = False,
    ) -> dict[str, Any]:
        """Add a light source.

        If *ghost* is True, the light is stored in project.json but not
        exported into the GLB — useful for debug/render-only lighting.
        """
        if light_type not in {"SUN", "POINT", "SPOT", "AREA"}:
            msg = f"Unknown light type {light_type!r}"
            raise ValueError(msg)
        light: dict[str, Any] = {
            "uid": str(uuid.uuid4()),
            "name": name,
            "type": light_type,
            "location": list(location or [0.0, 0.0, 0.0]),
            "rotation": list(rotation or [0.0, 0.0, 0.0]),
            "energy": energy,
            "color": list(color or [1.0, 1.0, 1.0]),
        }
        if bevy_components is not None:
            light["bevy_components"] = dict(bevy_components)
        if ghost:
            light["ghost"] = True
        self.data["lights"].append(light)
        return light

    def remove_light(self, name_or_index: str | int) -> dict[str, Any]:
        """Remove a light by name or index."""
        lights = self.data["lights"]
        if isinstance(name_or_index, int):
            if name_or_index < 0 or name_or_index >= len(lights):
                msg = f"Light index {name_or_index} out of range"
                raise IndexError(msg)
            return lights.pop(name_or_index)
        for i, light in enumerate(lights):
            if light["name"] == name_or_index:
                return lights.pop(i)
        msg = f"No light named {name_or_index!r}"
        raise KeyError(msg)

    # -- GLB bridge ----------------------------------------------------------

    def export_glb(self, out: str | Path) -> Path:
        """Export the project to a GLB file via Blender.

        Full scene builder: replays terrain recipe, creates primitives,
        loads prefabs, builds GPU instances, loads PBR materials, sets up
        cameras (static + path-animated), lights, modifiers, keyframes,
        world/render settings, and saves as GLB.
        """
        import math

        import bpy

        from blender_cli.scene import Scene

        out = Path(out)
        scene = Scene.new()

        # -- Resolve material name → bpy Material -------------------------
        mat_cache: dict[str, Any] = {}  # name → Material object

        def _resolve_material(name: str | None) -> Any:
            if not name:
                return None
            if name in mat_cache:
                return mat_cache[name]
            mat_def = self.find_material(name)
            if mat_def is None:
                return None

            import bpy as _bpy
            from blender_cli.assets.material import Material

            pbr = mat_def.get("pbr_folder")
            if pbr and Path(pbr).is_dir():
                tiling = mat_def.get("tiling")
                tile = tiling[0] if tiling else 1.0
                mat = Material.from_pbr_folder(
                    name, pbr, tile_scale=tile,
                )
            else:
                # Create material with Principled BSDF, then set properties
                mat = Material(name=name)
                bpy_mat = mat.get_or_create()
                bsdf = bpy_mat.node_tree.nodes.get("Principled BSDF") if bpy_mat.node_tree else None
                if bsdf:
                    color = mat_def.get("color", [0.5, 0.5, 0.5, 1.0])
                    if len(color) == 3:
                        color = [*color, 1.0]
                    bc = bsdf.inputs.get("Base Color")
                    if bc:
                        bc.default_value = tuple(color)
                    rough = bsdf.inputs.get("Roughness")
                    if rough:
                        rough.default_value = mat_def.get("roughness", 0.5)
                    metal = bsdf.inputs.get("Metallic")
                    if metal:
                        metal.default_value = mat_def.get("metallic", 0.0)
            mat_cache[name] = mat
            return mat

        # -- 1. Terrain ----------------------------------------------------
        terrain = self.data.get("terrain")
        if terrain is not None:
            from blender_cli.geometry import Heightfield

            w = terrain["width"]
            h = terrain["height"]
            mpp = terrain.get("meters_per_px", 1.0)

            hf = Heightfield.flat(w, h, z=0.0, meters_per_px=mpp)
            for op in terrain.get("operations", []):
                op_type = op["op"]
                if op_type == "flat":
                    hf = Heightfield.flat(w, h, z=op.get("z", 0.0), meters_per_px=mpp)
                elif op_type == "noise":
                    hf = hf.add_noise(
                        op.get("type", "fbm"),
                        amp=op.get("amp", 10.0),
                        freq=op.get("freq", 0.01),
                        seed=op.get("seed", terrain.get("seed", 0)),
                        octaves=op.get("octaves", 4),
                    )
                elif op_type == "smooth":
                    hf = hf.smooth(
                        radius=op.get("radius", 3),
                        iters=op.get("iters", 1),
                    )
                elif op_type == "terrace":
                    hf = hf.terrace(
                        steps=op.get("steps", 5),
                        strength=op.get("strength", 0.5),
                    )
                elif op_type == "clamp":
                    hf = hf.clamp(
                        min_z=op.get("min_z", 0.0),
                        max_z=op.get("max_z", 100.0),
                    )
                elif op_type == "stamp":
                    hf = hf.stamp(
                        shape=op.get("shape", "circle"),
                        center=tuple(op.get("center", [w // 2, h // 2])),
                        radius=op.get("radius", 20.0),
                        operation=op.get("operation", "add"),
                        amount=op.get("amount", -2.0),
                        falloff=op.get("falloff", "smooth"),
                    )
                elif op_type == "erode":
                    hf = hf.erode(
                        type=op.get("type", "hydraulic"),
                        iterations=op.get("iterations", 30),
                    )
                elif op_type == "radial_falloff":
                    hf = hf.radial_falloff(
                        center=tuple(op.get("center", [w // 2, h // 2])),
                        radius=op.get("radius", float(min(w, h)) / 2),
                        edge_width=op.get("edge_width", 20.0),
                        curve=op.get("curve", "smooth"),
                    )
                elif op_type == "remap_curve":
                    hf = hf.remap_curve(
                        points=[tuple(p) for p in op.get("points", [[0, 0], [1, 1]])],
                    )

            mesh_params = terrain.get("mesh", {})
            terrain_mat = _resolve_material(terrain.get("material"))
            terrain_entity = hf.to_mesh(
                lod=mesh_params.get("lod", 2),
                skirts=mesh_params.get("skirts", 5.0),
                material=terrain_mat,
                tile_scale=mesh_params.get("tile_scale", 5.0),
            )
            scene.add(
                terrain_entity,
                name="terrain",
                tags={"terrain"},
                annotations={"GROUND_SURFACE"},
            )

        # -- 2. Anchors ----------------------------------------------------
        for anchor_data in self.data.get("anchors", []):
            pos = anchor_data["position"]
            ann = anchor_data.get("annotation", anchor_data["name"].upper())
            from blender_cli.types import Vec3 as _Vec3
            scene.ensure_anchor(
                anchor_data["name"],
                ann,
                _Vec3(pos[0], pos[1], pos[2]),
            )

        # -- 3. Materials (pre-resolve all) --------------------------------
        for mat_def in self.data.get("materials", []):
            _resolve_material(mat_def["name"])

        # -- 4. Objects (primitives + prefabs + empties) -------------------
        uid_to_obj: dict[str, bpy.types.Object] = {}

        for obj_data in self.data["objects"]:
            loc = obj_data.get("location", [0, 0, 0])
            rot = obj_data.get("rotation", [0, 0, 0])
            scl = obj_data.get("scale", [1, 1, 1])
            obj_name = obj_data.get("name", "object")
            obj_mat = _resolve_material(obj_data.get("material"))

            bpy_obj: bpy.types.Object | None = None
            entity = None

            # Primitive geometry
            prim = obj_data.get("primitive")
            if prim:
                from blender_cli.scene import primitives as prim_mod

                ptype = prim["type"]
                if ptype == "box":
                    entity = prim_mod.box(
                        obj_name,
                        size=tuple(prim.get("size", [1, 1, 1])),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )
                elif ptype == "plane":
                    entity = prim_mod.plane(
                        obj_name,
                        size=tuple(prim.get("size", [1, 1])),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )
                elif ptype == "cylinder":
                    entity = prim_mod.cylinder(
                        obj_name,
                        radius=prim.get("radius", 0.5),
                        height=prim.get("height", 1.0),
                        segments=prim.get("segments", 32),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )
                elif ptype == "sphere":
                    entity = prim_mod.sphere(
                        obj_name,
                        radius=prim.get("radius", 0.5),
                        segments=prim.get("segments", 32),
                        rings=prim.get("rings", 16),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )
                elif ptype == "cone":
                    entity = prim_mod.cone(
                        obj_name,
                        radius1=prim.get("radius1", 1.0),
                        radius2=prim.get("radius2", 0.0),
                        depth=prim.get("depth", 2.0),
                        vertices=prim.get("vertices", 32),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )
                elif ptype == "torus":
                    entity = prim_mod.torus(
                        obj_name,
                        major_radius=prim.get("major_radius", 1.0),
                        minor_radius=prim.get("minor_radius", 0.25),
                        major_segments=prim.get("major_segments", 48),
                        minor_segments=prim.get("minor_segments", 12),
                        material=obj_mat,
                        tile_scale=prim.get("tile_scale"),
                    )

                if entity is not None:
                    entity.at(*loc).rot(*rot, degrees=False)
                    entity.target.scale = tuple(scl)

            # Asset (prefab) loading
            asset_path = obj_data.get("asset_path")
            if entity is None and asset_path and Path(asset_path).exists():
                before = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(filepath=str(Path(asset_path)))
                after = set(bpy.data.objects)
                new_objs = after - before
                if new_objs:
                    bpy_obj = next(iter(new_objs))
                    bpy_obj.name = obj_name
                    bpy_obj.location = tuple(loc)
                    bpy_obj.rotation_euler = tuple(rot)
                    bpy_obj.scale = tuple(scl)

            # Fallback: create empty
            if entity is None and bpy_obj is None:
                bpy.ops.object.empty_add(location=tuple(loc))
                bpy_obj = bpy.context.active_object
                if bpy_obj is not None:
                    bpy_obj.name = obj_name
                    bpy_obj.rotation_euler = tuple(rot)
                    bpy_obj.scale = tuple(scl)

            # Get final bpy_obj reference
            target = entity.target if entity is not None else bpy_obj
            if target is None:
                continue

            if not obj_data.get("visible", True):
                target.hide_render = True
                target.hide_viewport = True

            # Attach Blenvy/Bevy components to the entity before adding.
            obj_bevy = obj_data.get("bevy_components")
            if obj_bevy and entity is not None:
                for comp_name, comp_val in obj_bevy.items():
                    entity.component(comp_name, comp_val)
            elif obj_bevy and target is not None:
                from blender_cli.blenvy import apply_bevy_components
                apply_bevy_components(target, obj_bevy)

            # Add to scene with metadata
            add_target = entity if entity is not None else target
            result = scene.add(
                add_target,
                name=obj_name,
                tags=set(obj_data.get("tags", [])),
                annotations=set(obj_data.get("annotations", [])),
                props=obj_data.get("props", {}),
            )

            uid = obj_data.get("uid", "")
            if uid:
                uid_to_obj[uid] = target

        # Resolve parent relationships
        for obj_data in self.data["objects"]:
            parent_id = obj_data.get("parent_id")
            uid = obj_data.get("uid", "")
            if parent_id and uid and parent_id in uid_to_obj and uid in uid_to_obj:
                uid_to_obj[uid].parent = uid_to_obj[parent_id]

        # -- 4b. Snap objects via raycast -----------------------------------
        _SNAP_FAR = 10_000.0
        _AXIS_MAP = {
            "-X": (-1, 0, 0), "+X": (1, 0, 0),
            "-Y": (0, -1, 0), "+Y": (0, 1, 0),
            "-Z": (0, 0, -1), "+Z": (0, 0, 1),
        }
        snap_objs = [
            (obj_data, uid_to_obj.get(obj_data.get("uid", "")))
            for obj_data in self.data["objects"]
            if obj_data.get("snap") and obj_data.get("uid", "") in uid_to_obj
        ]
        if snap_objs:
            import bpy as _snap_bpy

            # Optionally hide objects by tag for snap filtering
            for obj_data, bpy_target in snap_objs:
                if bpy_target is None:
                    continue
                snap_spec = obj_data["snap"]
                axis_str = snap_spec.get("axis", "-Z")
                dx, dy, dz = _AXIS_MAP.get(axis_str, (0, 0, -1))

                # Filter: hide excluded/non-target objects during raycast
                hidden_for_snap: list = []
                exclude_tags = set(snap_spec.get("exclude_tags", []))
                target_tags = set(snap_spec.get("target_tags", []))
                if exclude_tags or target_tags:
                    for other in scene.bpy_scene.collection.all_objects:
                        if other is bpy_target:
                            continue
                        other_tags = set()
                        try:
                            other_tags = Scene.tags(other)
                        except Exception:
                            pass
                        should_hide = False
                        if target_tags and not (other_tags & target_tags):
                            should_hide = True
                        if exclude_tags and (other_tags & exclude_tags):
                            should_hide = True
                        if should_hide:
                            other.hide_viewport = True
                            hidden_for_snap.append(other)

                # Also hide self
                bpy_target.hide_viewport = True

                # Compute ray origin: opposite of direction, far away
                loc = bpy_target.location
                origin = (
                    loc.x if dx == 0 else loc.x + (-dx * _SNAP_FAR),
                    loc.y if dy == 0 else loc.y + (-dy * _SNAP_FAR),
                    loc.z if dz == 0 else loc.z + (-dz * _SNAP_FAR),
                )
                direction = (float(dx), float(dy), float(dz))

                depsgraph = _snap_bpy.context.evaluated_depsgraph_get()
                hit, loc_hit, normal_hit, _, _, _ = _snap_bpy.context.scene.ray_cast(
                    depsgraph, origin, direction,
                )

                # Restore hidden objects
                bpy_target.hide_viewport = False
                for h in hidden_for_snap:
                    h.hide_viewport = False

                if hit:
                    # Compute half-extent along snap axis for bottom-alignment
                    prim = obj_data.get("primitive")
                    hx, hy, hz = self._half_extents(prim, obj_data.get("scale"))
                    half = [hx, hy, hz]
                    # Find which axis is the snap axis
                    for ax_idx in range(3):
                        d_comp = [dx, dy, dz][ax_idx]
                        if abs(d_comp) > 0.5:
                            # Offset object so its face sits on the hit point
                            setattr(
                                bpy_target.location,
                                "xyz"[ax_idx],
                                loc_hit[ax_idx] + half[ax_idx] * (-d_comp),
                            )

                    # ORIENT policy: align to surface normal
                    if snap_spec.get("policy") == "ORIENT" and normal_hit:
                        import math as _m
                        nx, ny, nz = normal_hit.x, normal_hit.y, normal_hit.z
                        pitch = _m.atan2(-ny, nz)
                        roll = _m.atan2(nx, nz)
                        bpy_target.rotation_euler = (pitch, roll, bpy_target.rotation_euler.z)

        # -- 5. Instances (GPU-instanced groups) ---------------------------
        for inst_data in self.data.get("instances", []):
            prefab_path = inst_data.get("prefab", "")
            if not prefab_path or not Path(prefab_path).exists():
                continue

            from blender_cli.geometry import PointSet
            from blender_cli.scene.instances import Instances

            points = inst_data.get("points", [])
            rotations = inst_data.get("rotations", [0.0] * len(points))
            scales = inst_data.get("scales", [1.0] * len(points))
            align = inst_data.get("align", "bottom")

            # Load prefab
            from blender_cli.assets.registry import AssetRegistry
            registry = scene.assets
            prefab = registry.load(prefab_path)

            from blender_cli.types import Vec3
            vec3_points = [Vec3(*p) for p in points]

            instances = Instances(
                prefab=prefab,
                points=vec3_points,
                rotations=list(rotations),
                scales=list(scales),
                align=align,
                name=inst_data.get("name", "instances"),
                tags=set(inst_data.get("tags", [])),
                annotations=set(inst_data.get("annotations", [])),
                props=inst_data.get("props", {}),
            )
            scene.add(instances)

        # -- 6. Cameras (static + path-animated) --------------------------
        for cam_data in self.data.get("cameras", []):
            if cam_data.get("ghost"):
                continue  # ghost cameras are not exported to GLB
            loc = cam_data.get("location", [0, 0, 0])
            rot = cam_data.get("rotation", [0, 0, 0])
            cam_bpy = bpy.data.cameras.new(cam_data.get("name", "Camera"))
            cam_bpy.lens = cam_data.get("lens", 50.0)
            cam_bpy.clip_start = cam_data.get("clip_start", 0.1)
            cam_bpy.clip_end = cam_data.get("clip_end", 1000.0)
            if cam_data.get("type") == "ORTHO":
                cam_bpy.type = "ORTHO"
            cam_obj = bpy.data.objects.new(cam_data.get("name", "Camera"), cam_bpy)
            cam_obj.location = tuple(loc)
            cam_obj.rotation_euler = tuple(rot)
            scene.bpy_scene.collection.objects.link(cam_obj)
            # Apply Bevy/Blenvy components
            cam_bcomps = cam_data.get("bevy_components")
            if cam_bcomps:
                from blender_cli.blenvy import apply_bevy_components
                apply_bevy_components(cam_obj, cam_bcomps)

            # Animate camera path
            path_def = cam_data.get("path")
            if path_def:
                path_type = path_def.get("type")
                frames = path_def.get("frames", 120)

                if path_type == "orbit":
                    center = path_def.get("center", [0, 0, 0])
                    radius = path_def.get("radius", 50.0)
                    elevation = math.radians(path_def.get("elevation", 30.0))

                    for f in range(frames):
                        angle = 2 * math.pi * f / frames
                        x = center[0] + radius * math.cos(angle) * math.cos(elevation)
                        y = center[1] + radius * math.sin(angle) * math.cos(elevation)
                        z = center[2] + radius * math.sin(elevation)
                        cam_obj.location = (x, y, z)
                        cam_obj.keyframe_insert(data_path="location", frame=f + 1)

                        # Point at center
                        dx, dy, dz = center[0] - x, center[1] - y, center[2] - z
                        cam_obj.rotation_euler = (
                            math.atan2(math.sqrt(dx**2 + dy**2), -dz),
                            0.0,
                            math.atan2(dy, dx) - math.pi / 2,
                        )
                        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f + 1)

                elif path_type == "dolly":
                    start = path_def.get("start", [0, 0, 10])
                    end = path_def.get("end", [100, 100, 10])
                    look_at = path_def.get("look_at", [50, 50, 0])

                    for f in range(frames):
                        t = f / max(frames - 1, 1)
                        x = start[0] + (end[0] - start[0]) * t
                        y = start[1] + (end[1] - start[1]) * t
                        z = start[2] + (end[2] - start[2]) * t
                        cam_obj.location = (x, y, z)
                        cam_obj.keyframe_insert(data_path="location", frame=f + 1)

                        dx = look_at[0] - x
                        dy = look_at[1] - y
                        dz = look_at[2] - z
                        cam_obj.rotation_euler = (
                            math.atan2(math.sqrt(dx**2 + dy**2), -dz),
                            0.0,
                            math.atan2(dy, dx) - math.pi / 2,
                        )
                        cam_obj.keyframe_insert(data_path="rotation_euler", frame=f + 1)

                elif path_type == "flyover":
                    waypoints = path_def.get("points", [])
                    look_at = path_def.get("look_at")
                    if len(waypoints) >= 2:
                        for f in range(frames):
                            t = f / max(frames - 1, 1)
                            seg_float = t * (len(waypoints) - 1)
                            seg_idx = min(int(seg_float), len(waypoints) - 2)
                            seg_t = seg_float - seg_idx
                            p0, p1 = waypoints[seg_idx], waypoints[seg_idx + 1]
                            x = p0[0] + (p1[0] - p0[0]) * seg_t
                            y = p0[1] + (p1[1] - p0[1]) * seg_t
                            z = p0[2] + (p1[2] - p0[2]) * seg_t
                            cam_obj.location = (x, y, z)
                            cam_obj.keyframe_insert(data_path="location", frame=f + 1)

                            if look_at:
                                dx = look_at[0] - x
                                dy = look_at[1] - y
                                dz = look_at[2] - z
                                cam_obj.rotation_euler = (
                                    math.atan2(math.sqrt(dx**2 + dy**2), -dz),
                                    0.0,
                                    math.atan2(dy, dx) - math.pi / 2,
                                )
                                cam_obj.keyframe_insert(data_path="rotation_euler", frame=f + 1)

        # -- 7. Lights -----------------------------------------------------
        for light_data in self.data.get("lights", []):
            if light_data.get("ghost"):
                continue  # ghost lights are not exported to GLB
            loc = light_data.get("location", [0, 0, 0])
            rot = light_data.get("rotation", [0, 0, 0])
            light_type = light_data.get("type", "POINT")
            light_bpy = bpy.data.lights.new(
                light_data.get("name", "Light"), type=light_type
            )
            light_bpy.energy = light_data.get("energy", 1.0)
            color = light_data.get("color", [1.0, 1.0, 1.0])
            light_bpy.color = (color[0], color[1], color[2])
            light_obj = bpy.data.objects.new(
                light_data.get("name", "Light"), light_bpy
            )
            light_obj.location = tuple(loc)
            light_obj.rotation_euler = tuple(rot)
            scene.bpy_scene.collection.objects.link(light_obj)
            # Apply Bevy/Blenvy components
            bcomps = light_data.get("bevy_components")
            if bcomps:
                from blender_cli.blenvy import apply_bevy_components
                apply_bevy_components(light_obj, bcomps)

        # -- 8. Keyframes on objects ---------------------------------------
        for obj_data in self.data["objects"]:
            kfs = obj_data.get("keyframes", [])
            uid = obj_data.get("uid", "")
            target = uid_to_obj.get(uid)
            if not target or not kfs:
                continue
            for kf in kfs:
                frame = kf.get("frame", 1)
                prop = kf.get("property", "")
                value = kf.get("value")
                interp = kf.get("interpolation", "BEZIER")
                if prop == "location" and isinstance(value, list):
                    target.location = tuple(value)
                    target.keyframe_insert(data_path="location", frame=frame)
                elif prop == "rotation" and isinstance(value, list):
                    target.rotation_euler = tuple(value)
                    target.keyframe_insert(data_path="rotation_euler", frame=frame)
                elif prop == "scale" and isinstance(value, list):
                    target.scale = tuple(value)
                    target.keyframe_insert(data_path="scale", frame=frame)
                elif prop == "visible" and isinstance(value, bool):
                    target.hide_render = not value
                    target.hide_viewport = not value
                    target.keyframe_insert(data_path="hide_render", frame=frame)
                    target.keyframe_insert(data_path="hide_viewport", frame=frame)

        # -- 9. World settings ---------------------------------------------
        world = self.data.get("world", {})
        if world.get("use_hdri") and world.get("hdri_path"):
            hdri = Path(world["hdri_path"])
            if hdri.exists():
                scene.set_world_hdri(
                    hdri,
                    rotation=world.get("hdri_rotation", 0.0),
                    strength=world.get("hdri_strength", 1.0),
                )
        elif world.get("background_color"):
            bg = world["background_color"]
            if bpy.context.scene.world is None:
                bpy.context.scene.world = bpy.data.worlds.new("World")
            w = bpy.context.scene.world
            w.use_nodes = True
            tree = w.node_tree
            if tree:
                tree.nodes.clear()
                bg_node = tree.nodes.new("ShaderNodeBackground")
                bg_node.inputs["Color"].default_value = (bg[0], bg[1], bg[2], 1.0)
                bg_node.inputs["Strength"].default_value = 1.0
                out_node = tree.nodes.new("ShaderNodeOutputWorld")
                tree.links.new(bg_node.outputs["Background"], out_node.inputs["Surface"])

        # -- 10. Render settings -------------------------------------------
        rnd = self.data.get("render", {})
        bpy_render = bpy.context.scene.render
        engine_map = {"CYCLES": "CYCLES", "EEVEE": "BLENDER_EEVEE", "WORKBENCH": "BLENDER_WORKBENCH"}
        if rnd.get("engine"):
            bpy_render.engine = engine_map.get(rnd["engine"], rnd["engine"])
        if rnd.get("resolution"):
            bpy_render.resolution_x = rnd["resolution"][0]
            bpy_render.resolution_y = rnd["resolution"][1]
        if rnd.get("samples"):
            bpy.context.scene.cycles.samples = rnd["samples"]
            bpy.context.scene.eevee.taa_render_samples = rnd["samples"]
        if rnd.get("resolution_pct"):
            bpy_render.resolution_percentage = rnd["resolution_pct"]
        bpy_render.film_transparent = rnd.get("film_transparent", False)

        # -- 11. Scene settings --------------------------------------------
        scn = self.data.get("scene", {})
        bpy.context.scene.frame_start = scn.get("frame_start", 1)
        bpy.context.scene.frame_end = scn.get("frame_end", 250)
        bpy.context.scene.render.fps = scn.get("fps", 24)

        # If any object has bevy_components, also write .meta.ron for Blenvy.
        has_bevy = any(
            obj_data.get("bevy_components")
            for obj_data in self.data["objects"]
        )
        scene.save(out, blenvy_meta=has_bevy)
        return out

    @classmethod
    def import_glb(cls, glb_path: str | Path, name: str | None = None) -> ProjectFile:
        """Import a GLB file into a new project.

        Reads the manifest from the GLB scene-level ``_scene_manifest``
        extra.  Falls back to iterating bpy scene objects when the manifest
        is absent or has no objects array.

        Extracts: objects (with asset_path), anchors, materials (PBR folder
        refs from texture paths), cameras (with type), and lights.
        """
        import bpy as _bpy

        from blender_cli.scene import Scene

        glb_path = Path(glb_path)
        scene = Scene.load(glb_path)

        project = cls.new(name or glb_path.stem)

        imported = False

        # Read manifest from GLB scene-level extras
        manifest: dict[str, Any] = {}
        manifest_str = _bpy.context.scene.get("_scene_manifest")
        if manifest_str:
            try:
                manifest = json.loads(manifest_str)
            except (json.JSONDecodeError, TypeError):
                manifest = {}

            manifest_objects = manifest.get("objects", [])
            if manifest_objects:
                for obj in manifest_objects:
                    obj_record: dict[str, Any] = {
                        "uid": obj.get("uid", ""),
                        "name": obj.get("name", ""),
                        "type": obj.get("type", "MESH"),
                        "tags": obj.get("tags", []),
                        "annotations": obj.get("annotations", []),
                        "props": obj.get("props", {}),
                        "location": obj.get("location", [0, 0, 0]),
                        "rotation": obj.get("rotation", [0, 0, 0]),
                        "scale": obj.get("scale", [1, 1, 1]),
                        "parent_id": obj.get("parent_uid"),
                        "modifiers": [],
                        "keyframes": [],
                        "visible": True,
                    }
                    # Preserve asset_path if present
                    if obj.get("asset_path"):
                        obj_record["asset_path"] = obj["asset_path"]
                    project.data["objects"].append(obj_record)
                imported = True

            # Import anchors
            anchors = manifest.get("anchors", [])
            if anchors:
                project.data["anchors"] = list(anchors)

            # Import material references from manifest assets
            mat_ids = manifest.get("assets", {}).get("material_ids", [])
            # (material_ids are just names; we can't reconstruct PBR paths
            # from GLB alone, but record them as basic materials)
            seen_mats: set[str] = set()
            for mid in mat_ids:
                if mid and mid not in seen_mats:
                    project.data["materials"].append({"name": mid})
                    seen_mats.add(mid)

        # Fallback: iterate all bpy scene objects directly
        if not imported:
            col = scene.bpy_scene.collection
            all_objs = col.all_objects if col else []
            for obj in all_objs:
                loc = obj.matrix_world.translation
                rot = obj.rotation_euler
                scl = obj.scale
                props = Scene.props(obj)
                obj_record = {
                    "uid": Scene.uid(obj) or str(uuid.uuid4()),
                    "name": obj.name,
                    "type": obj.type,
                    "tags": sorted(Scene.tags(obj)),
                    "annotations": sorted(Scene.annotations(obj)),
                    "props": props,
                    "location": [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
                    "rotation": [round(rot.x, 4), round(rot.y, 4), round(rot.z, 4)],
                    "scale": [round(scl.x, 4), round(scl.y, 4), round(scl.z, 4)],
                    "parent_id": (
                        Scene.uid(obj.parent) if obj.parent else None
                    ),
                    "modifiers": [],
                    "keyframes": [],
                    "visible": True,
                }
                # Extract asset_path from props if available
                ap = props.get("asset_path")
                if ap:
                    obj_record["asset_path"] = ap
                project.data["objects"].append(obj_record)

        # Import cameras (with type)
        for cam in scene.cameras():
            cam_obj = cam.bpy_object
            cam_data = cam.bpy_data
            loc = cam_obj.matrix_world.translation
            rot = cam_obj.rotation_euler
            project.data["cameras"].append({
                "uid": Scene.uid(cam_obj) or str(uuid.uuid4()),
                "name": cam_obj.name,
                "type": getattr(cam_data, "type", "PERSP"),
                "location": [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
                "rotation": [round(rot.x, 4), round(rot.y, 4), round(rot.z, 4)],
                "lens": round(cam_data.lens, 2),
                "clip_start": round(cam_data.clip_start, 4),
                "clip_end": round(cam_data.clip_end, 2),
            })

        # Import lights
        import bpy

        col = scene.bpy_scene.collection
        for obj in (col.all_objects if col else []):
            if obj.type == "LIGHT" and obj.data is not None:
                loc = obj.matrix_world.translation
                rot = obj.rotation_euler
                light_data = obj.data
                project.data["lights"].append({
                    "uid": Scene.uid(obj) or str(uuid.uuid4()),
                    "name": obj.name,
                    "type": getattr(light_data, "type", "POINT"),
                    "location": [round(loc.x, 4), round(loc.y, 4), round(loc.z, 4)],
                    "rotation": [round(rot.x, 4), round(rot.y, 4), round(rot.z, 4)],
                    "energy": getattr(light_data, "energy", 1.0),
                    "color": list(getattr(light_data, "color", [1.0, 1.0, 1.0])),
                })

        # Extract materials from bpy scene (PBR folder detection)
        seen_mats = {m["name"] for m in project.data["materials"]}
        for bpy_mat in bpy.data.materials:
            if bpy_mat.name in seen_mats or bpy_mat.name.startswith("Dots Stroke"):
                continue
            mat_record: dict[str, Any] = {"name": bpy_mat.name}
            # Try to detect PBR folder from texture image paths
            if bpy_mat.node_tree:
                for node in bpy_mat.node_tree.nodes:
                    if node.type == "TEX_IMAGE" and node.image and node.image.filepath:
                        tex_path = Path(node.image.filepath)
                        if tex_path.parent.is_dir():
                            mat_record["pbr_folder"] = str(tex_path.parent)
                            break
            # Extract base color from Principled BSDF
            if bpy_mat.node_tree:
                for node in bpy_mat.node_tree.nodes:
                    if node.type == "BSDF_PRINCIPLED":
                        bc = node.inputs.get("Base Color")
                        if bc and not bc.is_linked:
                            mat_record["color"] = list(bc.default_value)
                        rough = node.inputs.get("Roughness")
                        if rough and not rough.is_linked:
                            mat_record["roughness"] = rough.default_value
                        metal = node.inputs.get("Metallic")
                        if metal and not metal.is_linked:
                            mat_record["metallic"] = metal.default_value
                        break
            project.data["materials"].append(mat_record)
            seen_mats.add(bpy_mat.name)

        return project
