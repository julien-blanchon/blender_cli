"""Render engine settings, presets, and camera DOF configuration."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Valid engines and output formats
# ---------------------------------------------------------------------------

VALID_ENGINES = {"CYCLES", "EEVEE", "WORKBENCH"}
VALID_FORMATS = {"PNG", "JPEG", "TIFF", "OPEN_EXR", "HDR"}

# Blender uses different engine identifiers internally
_ENGINE_BPY_MAP = {
    "CYCLES": "CYCLES",
    "EEVEE": "BLENDER_EEVEE_NEXT",
    "WORKBENCH": "BLENDER_WORKBENCH",
}

# ---------------------------------------------------------------------------
# Render presets
# ---------------------------------------------------------------------------

RENDER_PRESETS: dict[str, dict[str, Any]] = {
    "cycles_default": {
        "engine": "CYCLES",
        "samples": 128,
        "denoising": True,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 100,
    },
    "cycles_high": {
        "engine": "CYCLES",
        "samples": 512,
        "denoising": True,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 100,
    },
    "cycles_preview": {
        "engine": "CYCLES",
        "samples": 32,
        "denoising": False,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 50,
    },
    "eevee_default": {
        "engine": "EEVEE",
        "samples": 64,
        "denoising": False,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 100,
    },
    "eevee_high": {
        "engine": "EEVEE",
        "samples": 256,
        "denoising": False,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 100,
    },
    "eevee_preview": {
        "engine": "EEVEE",
        "samples": 16,
        "denoising": False,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 50,
    },
    "workbench": {
        "engine": "WORKBENCH",
        "samples": 1,
        "denoising": False,
        "film_transparent": False,
        "output_format": "PNG",
        "resolution_pct": 100,
    },
}

# Map --quality CLI flag tiers to preset names
QUALITY_PRESET_MAP = {
    "draft": "eevee_preview",
    "preview": "eevee_default",
    "final": "cycles_default",
}

# ---------------------------------------------------------------------------
# RenderSettings
# ---------------------------------------------------------------------------


class RenderSettings:
    """
    Render engine settings with validation.

    Can be created directly or from a named preset.
    """

    __slots__ = (
        "denoising",
        "engine",
        "film_transparent",
        "output_format",
        "resolution_pct",
        "samples",
    )

    def __init__(
        self,
        engine: str = "EEVEE",
        samples: int = 64,
        denoising: bool = False,
        film_transparent: bool = False,
        output_format: str = "PNG",
        resolution_pct: int = 100,
    ) -> None:
        self.engine = engine
        self.samples = samples
        self.denoising = denoising
        self.film_transparent = film_transparent
        self.output_format = output_format
        self.resolution_pct = resolution_pct
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if any field is invalid."""
        if self.engine not in VALID_ENGINES:
            msg = f"engine must be one of {sorted(VALID_ENGINES)}, got {self.engine!r}"
            raise ValueError(msg)
        if not isinstance(self.samples, int) or self.samples < 1:
            msg = f"samples must be int >= 1, got {self.samples!r}"
            raise ValueError(msg)
        if self.output_format not in VALID_FORMATS:
            msg = f"output_format must be one of {sorted(VALID_FORMATS)}, got {self.output_format!r}"
            raise ValueError(msg)
        if not isinstance(self.resolution_pct, int) or not (
            1 <= self.resolution_pct <= 100
        ):
            msg = f"resolution_pct must be int 1..100, got {self.resolution_pct!r}"
            raise ValueError(msg)

    @classmethod
    def from_preset(cls, name: str) -> RenderSettings:
        """Create RenderSettings from a named preset."""
        if name not in RENDER_PRESETS:
            msg = f"Unknown render preset {name!r}. Available: {sorted(RENDER_PRESETS)}"
            raise ValueError(msg)
        return cls(**RENDER_PRESETS[name])

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for project JSON."""
        return {
            "engine": self.engine,
            "samples": self.samples,
            "denoising": self.denoising,
            "film_transparent": self.film_transparent,
            "output_format": self.output_format,
            "resolution_pct": self.resolution_pct,
        }

    def __repr__(self) -> str:
        return (
            f"RenderSettings(engine={self.engine!r}, samples={self.samples}, "
            f"denoising={self.denoising}, transparent={self.film_transparent}, "
            f"format={self.output_format!r}, resolution_pct={self.resolution_pct})"
        )


# ---------------------------------------------------------------------------
# Camera DOF
# ---------------------------------------------------------------------------


def validate_dof(dof: dict[str, Any]) -> list[str]:
    """Validate camera DOF dict. Returns list of error messages."""
    errors: list[str] = []
    if "dof_enabled" in dof and not isinstance(dof["dof_enabled"], bool):
        errors.append("dof_enabled must be bool")
    if "dof_focus_distance" in dof:
        v = dof["dof_focus_distance"]
        if not isinstance(v, (int, float)) or v < 0:
            errors.append("dof_focus_distance must be a non-negative number (metres)")
    if "dof_aperture" in dof:
        v = dof["dof_aperture"]
        if not isinstance(v, (int, float)) or v <= 0:
            errors.append("dof_aperture must be a positive number (f-stop)")
    return errors


def default_dof() -> dict[str, Any]:
    """Return default DOF settings dict."""
    return {
        "dof_enabled": False,
        "dof_focus_distance": 10.0,
        "dof_aperture": 2.8,
    }


# ---------------------------------------------------------------------------
# bpy codegen
# ---------------------------------------------------------------------------


def render_settings_codegen(
    settings: dict[str, Any], resolution: list[int] | None = None
) -> str:
    """
    Generate bpy code to configure render engine settings.

    Args:
        settings: Render settings dict from project JSON.
        resolution: Optional [width, height] from project render config.

    Returns:
        Multi-line Python code string.

    """
    lines: list[str] = ["import bpy", "bs = bpy.context.scene", ""]

    engine = settings.get("engine", "EEVEE")
    bpy_engine = _ENGINE_BPY_MAP.get(engine, "BLENDER_EEVEE_NEXT")
    lines.append(f"bs.render.engine = {bpy_engine!r}")

    # Resolution
    if resolution:
        lines.extend((
            f"bs.render.resolution_x = {resolution[0]}",
            f"bs.render.resolution_y = {resolution[1]}",
        ))

    pct = settings.get("resolution_pct", 100)
    lines.append(f"bs.render.resolution_percentage = {pct}")

    # Output format
    fmt = settings.get("output_format", "PNG")
    lines.append(f"bs.render.image_settings.file_format = {fmt!r}")

    # Film transparency
    if settings.get("film_transparent"):
        lines.extend((
            "bs.render.film_transparent = True",
            "bs.render.image_settings.color_mode = 'RGBA'",
        ))
    else:
        lines.append("bs.render.film_transparent = False")

    # Engine-specific samples
    samples = settings.get("samples", 64)
    if engine == "CYCLES":
        lines.append(f"bs.cycles.samples = {samples}")
        if settings.get("denoising"):
            lines.append("bs.cycles.use_denoising = True")
        else:
            lines.append("bs.cycles.use_denoising = False")
    elif engine == "EEVEE":
        lines.append(f"bs.eevee.taa_render_samples = {samples}")
    # WORKBENCH has no per-engine sample control

    return "\n".join(lines)


def camera_dof_codegen(cam_var: str, dof: dict[str, Any]) -> str:
    """
    Generate bpy code to configure camera DOF.

    Args:
        cam_var: Variable name for the camera data (bpy.types.Camera).
        dof: DOF settings dict.

    Returns:
        Multi-line Python code string.

    """
    if not dof.get("dof_enabled"):
        return f"{cam_var}.dof.use_dof = False"

    lines = [
        f"{cam_var}.dof.use_dof = True",
        f"{cam_var}.dof.focus_distance = {dof.get('dof_focus_distance', 10.0)}",
        f"{cam_var}.dof.aperture_fstop = {dof.get('dof_aperture', 2.8)}",
    ]
    return "\n".join(lines)
