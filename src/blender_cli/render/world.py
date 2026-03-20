"""World/environment settings, validation, and bpy codegen."""

from __future__ import annotations

from typing import Any


class WorldSettings:
    """
    World environment settings with validation.

    Fields:
        background_color: [R, G, B] floats in 0..1
        use_hdri: whether to use an HDRI environment texture
        hdri_path: path to the HDRI file (str or None)
        hdri_strength: HDRI background strength (> 0)
        hdri_rotation: HDRI Z-rotation in radians
    """

    __slots__ = (
        "background_color",
        "hdri_path",
        "hdri_rotation",
        "hdri_strength",
        "use_hdri",
    )

    def __init__(
        self,
        background_color: list[float] | None = None,
        use_hdri: bool = False,
        hdri_path: str | None = None,
        hdri_strength: float = 1.0,
        hdri_rotation: float = 0.0,
    ) -> None:
        self.background_color = (
            background_color if background_color is not None else [0.05, 0.05, 0.05]
        )
        self.use_hdri = use_hdri
        self.hdri_path = hdri_path
        self.hdri_strength = hdri_strength
        self.hdri_rotation = hdri_rotation
        self.validate()

    def validate(self) -> None:
        """Raise ValueError if any field is invalid."""
        errors = validate_world(self.to_dict())
        if errors:
            raise ValueError("; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for project JSON."""
        return {
            "background_color": list(self.background_color),
            "use_hdri": self.use_hdri,
            "hdri_path": self.hdri_path,
            "hdri_strength": self.hdri_strength,
            "hdri_rotation": self.hdri_rotation,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorldSettings:
        """Create from a project JSON world dict."""
        return cls(
            background_color=d.get("background_color", [0.05, 0.05, 0.05]),
            use_hdri=d.get("use_hdri", False),
            hdri_path=d.get("hdri_path"),
            hdri_strength=d.get("hdri_strength", 1.0),
            hdri_rotation=d.get("hdri_rotation", 0.0),
        )

    def __repr__(self) -> str:
        return (
            f"WorldSettings(bg={self.background_color}, "
            f"hdri={self.use_hdri}, path={self.hdri_path!r}, "
            f"strength={self.hdri_strength}, rotation={self.hdri_rotation})"
        )


def validate_world(world: dict[str, Any]) -> list[str]:
    """Validate world settings dict. Returns list of error messages."""
    errors: list[str] = []

    bg = world.get("background_color")
    if bg is not None:
        if not (
            isinstance(bg, list)
            and len(bg) == 3
            and all(isinstance(v, (int, float)) for v in bg)
        ):
            errors.append("background_color must be [R, G, B] with numeric values")
        elif not all(0.0 <= v <= 1.0 for v in bg):
            errors.append("background_color values must be in range 0..1")

    if "hdri_strength" in world:
        s = world["hdri_strength"]
        if not isinstance(s, (int, float)) or s <= 0:
            errors.append("hdri_strength must be a positive number")

    if "hdri_rotation" in world:
        r = world["hdri_rotation"]
        if not isinstance(r, (int, float)):
            errors.append("hdri_rotation must be a number (radians)")

    return errors


# ---------------------------------------------------------------------------
# bpy codegen
# ---------------------------------------------------------------------------


def world_codegen(world: dict[str, Any]) -> str:
    """
    Generate bpy code to configure world/environment settings.

    Produces code that sets up either a solid background color or an
    HDRI environment texture with rotation and strength.
    """
    lines: list[str] = ["import bpy", "bs = bpy.context.scene", ""]

    # Ensure world exists
    lines.extend((
        "if bs.world is None:",
        "    bs.world = bpy.data.worlds.new('World')",
        "world = bs.world",
        "world.use_nodes = True",
        "tree = world.node_tree",
        "",
    ))

    use_hdri = world.get("use_hdri", False)
    hdri_path = world.get("hdri_path")

    if use_hdri and hdri_path:
        rotation = world.get("hdri_rotation", 0.0)
        strength = world.get("hdri_strength", 1.0)

        # HDRI node setup: TexCoord -> Mapping -> EnvironmentTexture -> Background -> WorldOutput
        lines.extend((
            "# Clear existing nodes",
            "for n in list(tree.nodes):",
            "    tree.nodes.remove(n)",
            "",
            "tc = tree.nodes.new('ShaderNodeTexCoord')",
            "mp = tree.nodes.new('ShaderNodeMapping')",
            f"mp.inputs['Rotation'].default_value[2] = {rotation!r}",
            "et = tree.nodes.new('ShaderNodeTexEnvironment')",
            f"et.image = bpy.data.images.load({str(hdri_path)!r})",
            "bg = tree.nodes.new('ShaderNodeBackground')",
            f"bg.inputs['Strength'].default_value = {strength!r}",
            "wo = tree.nodes.new('ShaderNodeOutputWorld')",
            "",
            "tree.links.new(tc.outputs['Generated'], mp.inputs['Vector'])",
            "tree.links.new(mp.outputs['Vector'], et.inputs['Vector'])",
            "tree.links.new(et.outputs['Color'], bg.inputs['Color'])",
            "tree.links.new(bg.outputs['Background'], wo.inputs['Surface'])",
        ))
    else:
        # Solid background color
        bg = world.get("background_color", [0.05, 0.05, 0.05])
        lines.extend((
            "# Solid background color",
            "bg = tree.nodes.get('Background')",
            "if bg is None:",
            "    bg = tree.nodes.new('ShaderNodeBackground')",
            f"bg.inputs['Color'].default_value = ({bg[0]!r}, {bg[1]!r}, {bg[2]!r}, 1.0)",
        ))

    return "\n".join(lines)
