"""Material asset wrapper — creates/references Blender materials with PBR support."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import bpy
import numpy as np
import numpy.typing as npt

from blender_cli.utils.strings import stem_matches_keywords

if TYPE_CHECKING:
    from blender_cli.assets.image import Image

# Keyword patterns for auto-detecting PBR texture channels from filenames.
# Each channel maps to lowercase substrings matched against the file stem.
_PBR_CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "base_color": [
        "diff",
        "color",
        "col",
        "albedo",
        "base_color",
        "basecolor",
        "diffuse",
    ],
    "normal": ["normal", "nor", "nrm"],
    "metallic": ["metal", "metallic", "metalness"],
    "roughness": ["rough", "roughness"],
    "ao": ["ao", "ambient_occlusion", "ambientocclusion", "occlusion"],
    "displacement": ["disp", "displacement", "height", "bump"],
}

_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".exr",
    ".hdr",
    ".bmp",
    ".tga",
}


def _detect_pbr_files(folder: Path) -> dict[str, Path]:
    """
    Scan *folder* for PBR texture files and return {channel: path} mapping.

    Matching is case-insensitive against the file stem.  The first match per
    channel wins (files are sorted for determinism).
    """
    image_files = sorted(
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    )

    detected: dict[str, Path] = {}
    for channel, keywords in _PBR_CHANNEL_KEYWORDS.items():
        for img_file in image_files:
            stem = img_file.stem.lower()
            if stem_matches_keywords(stem, keywords):
                detected[channel] = img_file
                break

    return detected


# Labels used to identify channel nodes for idempotent replacement.
_LABEL_PREFIX = "_mc_"

# Channel → BSDF input name mapping for texture overlay blending.
_OVERLAY_BSDF_INPUT: dict[str, str] = {
    "base_color": "Base Color",
    "roughness": "Roughness",
}


@dataclass(frozen=True, slots=True)
class Material:
    """
    Immutable reference to a Blender material by name.

    On construction, retrieves an existing material or creates a new one
    with a Principled BSDF node tree.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            msg = "Material name must not be empty"
            raise ValueError(msg)

    # -- Blender datablock access --

    def get_or_create(self) -> bpy.types.Material:
        """Return the Blender material, creating it with a Principled BSDF if needed."""
        mat = bpy.data.materials.get(self.name)
        if mat is not None:
            return mat
        mat = bpy.data.materials.new(name=self.name)
        # Blender 5+ defaults to use_nodes=True; ensure Principled BSDF exists.
        if not mat.node_tree.nodes.get("Principled BSDF"):
            mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
        return mat

    # -- internal helpers --

    def _tree_and_bsdf(
        self,
    ) -> tuple[bpy.types.NodeTree, bpy.types.Node]:
        """Return (node_tree, Principled BSDF) or raise."""
        mat = self.get_or_create()
        tree = mat.node_tree
        if tree is None:
            msg = "Material has no node tree"
            raise ValueError(msg)
        bsdf = tree.nodes.get("Principled BSDF")
        if bsdf is None:
            msg = "Material has no Principled BSDF node"
            raise ValueError(msg)
        return tree, bsdf

    @staticmethod
    def _clear_channel(tree: bpy.types.NodeTree, channel: str) -> None:
        """Remove all nodes labeled for *channel* (idempotency)."""
        label = f"{_LABEL_PREFIX}{channel}"
        to_remove = [n for n in tree.nodes if n.label == label]
        for n in to_remove:
            tree.nodes.remove(n)

    @staticmethod
    def _make_tex_node(
        tree: bpy.types.NodeTree,
        image: Image,
        channel: str,
        *,
        non_color: bool = False,
    ) -> bpy.types.Node:
        """Create a labeled ShaderNodeTexImage."""
        tex = tree.nodes.new("ShaderNodeTexImage")
        tex.label = f"{_LABEL_PREFIX}{channel}"
        tex.image = image.load()
        if non_color:
            if tex.image is None:
                msg = f"Failed to load image for channel {channel!r}"
                raise RuntimeError(msg)
            cs = tex.image.colorspace_settings
            if cs is None:
                msg = f"Image for channel {channel!r} has no colorspace settings"
                raise RuntimeError(msg)
            cs.name = "Non-Color"
        return tex

    # -- PBR folder auto-detect --

    @classmethod
    def from_pbr_folder(
        cls,
        name: str,
        path: str | Path,
        tile_scale: float = 1.0,
    ) -> Material:
        """
        Create a material by auto-detecting PBR maps in *path*.

        Scans the folder for image files matching common PBR naming conventions
        (Poly Haven, ambientCG, manual naming).  Only ``base_color`` is required;
        all other channels are silently skipped if not found.

        If *tile_scale* != 1.0, a shared Texture Coordinate → Mapping node pair
        is created and linked to every texture node for UV tiling.
        """
        from blender_cli.assets.image import Image

        folder = Path(path)
        if not folder.is_dir():
            msg = f"PBR folder not found: {folder}"
            raise FileNotFoundError(msg)

        detected = _detect_pbr_files(folder)

        if "base_color" not in detected:
            msg = f"No base color texture found in {folder}"
            raise FileNotFoundError(msg)

        mat = cls(name)

        # Channel → setter mapping (order matters: base_color before ao)
        setters: dict[str, str] = {
            "base_color": "set_base_color_texture",
            "normal": "set_normal_texture",
            "metallic": "set_metallic_texture",
            "roughness": "set_roughness_texture",
            "ao": "set_ao_texture",
            "displacement": "set_displacement_texture",
        }

        for channel, method_name in setters.items():
            file_path = detected.get(channel)
            if file_path is None:
                continue
            img = Image(file_path)
            getattr(mat, method_name)(img)

        if not math.isclose(tile_scale, 1.0):
            mat._set_tile_scale(tile_scale)

        return mat

    def _set_tile_scale(self, scale: float) -> None:
        """Add a shared Texture Coordinate → Mapping node pair and link to all texture nodes."""
        tree, _ = self._tree_and_bsdf()

        # Create shared UV nodes
        tex_coord = tree.nodes.new("ShaderNodeTexCoord")
        tex_coord.label = f"{_LABEL_PREFIX}uv_coord"

        mapping = tree.nodes.new("ShaderNodeMapping")
        mapping.label = f"{_LABEL_PREFIX}uv_mapping"
        mapping.inputs["Scale"].default_value = (scale, scale, scale)

        tree.links.new(tex_coord.outputs["UV"], mapping.inputs["Vector"])

        # Connect mapping to every ShaderNodeTexImage in the tree
        for node in tree.nodes:
            if node.type == "TEX_IMAGE":
                tree.links.new(mapping.outputs["Vector"], node.inputs["Vector"])

    # -- PBR helpers --

    def set_base_color(self, color: tuple[float, float, float, float]) -> None:
        """Set a flat RGBA base color (0..1 each) on the Principled BSDF."""
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "base_color")
        bsdf.inputs["Base Color"].default_value = color

    def set_base_color_texture(self, image: Image) -> None:
        """Connect an :class:`Image` as the base-color texture (idempotent)."""
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "base_color")
        # Also clear AO mix node since it sits between base_color and BSDF
        self._clear_channel(tree, "ao")
        tex = self._make_tex_node(tree, image, "base_color")
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

    def set_normal_texture(self, image: Image, strength: float = 1.0) -> None:
        """Connect an :class:`Image` as the normal map via a Normal Map node (idempotent)."""
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "normal")
        tex = self._make_tex_node(tree, image, "normal", non_color=True)
        normal_map = tree.nodes.new("ShaderNodeNormalMap")
        normal_map.label = f"{_LABEL_PREFIX}normal"
        normal_map.inputs["Strength"].default_value = strength
        tree.links.new(tex.outputs["Color"], normal_map.inputs["Color"])
        tree.links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    def set_metallic_texture(self, image: Image) -> None:
        """Connect an :class:`Image` as the metallic map (idempotent)."""
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "metallic")
        tex = self._make_tex_node(tree, image, "metallic", non_color=True)
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Metallic"])

    def set_roughness_texture(self, image: Image) -> None:
        """Connect an :class:`Image` as the roughness map (idempotent)."""
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "roughness")
        tex = self._make_tex_node(tree, image, "roughness", non_color=True)
        tree.links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])

    def set_ao_texture(self, image: Image, strength: float = 1.0) -> None:
        """
        Connect an :class:`Image` as AO via MixRGB multiply on base color (idempotent).

        The AO texture is multiplied with whatever is currently driving the
        Base Color input, darkening crevices.  *strength* controls the mix
        factor (1.0 = full AO effect, 0.0 = no effect).
        """
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "ao")

        # Capture current base color source (texture link or default value)
        bc_input = bsdf.inputs["Base Color"]
        existing_link = bc_input.links[0] if bc_input.links else None

        tex = self._make_tex_node(tree, image, "ao", non_color=True)

        mix = tree.nodes.new("ShaderNodeMix")
        mix.label = f"{_LABEL_PREFIX}ao"
        mix.data_type = "RGBA"
        mix.blend_type = "MULTIPLY"
        mix.inputs["Factor"].default_value = strength

        if existing_link:
            # Re-route: old source → mix A, AO → mix B, mix → BSDF
            tree.links.new(existing_link.from_socket, mix.inputs["A"])
            tree.links.remove(existing_link)
        else:
            # No texture linked — use the flat default value
            mix.inputs["A"].default_value = bc_input.default_value  # type: ignore[assignment]

        tree.links.new(tex.outputs["Color"], mix.inputs["B"])
        tree.links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    def set_displacement_texture(self, image: Image, scale: float = 0.1) -> None:
        """
        Connect an :class:`Image` as displacement (idempotent).

        Uses a ShaderNodeDisplacement connected to the Material Output
        displacement socket. Works as bump in EEVEE, real displacement in
        Cycles with Adaptive Subdivision.
        """
        tree, _bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "displacement")

        tex = self._make_tex_node(tree, image, "displacement", non_color=True)

        disp = tree.nodes.new("ShaderNodeDisplacement")
        disp.label = f"{_LABEL_PREFIX}displacement"
        disp.inputs["Scale"].default_value = scale

        tree.links.new(tex.outputs["Color"], disp.inputs["Height"])

        # Connect to Material Output displacement socket
        mat_output = tree.nodes.get("Material Output")
        if mat_output is None:
            mat_output = tree.nodes.new("ShaderNodeOutputMaterial")
        tree.links.new(disp.outputs["Displacement"], mat_output.inputs["Displacement"])

    def set_pbr(
        self,
        *,
        metallic: float | None = None,
        roughness: float | None = None,
    ) -> None:
        """Set scalar PBR properties on the Principled BSDF."""
        _, bsdf = self._tree_and_bsdf()
        if metallic is not None:
            bsdf.inputs["Metallic"].default_value = metallic
        if roughness is not None:
            bsdf.inputs["Roughness"].default_value = roughness

    # -- Alpha / transparency (glTF-compatible) --

    def set_alpha_mode(
        self,
        mode: str = "OPAQUE",
        cutoff: float = 0.5,
    ) -> None:
        """
        Set blend method (maps to glTF ``alphaMode``).

        *mode*: ``"OPAQUE"``, ``"BLEND"``, or ``"MASK"``.
        *cutoff*: alpha threshold for ``"MASK"`` mode.
        """
        mat = self.get_or_create()
        mode_upper = mode.upper()
        if mode_upper == "OPAQUE":
            mat.blend_method = "OPAQUE"
        elif mode_upper == "BLEND":
            mat.blend_method = "BLEND"
            mat.show_transparent_back = False
        elif mode_upper == "MASK":
            mat.blend_method = "CLIP"
            mat.alpha_threshold = cutoff
        else:
            msg = f"Unknown alpha mode '{mode}', expected: OPAQUE, BLEND, MASK"
            raise ValueError(msg)

    def set_vertex_color_alpha(self, attribute_name: str = "COLOR_0") -> None:
        """
        Connect vertex color alpha to the Principled BSDF Alpha input.

        Adds an Attribute node reading *attribute_name* (glTF standard:
        ``COLOR_0``) and wires its Alpha output to the BSDF Alpha input.
        """
        tree, bsdf = self._tree_and_bsdf()
        self._clear_channel(tree, "vertex_alpha")

        attr_node = tree.nodes.new("ShaderNodeAttribute")
        attr_node.label = f"{_LABEL_PREFIX}vertex_alpha"
        attr_node.attribute_name = attribute_name

        tree.links.new(attr_node.outputs["Alpha"], bsdf.inputs["Alpha"])

    # -- Texture overlay --

    @staticmethod
    def _numpy_to_bpy_image(
        array: npt.NDArray[np.floating], name: str
    ) -> bpy.types.Image:
        """
        Create a Blender image datablock from a 2D numpy array (0..1).

        The array is converted to RGBA float and flipped vertically
        (Blender stores images bottom-to-top).
        """
        h, w = array.shape[:2]
        img = bpy.data.images.new(name, width=w, height=h, float_buffer=True)
        # Flip vertically for Blender's coordinate system
        flipped = np.flipud(array.astype(np.float32))
        # Expand to RGBA (mask → R=G=B=A=value)
        rgba = np.ones((h, w, 4), dtype=np.float32)
        rgba[:, :, 0] = flipped
        rgba[:, :, 1] = flipped
        rgba[:, :, 2] = flipped
        rgba[:, :, 3] = flipped
        img.pixels.foreach_set(rgba.ravel())
        img.pack()
        return img

    @staticmethod
    def _extract_channel_image(
        tree: bpy.types.NodeTree, channel: str
    ) -> bpy.types.Node | None:
        """Find the TexImage node labeled ``_mc_<channel>``."""
        label = f"{_LABEL_PREFIX}{channel}"
        for node in tree.nodes:
            if node.label == label and node.type == "TEX_IMAGE":
                return node
        return None

    @staticmethod
    def _clear_nodes_by_label(tree: bpy.types.NodeTree, label: str) -> None:
        """Remove all nodes with the given label (idempotent cleanup)."""
        to_remove = [n for n in tree.nodes if n.label == label]
        for n in to_remove:
            tree.nodes.remove(n)

    def apply_texture_overlay(
        self,
        overlay: Material,
        mask_array: npt.NDArray[np.floating],
        channels: tuple[str, ...] = ("base_color", "roughness"),
        overlay_tile_scale: float | None = None,
    ) -> None:
        """
        Blend *overlay* textures onto this material using a mask.

        For each *channel*, inserts a Mix node controlled by *mask_array*
        that blends this material's texture (A) with the overlay's texture (B).
        The mask uses **Generated** texture coordinates for automatic alignment
        with the terrain mesh bounding box.

        Parameters
        ----------
        overlay:
            Material whose textures will be painted on top.
        mask_array:
            2D numpy array (0..1) — the spline corridor mask.
        channels:
            PBR channels to blend (default: base_color and roughness).
        overlay_tile_scale:
            UV tiling scale for overlay textures. If None, inherits from
            the overlay material's existing Mapping node or defaults to 1.0.

        """
        tree, bsdf = self._tree_and_bsdf()

        # Ensure overlay material exists and has a node tree
        overlay_mat = overlay.get_or_create()
        overlay_tree = overlay_mat.node_tree
        if overlay_tree is None:
            msg = "Overlay material has no node tree"
            raise ValueError(msg)

        # Create mask image datablock
        overlay_label = "_mc_overlay"
        self._clear_nodes_by_label(tree, overlay_label)

        mask_img = self._numpy_to_bpy_image(mask_array, f"{self.name}_overlay_mask")

        # Mask TexImage node using Generated coordinates
        tex_coord_gen = tree.nodes.new("ShaderNodeTexCoord")
        tex_coord_gen.label = overlay_label

        mask_tex = tree.nodes.new("ShaderNodeTexImage")
        mask_tex.label = overlay_label
        mask_tex.image = mask_img
        mask_tex.interpolation = "Linear"
        if mask_img.colorspace_settings is not None:
            mask_img.colorspace_settings.name = "Non-Color"
        tree.links.new(tex_coord_gen.outputs["Generated"], mask_tex.inputs["Vector"])

        # Overlay UV tiling: TexCoord:UV → Mapping(scale) for overlay textures
        overlay_uv_coord = tree.nodes.new("ShaderNodeTexCoord")
        overlay_uv_coord.label = overlay_label

        overlay_mapping: bpy.types.ShaderNodeMapping | None = None
        if overlay_tile_scale is not None and not math.isclose(overlay_tile_scale, 1.0):
            mapping_node = tree.nodes.new("ShaderNodeMapping")
            assert isinstance(mapping_node, bpy.types.ShaderNodeMapping)
            overlay_mapping = mapping_node
            overlay_mapping.label = overlay_label
            overlay_mapping.inputs["Scale"].default_value = (
                overlay_tile_scale,
                overlay_tile_scale,
                overlay_tile_scale,
            )
            tree.links.new(
                overlay_uv_coord.outputs["UV"], overlay_mapping.inputs["Vector"]
            )

        for channel in channels:
            bsdf_input_name = _OVERLAY_BSDF_INPUT.get(channel)
            if bsdf_input_name is None:
                continue

            bsdf_input = bsdf.inputs.get(bsdf_input_name)
            if bsdf_input is None:
                continue

            # Find what currently drives this BSDF input
            existing_link = bsdf_input.links[0] if bsdf_input.links else None

            # Copy overlay texture into this tree
            overlay_tex_node = self._extract_channel_image(overlay_tree, channel)
            if overlay_tex_node is None:
                continue  # overlay has no texture for this channel

            tex_node = tree.nodes.new("ShaderNodeTexImage")
            assert isinstance(tex_node, bpy.types.ShaderNodeTexImage)
            new_overlay_tex = tex_node
            new_overlay_tex.label = overlay_label
            assert isinstance(overlay_tex_node, bpy.types.ShaderNodeTexImage)
            new_overlay_tex.image = overlay_tex_node.image
            if channel != "base_color" and new_overlay_tex.image is not None:
                cs = new_overlay_tex.image.colorspace_settings
                if cs is not None:
                    cs.name = "Non-Color"

            # Connect UV tiling to overlay texture
            if overlay_mapping is not None:
                tree.links.new(
                    overlay_mapping.outputs["Vector"],
                    new_overlay_tex.inputs["Vector"],
                )
            else:
                tree.links.new(
                    overlay_uv_coord.outputs["UV"],
                    new_overlay_tex.inputs["Vector"],
                )

            # Create Mix node
            mix = tree.nodes.new("ShaderNodeMix")
            mix.label = overlay_label
            if channel == "base_color":
                mix.data_type = "RGBA"
            else:
                mix.data_type = "FLOAT"

            # Wire: mask → Factor
            tree.links.new(mask_tex.outputs["Color"], mix.inputs["Factor"])

            # Wire: existing terrain texture → A, overlay → B
            if channel == "base_color":
                if existing_link:
                    tree.links.new(existing_link.from_socket, mix.inputs["A"])
                    tree.links.remove(existing_link)
                new_overlay_out = new_overlay_tex.outputs["Color"]
                tree.links.new(new_overlay_out, mix.inputs["B"])
                tree.links.new(mix.outputs["Result"], bsdf_input)
            else:
                # Float channels (roughness, metallic, etc.)
                if existing_link:
                    tree.links.new(existing_link.from_socket, mix.inputs["A"])
                    tree.links.remove(existing_link)
                new_overlay_out = new_overlay_tex.outputs["Color"]
                tree.links.new(new_overlay_out, mix.inputs["B"])
                tree.links.new(mix.outputs["Result"], bsdf_input)
