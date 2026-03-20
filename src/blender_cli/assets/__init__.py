"""Asset wrappers — images, prefabs, materials, and registries."""

from blender_cli.assets.image import Image
from blender_cli.assets.material import Material
from blender_cli.assets.prefab import Prefab
from blender_cli.assets.registry import AssetRegistry, MaterialRegistry

__all__ = [
    "AssetRegistry",
    "Image",
    "Material",
    "MaterialRegistry",
    "Prefab",
]
