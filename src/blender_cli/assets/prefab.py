"""Prefab asset wrapper — imports GLB files via Blender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy

from blender_cli.scene.entity import Entity


@dataclass(frozen=True, slots=True)
class Prefab:
    """
    Immutable reference to a GLB prefab file on disk.

    Validates that the file exists on construction.
    Call :meth:`load` to import the GLB into a fresh Blender collection.
    """

    path: Path

    def __post_init__(self) -> None:
        resolved = Path(self.path).resolve()
        if not resolved.is_file():
            msg = f"Prefab not found: {resolved}"
            raise FileNotFoundError(msg)
        if resolved.suffix.lower() not in {".glb", ".gltf"}:
            msg = f"Prefab must be a .glb or .gltf file: {resolved}"
            raise ValueError(msg)
        object.__setattr__(self, "path", resolved)

    def load(self, name: str | None = None) -> bpy.types.Collection:
        """
        Import the GLB into a new collection and return it.

        *name* defaults to the file stem (e.g. ``tree_pine``).
        """
        col_name = name or self.path.stem
        collection = bpy.data.collections.new(col_name)
        if bpy.context.scene is None:
            msg = "Scene is not set"
            raise ValueError(msg)
        if bpy.context.scene.collection is None:
            msg = "Scene collection is not set"
            raise ValueError(msg)
        bpy.context.scene.collection.children.link(collection)

        # Remember what objects exist before import.
        before = set(bpy.data.objects)

        bpy.ops.import_scene.gltf(filepath=str(self.path))

        # Move newly-imported objects into our collection.
        for obj in set(bpy.data.objects) - before:
            # Unlink from any existing collections first.
            for col in list(obj.users_collection):
                col.objects.unlink(obj)
            collection.objects.link(obj)

        return collection

    def spawn(self, name: str | None = None) -> Entity:
        """Import and return a fluent entity for the first prefab object."""
        col = self.load(name)
        objects = list(col.objects)
        if not objects:
            msg = f"Prefab {self.path} imported with no objects"
            raise RuntimeError(msg)
        ent = Entity(objects[0]).asset(self.path.name, self.path)
        return ent
