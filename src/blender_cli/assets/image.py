"""Image asset wrapper — loads image files via Blender."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import bpy


@dataclass(frozen=True, slots=True)
class Image:
    """
    Immutable reference to an image file on disk.

    Validates that the file exists on construction.
    Call :meth:`load` to load into ``bpy.data.images``.
    """

    path: Path

    def __post_init__(self) -> None:
        # Resolve and validate; store as absolute Path.
        resolved = Path(self.path).resolve()
        if not resolved.is_file():
            msg = f"Image not found: {resolved}"
            raise FileNotFoundError(msg)
        object.__setattr__(self, "path", resolved)

    def load(self) -> bpy.types.Image:
        """Load (or reuse) the image in Blender and return the datablock."""
        # Reuse if already loaded with the same absolute path.
        for img in bpy.data.images:
            if img.filepath and Path(img.filepath).resolve() == self.path:
                return img
        return bpy.data.images.load(str(self.path))
