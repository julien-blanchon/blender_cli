"""Mask — binary/density selection map (0..1) with morphological operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np
import numpy.typing as npt

from blender_cli.geometry.field2d import CombineOp, Field2D

if TYPE_CHECKING:
    from pathlib import Path

    from blender_cli.geometry.heightfield import Heightfield
    from blender_cli.geometry.spline import Spline

HeightfieldMode = Literal["slope", "height", "curvature"]


class Mask:
    """
    Immutable 0..1 selection/density map with morphological operations.

    All operations return new Mask instances — the source is never mutated.
    Internally wraps a Field2D whose values are clamped to 0..1.
    """

    __slots__ = ("_field",)

    def __init__(self, field: Field2D) -> None:
        # Clamp to 0..1 on construction
        data = np.clip(field.to_numpy(), 0.0, 1.0)
        self._field = Field2D(data, field.meters_per_px)

    # -- properties --

    @property
    def width(self) -> int:
        return self._field.width

    @property
    def height(self) -> int:
        return self._field.height

    @property
    def meters_per_px(self) -> float:
        return self._field.meters_per_px

    @property
    def shape(self) -> tuple[int, int]:
        return self._field.shape

    # -- factories --

    @staticmethod
    def from_image(
        path: str | Path,
        channel: str = "r",
        remap: tuple[float, float] = (0.0, 1.0),
        meters_per_px: float = 1.0,
    ) -> Mask:
        """Load a mask from an image file."""
        field = Field2D.from_image(
            path, channel=channel, remap=remap, meters_per_px=meters_per_px
        )
        return Mask(field)

    @staticmethod
    def from_heightfield(hf: Heightfield, mode: HeightfieldMode = "slope") -> Mask:
        """
        Derive a mask from terrain properties.

        Modes:
          - 'slope': gradient magnitude (steep = high)
          - 'height': normalized elevation (high = high)
          - 'curvature': second-derivative magnitude (curved = high)
        """
        data = hf.to_field().to_numpy()
        mpp = hf.meters_per_px

        if mode == "height":
            result = data.copy()
        elif mode == "slope":
            # Gradient magnitude using finite differences (Sobel)
            dx = cv2.Sobel(data, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * mpp)
            dy = cv2.Sobel(data, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * mpp)
            result = np.sqrt(dx * dx + dy * dy)
        elif mode == "curvature":
            # Laplacian (second derivative magnitude)
            lap = cv2.Laplacian(data, cv2.CV_32F, ksize=3) / (mpp * mpp)
            result = np.abs(lap)
        else:
            msg = f"Unknown mode '{mode}', expected: slope, height, curvature"
            raise ValueError(msg)

        # Normalize to 0..1
        lo, hi = float(result.min()), float(result.max())
        result = np.zeros_like(result) if hi - lo < 1e-12 else (result - lo) / (hi - lo)

        return Mask(Field2D(result.astype(np.float32), mpp))

    @staticmethod
    def from_spline(
        spline: Spline,
        width: float,
        falloff: float = 0.0,
        reference: Heightfield | Field2D | None = None,
        resolution: tuple[int, int] = (256, 256),
        meters_per_px: float = 1.0,
    ) -> Mask:
        """
        Generate a mask from a spline corridor.

        1.0 inside the corridor (*width*/2 from centre), smooth falloff at edges.

        If *reference* is given (a :class:`Heightfield` or :class:`Field2D`),
        the output grid matches its shape and ``meters_per_px`` exactly —
        *resolution* and *meters_per_px* are ignored.

        Uses ``cv2.distanceTransform`` for fast O(w×h) evaluation.
        """
        # Resolve grid parameters from reference or explicit args
        if reference is not None:
            from blender_cli.geometry.heightfield import Heightfield as HF

            ref_field = reference._field if isinstance(reference, HF) else reference
            h_px, w_px = ref_field.shape
            mpp = ref_field.meters_per_px
        else:
            w_px, h_px = resolution
            mpp = meters_per_px

        half_w = width / 2.0
        outer = half_w + falloff

        # Dense spline samples → pixel coordinates
        n_samples = max(256, int(spline.length() / (mpp * 0.5)))
        pts_px = np.empty((n_samples + 1, 1, 2), dtype=np.int32)
        for i in range(n_samples + 1):
            t = i / n_samples
            pt = spline.sample(t)
            pts_px[i, 0, 0] = round(pt.x / mpp)  # col
            pts_px[i, 0, 1] = round(pt.y / mpp)  # row

        # Draw centre-line on binary image, then distance transform
        binary = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.polylines(binary, [pts_px], isClosed=False, color=(255,), thickness=1)
        # distanceTransform needs 0=foreground, so invert: line pixels → 0
        inv = 255 - binary
        dist_map = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist_map = dist_map.astype(np.float32) * mpp  # pixels → metres

        # Build mask from distance values
        data = np.zeros((h_px, w_px), dtype=np.float32)
        inside = dist_map <= half_w
        data[inside] = 1.0

        if falloff > 0:
            band = (dist_map > half_w) & (dist_map <= outer)
            t = (dist_map[band] - half_w) / falloff
            t = np.clip(t, 0.0, 1.0)
            data[band] = 1.0 - t * t * (3.0 - 2.0 * t)  # smoothstep

        return Mask(Field2D(data, mpp))

    # -- sampling --

    def sample(self, x: float, y: float) -> float:
        """Interpolated mask value at world coordinates (x, y)."""
        return self._field.sample(x, y)

    # -- operations (all return new Mask) --

    def blur(self, radius: int) -> Mask:
        """Gaussian blur with given pixel radius."""
        return Mask(self._field.blur(radius))

    def threshold(self, t: float) -> Mask:
        """Hard threshold: values >= t become 1, else 0."""
        data = (self._field.to_numpy() >= t).astype(np.float32)
        return Mask(Field2D(data, self.meters_per_px))

    def combine(self, other: Mask, op: CombineOp = "mul") -> Mask:
        """Element-wise combination with another Mask."""
        return Mask(self._field.combine(other._field, op))

    def invert(self) -> Mask:
        """Invert: 1 - value."""
        data = 1.0 - self._field.to_numpy()
        return Mask(Field2D(data, self.meters_per_px))

    def dilate(self, radius: int) -> Mask:
        """Morphological dilation with circular kernel."""
        if radius <= 0:
            return Mask(self._field)
        ksize = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        data = cv2.dilate(self._field.to_numpy(), kernel)
        return Mask(Field2D(np.asarray(data, dtype=np.float32), self.meters_per_px))

    def erode(self, radius: int) -> Mask:
        """Morphological erosion with circular kernel."""
        if radius <= 0:
            return Mask(self._field)
        ksize = radius * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        data = cv2.erode(self._field.to_numpy(), kernel)
        return Mask(Field2D(np.asarray(data, dtype=np.float32), self.meters_per_px))

    def open(self, radius: int) -> Mask:
        """Morphological opening (erode then dilate). Removes small bright spots."""
        return self.erode(radius).dilate(radius)

    def close(self, radius: int) -> Mask:
        """Morphological closing (dilate then erode). Fills small dark gaps."""
        return self.dilate(radius).erode(radius)

    # -- conversions --

    def to_field(self) -> Field2D:
        """Convert to Field2D."""
        return Field2D(self._field.to_numpy(), self.meters_per_px)

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Return a copy of the underlying array (float32, 0..1)."""
        return self._field.to_numpy()

    def save_debug(self, path: str | Path) -> Path:
        """Save as grayscale PNG for visual inspection."""
        return self._field.save_debug(path)
