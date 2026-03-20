"""Field2D — generic 2D float grid for heightmaps, density maps, etc."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import Callable

CombineOp = Literal["add", "mul", "max", "min", "sub"]
GradientDirection = Literal["x", "y", "radial"]

_CHANNEL_INDEX = {"r": 2, "g": 1, "b": 0, "a": 3}  # BGR order in cv2


class Field2D:
    """
    Immutable 2D float grid with world-space sampling.

    All operations return new Field2D instances — the source is never mutated.
    """

    __slots__ = ("_data", "_mpp")

    def __init__(
        self, data: npt.NDArray[np.floating[Any]], meters_per_px: float = 1.0
    ) -> None:
        if data.ndim != 2:
            msg = f"Expected 2D array, got shape {data.shape}"
            raise ValueError(msg)
        if meters_per_px <= 0:
            msg = "meters_per_px must be positive"
            raise ValueError(msg)
        self._data: npt.NDArray[np.float32] = data.astype(np.float32, copy=True)
        self._mpp = float(meters_per_px)

    @property
    def width(self) -> int:
        return int(self._data.shape[1])

    @property
    def height(self) -> int:
        return int(self._data.shape[0])

    @property
    def meters_per_px(self) -> float:
        return self._mpp

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    # -- world-space metrics --

    @property
    def world_width(self) -> float:
        """Width of the field in world units (metres)."""
        return self.width * self._mpp

    @property
    def world_height(self) -> float:
        """Height of the field in world units (metres)."""
        return self.height * self._mpp

    @property
    def world_extent(self) -> tuple[float, float]:
        """``(world_width, world_height)`` in metres."""
        return (self.world_width, self.world_height)

    @property
    def world_area(self) -> float:
        """Ground area in square metres."""
        return self.world_width * self.world_height

    # -- factories --

    @staticmethod
    def from_numpy(
        arr: npt.NDArray[np.floating[Any]], meters_per_px: float = 1.0
    ) -> Field2D:
        return Field2D(arr, meters_per_px)

    @staticmethod
    def zeros(width: int, height: int, meters_per_px: float = 1.0) -> Field2D:
        return Field2D(np.zeros((height, width), dtype=np.float32), meters_per_px)

    @staticmethod
    def ones(width: int, height: int, meters_per_px: float = 1.0) -> Field2D:
        return Field2D(np.ones((height, width), dtype=np.float32), meters_per_px)

    @staticmethod
    def constant(
        width: int, height: int, value: float, meters_per_px: float = 1.0
    ) -> Field2D:
        """Create a uniform-value field."""
        return Field2D(np.full((height, width), value, dtype=np.float32), meters_per_px)

    @staticmethod
    def gradient(
        width: int,
        height: int,
        direction: GradientDirection = "x",
        meters_per_px: float = 1.0,
    ) -> Field2D:
        """
        Directional gradient from 0 to 1.

        Directions:
          - ``"x"``: left-to-right
          - ``"y"``: top-to-bottom
          - ``"radial"``: centre-to-edge (0 at centre, 1 at corners)
        """
        if direction == "x":
            row = np.linspace(0.0, 1.0, width, dtype=np.float32)
            data = np.tile(row, (height, 1))
        elif direction == "y":
            col = np.linspace(0.0, 1.0, height, dtype=np.float32)
            data = np.tile(col[:, np.newaxis], (1, width))
        elif direction == "radial":
            cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            max_dist = np.sqrt(cx**2 + cy**2)
            data = dist / max_dist if max_dist > 1e-12 else np.zeros_like(dist)
        else:
            msg = f"Unknown direction '{direction}', expected: x, y, radial"
            raise ValueError(msg)
        return Field2D(data.astype(np.float32), meters_per_px)

    @staticmethod
    def from_image(
        path: str | Path,
        channel: str = "r",
        remap: tuple[float, float] = (0.0, 1.0),
        meters_per_px: float = 1.0,
    ) -> Field2D:
        """Load a single channel from an image file, remapped to *remap* range."""
        p = Path(path).resolve()
        if not p.is_file():
            msg = f"Image not found: {p}"
            raise FileNotFoundError(msg)
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            msg = f"Failed to decode image: {p}"
            raise ValueError(msg)

        if img.ndim == 2:
            raw = img.astype(np.float32)
        else:
            idx = _CHANNEL_INDEX.get(channel)
            if idx is None:
                msg = f"Unknown channel '{channel}', expected r/g/b/a"
                raise ValueError(msg)
            if idx >= img.shape[2]:
                msg = f"Image has {img.shape[2]} channels, cannot extract '{channel}'"
                raise ValueError(msg)
            raw = img[:, :, idx].astype(np.float32)

        maxval = 65535.0 if raw.max() > 255.0 else 255.0
        raw /= maxval
        lo, hi = remap
        raw = lo + raw * (hi - lo)
        return Field2D(raw, meters_per_px)

    # -- sampling --

    def sample(self, x: float, y: float) -> float:
        """Bilinear-interpolated value at world coordinates (x, y)."""
        px = x / self._mpp
        py = y / self._mpp
        px = max(0.0, min(px, self.width - 1.0))
        py = max(0.0, min(py, self.height - 1.0))
        x0, y0 = int(px), int(py)
        x1 = min(x0 + 1, self.width - 1)
        y1 = min(y0 + 1, self.height - 1)
        fx, fy = px - x0, py - y0
        v00 = float(self._data[y0, x0])
        v10 = float(self._data[y0, x1])
        v01 = float(self._data[y1, x0])
        v11 = float(self._data[y1, x1])
        return (
            v00 * (1 - fx) * (1 - fy)
            + v10 * fx * (1 - fy)
            + v01 * (1 - fx) * fy
            + v11 * fx * fy
        )

    # -- operations (all return new Field2D) --

    def normalize(self) -> Field2D:
        """Normalize values to 0..1 range."""
        lo, hi = float(self._data.min()), float(self._data.max())
        if hi - lo < 1e-12:
            return Field2D(np.zeros_like(self._data), self._mpp)
        return Field2D((self._data - lo) / (hi - lo), self._mpp)

    def blur(self, radius: int) -> Field2D:
        """Gaussian blur with given pixel radius."""
        if radius <= 0:
            return Field2D(self._data, self._mpp)
        ksize = radius * 2 + 1
        blurred = np.asarray(
            cv2.GaussianBlur(self._data, (ksize, ksize), 0), dtype=np.float32
        )
        return Field2D(blurred, self._mpp)

    def combine(self, other: Field2D, op: CombineOp = "add") -> Field2D:
        """Element-wise combination with another Field2D."""
        if self.shape != other.shape:
            msg = f"Shape mismatch: {self.shape} vs {other.shape}"
            raise ValueError(msg)
        ops: dict[str, npt.NDArray[np.float32]] = {
            "add": self._data + other._data,
            "sub": self._data - other._data,
            "mul": self._data * other._data,
            "max": np.maximum(self._data, other._data),
            "min": np.minimum(self._data, other._data),
        }
        result = ops.get(op)
        if result is None:
            msg = f"Unknown op '{op}', expected: {', '.join(ops)}"
            raise ValueError(msg)
        return Field2D(result, self._mpp)

    def remap(
        self,
        in_range: tuple[float, float],
        out_range: tuple[float, float],
        clamp: bool = True,
    ) -> Field2D:
        """Remap values from *in_range* to *out_range*."""
        ilo, ihi = in_range
        olo, ohi = out_range
        span = ihi - ilo
        if abs(span) < 1e-12:
            return Field2D(np.full_like(self._data, olo), self._mpp)
        t = (self._data - ilo) / span
        if clamp:
            t = np.clip(t, 0.0, 1.0)
        result = olo + t * (ohi - olo)
        return Field2D(result, self._mpp)

    def add_noise(
        self,
        type: Literal["fbm", "ridged"] = "fbm",
        amp: float = 1.0,
        freq: float = 0.1,
        seed: int = 0,
        octaves: int = 6,
    ) -> Field2D:
        """Add procedural noise (same algorithm as Heightfield.add_noise)."""
        from blender_cli.geometry.heightfield import generate_noise

        noise = generate_noise(self.width, self.height, type, amp, freq, seed, octaves)
        return Field2D(self._data + noise, self._mpp)

    def apply(
        self, fn: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]]
    ) -> Field2D:
        """
        Apply an arbitrary function to the underlying array.

        ``fn`` receives a copy of the data and must return an array of the same shape.
        """
        result = fn(self._data.copy())
        if result.shape != self._data.shape:
            msg = f"apply() result shape {result.shape} != original {self._data.shape}"
            raise ValueError(msg)
        return Field2D(result.astype(np.float32), self._mpp)

    # -- conversions --

    def to_numpy(
        self, dtype: npt.DTypeLike = np.float32
    ) -> npt.NDArray[np.floating[Any]]:
        """Return a copy of the internal array, cast to *dtype*."""
        return self._data.astype(dtype, copy=True)

    def to_cv2(self) -> npt.NDArray[np.floating[Any]]:
        """Return a copy suitable for cv2 (float32 HxW)."""
        return self._data.copy()

    def save_debug(self, path: str | Path) -> Path:
        """Save as grayscale PNG for visual inspection. Returns resolved path."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        lo, hi = float(self._data.min()), float(self._data.max())
        if hi - lo < 1e-12:
            out = np.zeros_like(self._data, dtype=np.uint8)
        else:
            out = ((self._data - lo) / (hi - lo) * 255).astype(np.uint8)
        cv2.imwrite(str(p), out)
        return p

    def to_mask(self, threshold: float = 0.5, soft: bool = False) -> Field2D:
        """Convert to a mask (0..1). Hard threshold or soft (clamp to 0..1)."""
        if soft:
            data = np.clip(self._data, 0.0, 1.0)
        else:
            data = (self._data >= threshold).astype(np.float32)
        return Field2D(data, self._mpp)
