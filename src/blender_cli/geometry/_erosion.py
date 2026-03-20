"""Erosion algorithms — pure numpy, no Blender dependency."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def hydraulic_erosion(
    data: npt.NDArray[np.float32],
    meters_per_px: float,
    iterations: int = 50,
    rain_rate: float = 0.01,
    sediment_capacity: float = 0.05,
    evaporation: float = 0.02,
    seed: int = 0,
) -> npt.NDArray[np.float32]:
    """
    Simple grid-based hydraulic erosion.

    Water is added uniformly, flows downhill, picks up sediment, and deposits
    it when the water can carry no more.  Returns a new array.
    """
    rng = np.random.default_rng(seed)
    h, w = data.shape
    terrain = data.astype(np.float64, copy=True)
    water = np.zeros_like(terrain)
    sediment = np.zeros_like(terrain)

    for _ in range(iterations):
        # Rain
        water += rain_rate * (1.0 + 0.1 * rng.standard_normal((h, w)))
        water = np.maximum(water, 0.0)

        # Compute flow to lowest neighbour (4-connected)
        # Pad terrain+water to handle borders
        surface = terrain + water
        padded = np.pad(surface, 1, mode="edge")

        # Height differences to each neighbour (negative = downhill)
        diffs = np.zeros((4, h, w), dtype=np.float64)
        diffs[0] = surface - padded[:-2, 1:-1]  # up
        diffs[1] = surface - padded[2:, 1:-1]  # down
        diffs[2] = surface - padded[1:-1, :-2]  # left
        diffs[3] = surface - padded[1:-1, 2:]  # right

        # Only flow downhill
        diffs = np.maximum(diffs, 0.0)
        total_diff = diffs.sum(axis=0) + 1e-12

        # Amount of water to move
        flow = np.minimum(water, total_diff * 0.5)

        # Distribute proportionally
        for i, (dy, dx) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            fraction = diffs[i] / total_diff
            transfer = flow * fraction
            water -= transfer
            # Move water to neighbour via roll
            water += np.roll(np.roll(transfer, dy, axis=0), dx, axis=1)

        # Erosion: pick up sediment proportional to water speed (≈ slope)
        slope = total_diff / (meters_per_px * 4.0)
        capacity = water * slope * sediment_capacity
        erosion = np.minimum(capacity - sediment, terrain * 0.1)
        erosion = np.maximum(erosion, 0.0)
        terrain -= erosion
        sediment += erosion

        # Deposition: drop sediment when over capacity
        excess = sediment - capacity
        deposit = np.maximum(excess, 0.0) * 0.5
        terrain += deposit
        sediment -= deposit

        # Evaporation
        water *= 1.0 - evaporation

    return terrain.astype(np.float32)


def thermal_erosion(
    data: npt.NDArray[np.float32],
    meters_per_px: float,
    iterations: int = 50,
    talus_angle: float = 35.0,
) -> npt.NDArray[np.float32]:
    """
    Thermal weathering erosion.

    Material crumbles from steep slopes (above *talus_angle* degrees) and
    accumulates at the base.  Returns a new array.
    """
    h, w = data.shape
    terrain = data.astype(np.float64, copy=True)
    # Maximum stable height difference between adjacent cells
    max_diff = meters_per_px * np.tan(np.radians(talus_angle))

    for _ in range(iterations):
        padded = np.pad(terrain, 1, mode="edge")

        # Height differences to 4 neighbours
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dy, dx in offsets:
            neighbour = padded[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
            diff = terrain - neighbour
            excess = diff - max_diff
            move = np.maximum(excess, 0.0) * 0.25  # move quarter of excess
            terrain -= move
            # We can't directly add to neighbour via fancy indexing across
            # boundaries, so approximate by shifting the move array.
            terrain += np.roll(np.roll(move, dy, axis=0), dx, axis=1)

    return terrain.astype(np.float32)
