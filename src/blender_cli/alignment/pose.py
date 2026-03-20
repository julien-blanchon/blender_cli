"""Silhouette/CMA-ES pose estimation core."""

from __future__ import annotations

import io
import json
import logging
import os
import platform
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict, cast

if TYPE_CHECKING:
    from collections.abc import Callable

import cma
import cv2
import numpy as np
import numpy.typing as npt
import trimesh
from PIL import Image
from scipy import ndimage

from ._fal import subscribe as fal_subscribe
from ._fal import upload_image as fal_upload_image

# ── Platform setup ──────────────────────────────────────────────────
if platform.system() == "Linux" and "DISPLAY" not in os.environ:
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pyrender  # must come after EGL setup

log = logging.getLogger("blender_cli.alignment.pose")

# ── Types ───────────────────────────────────────────────────────────

# Mesh loaded from GLB — trimesh returns either
Mesh = trimesh.Scene | trimesh.Trimesh


class CameraParams(TypedDict):
    """JSON structure for camera configuration files."""

    position: list[float]
    look_at: list[float]
    fov_h_deg: float
    resolution: list[int]


class _TemplateHit(NamedTuple):
    yaw: float
    scale_factor: float
    scale: float
    iou: float


class _PoseCandidate(NamedTuple):
    yaw: float
    scale: float
    position: npt.NDArray[np.float64]
    score: float


# ── Coordinate conventions ──────────────────────────────────────────

M_SCENE_TO_GLB: npt.NDArray[np.float64] = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
M_GLB_TO_SCENE: npt.NDArray[np.float64] = np.linalg.inv(M_SCENE_TO_GLB)

# ── Private CMA-ES defaults ────────────────────────────────────────

_COARSE_SIGMA0 = 0.5
_COARSE_STDS = [30.0, 0.3, 0.15, 0.15, 0.05]

_FINE_SIGMA0 = 0.2
_FINE_STDS = [10.0, 0.1, 0.05, 0.05, 0.02]

# ── Dataclasses ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlignmentConfig:
    """Tuning knobs for the silhouette-alignment pipeline."""

    n_yaw: int = 36
    scale_range: tuple[float, ...] = (0.5, 0.75, 1.0, 1.3, 2.0)
    coarse_res: tuple[int, int] = (320, 180)
    fine_res: tuple[int, int] = (640, 360)
    coarse_restarts: int = 5
    fine_restarts: int = 2
    coarse_evals: int = 300
    fine_evals: int = 200
    w_iou: float = 0.7
    w_edge: float = 0.3
    seg_threshold: float = 30.0


@dataclass
class PoseResult:
    """Return type for :func:`estimate_pose`."""

    position: npt.NDArray[np.float64]
    yaw_deg: float
    scale: float
    iou: float
    pose_glb: npt.NDArray[np.float64]
    rendered_image: npt.NDArray[np.uint8]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, object]:
        """JSON-serializable representation (excludes rendered_image)."""
        return {
            "position": self.position.tolist(),
            "yaw_deg": float(self.yaw_deg),
            "scale": float(self.scale),
            "iou": float(self.iou),
            "pose_glb": self.pose_glb.tolist(),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


# ── Camera utilities ────────────────────────────────────────────────


def _load_camera(path: Path) -> CameraParams:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _camera_intrinsics(cam: CameraParams, w: int, h: int) -> npt.NDArray[np.float64]:
    fov_h_rad = np.radians(cam["fov_h_deg"])
    fx = w / (2.0 * np.tan(fov_h_rad / 2.0))
    return np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1]], dtype=np.float64)


def _build_camera_pose_opengl(
    pos: npt.NDArray[np.float64],
    look_at: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    eye = pos.astype(np.float64)
    target = look_at.astype(np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12
    up_world = np.array([0, 1, 0], dtype=np.float64)
    right = np.cross(fwd, up_world)
    if np.linalg.norm(right) < 1e-6:
        up_world = np.array([0, 0, 1], dtype=np.float64)
        right = np.cross(fwd, up_world)
    right /= np.linalg.norm(right) + 1e-12
    true_up = np.cross(right, fwd)
    T = np.eye(4, dtype=np.float64)
    T[:3, 0] = right
    T[:3, 1] = true_up
    T[:3, 2] = -fwd
    T[:3, 3] = eye
    return T


def _pixel_to_ray_scene(
    u: float,
    v: float,
    cam: CameraParams,
    w: int,
    h: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    K = _camera_intrinsics(cam, w, h)
    K_inv = np.linalg.inv(K)
    p_h = np.array([u, v, 1.0], dtype=np.float64)
    d_cam = K_inv @ p_h
    d_cam /= np.linalg.norm(d_cam)
    pos = np.array(cam["position"], dtype=np.float64)
    target = np.array(cam["look_at"], dtype=np.float64)
    fwd = target - pos
    fwd /= np.linalg.norm(fwd) + 1e-12
    up_scene = np.array([0.0, 0.0, 1.0])
    right = np.cross(fwd, up_scene)
    if np.linalg.norm(right) < 1e-6:
        up_scene = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, up_scene)
    right /= np.linalg.norm(right) + 1e-12
    down = np.cross(fwd, right)
    down /= np.linalg.norm(down) + 1e-12
    R_wc = np.column_stack([right, down, fwd])
    d_world = R_wc @ d_cam
    return pos.copy(), d_world


# ── Segmentation ────────────────────────────────────────────────────


def _cleanup_binary_mask(mask_u8: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = 1 + np.argmax(areas)
        mask = (labels == largest).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return cast("npt.NDArray[np.uint8]", mask)


def _segment_by_difference(
    query: npt.NDArray[np.uint8],
    reference: npt.NDArray[np.uint8],
    threshold: float = 30.0,
) -> npt.NDArray[np.bool_]:
    if query.shape != reference.shape:
        reference = np.array(
            Image.fromarray(reference).resize(
                (query.shape[1], query.shape[0]), Image.Resampling.LANCZOS
            )
        )
    diff = np.linalg.norm(
        query.astype(np.float32) - reference.astype(np.float32), axis=-1
    )
    mask = _cleanup_binary_mask((diff > threshold).astype(np.uint8))
    total_px = mask.shape[0] * mask.shape[1]
    if mask.sum() > 0.4 * total_px and threshold < 220:
        return _segment_by_difference(query, reference, threshold=threshold + 20)
    return mask.astype(bool)


def _mask_from_url(
    url: str,
    target_w: int,
    target_h: int,
) -> npt.NDArray[np.bool_] | None:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - explicit external API call
        payload = resp.read()
    mask_img = Image.open(io.BytesIO(payload))
    if mask_img.size != (target_w, target_h):
        mask_img = mask_img.resize((target_w, target_h), Image.Resampling.NEAREST)

    # SAM often returns RGBA where alpha stores the binary segmentation.
    if "A" in mask_img.getbands():
        alpha = np.asarray(mask_img.getchannel("A"), dtype=np.uint8)
        if alpha.max() > alpha.min():
            mask = alpha > 0
            if mask.sum() > 0:
                return mask

    mask = np.asarray(mask_img.convert("L"), dtype=np.uint8) > 127
    if mask.sum() == 0:
        return None
    return mask


def _segment_with_sam(
    query: npt.NDArray[np.uint8],
    diff_mask: npt.NDArray[np.bool_],
    *,
    sam_model: str,
    sam_object_name: str,
    sam_max_masks: int,
) -> npt.NDArray[np.bool_] | None:
    if not os.environ.get("FAL_KEY", "").strip():
        log.warning("SAM segmentation skipped: FAL_KEY environment variable is not set")
        return None

    ys, xs = np.where(diff_mask)
    if len(xs) < 10:
        return None
    pad = 10
    x_min = max(0, int(xs.min()) - pad)
    y_min = max(0, int(ys.min()) - pad)
    x_max = min(query.shape[1] - 1, int(xs.max()) + pad)
    y_max = min(query.shape[0] - 1, int(ys.max()) + pad)

    arguments: dict[str, Any] = {
        "image_url": fal_upload_image(Image.fromarray(query), fmt="png"),
        "prompt": sam_object_name,
        "box_prompts": [
            {
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }
        ],
        "return_multiple_masks": True,
        "max_objects": int(max(1, sam_max_masks)),
        "output_format": "png",
    }

    try:
        result = fal_subscribe(sam_model, arguments=arguments, with_logs=False)
    except Exception as exc:  # pragma: no cover - network/runtime variability
        log.warning("SAM segmentation failed (%s), falling back to diff mask", exc)
        return None

    candidates: list[dict[str, Any] | str] = []
    masks_value = result.get("masks")
    if isinstance(masks_value, list):
        candidates.extend(masks_value)
    single_mask = result.get("mask")
    if isinstance(single_mask, (dict, str)):
        candidates.append(single_mask)

    if not candidates:
        return None

    best_mask: npt.NDArray[np.bool_] | None = None
    best_score = -1.0
    best_intersection = -1.0
    diff_area = float(diff_mask.sum()) + 1e-6
    for item in candidates:
        url = item.get("url") if isinstance(item, dict) else str(item)
        if not isinstance(url, str) or not url:
            continue
        mask = _mask_from_url(url, target_w=query.shape[1], target_h=query.shape[0])
        if mask is None:
            continue
        inter = float((mask & diff_mask).sum())
        if inter <= 0:
            continue
        precision = inter / float(mask.sum() + 1e-6)
        recall = inter / diff_area
        score = (2.0 * precision * recall) / (precision + recall + 1e-6)
        if score > best_score or (
            abs(score - best_score) < 1e-9 and inter > best_intersection
        ):
            best_score = score
            best_intersection = inter
            best_mask = mask
    return best_mask


def _merge_diff_and_sam_masks(
    diff_mask: npt.NDArray[np.bool_],
    sam_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.bool_]:
    # Keep full SAM components that intersect the diff mask.
    # This removes pre-existing scene objects while preserving SAM shape.
    sam_u8 = sam_mask.astype(np.uint8)
    n_labels, labels, _stats, _ = cv2.connectedComponentsWithStats(
        sam_u8, connectivity=8
    )
    if n_labels <= 1:
        return sam_mask if (sam_mask & diff_mask).any() else diff_mask

    kept = np.zeros_like(sam_u8, dtype=np.uint8)
    for label in range(1, n_labels):
        comp = labels == label
        if (comp & diff_mask).any():
            kept[comp] = 1

    if kept.sum() == 0:
        return diff_mask
    return kept.astype(bool)


def _segment_object(
    query: npt.NDArray[np.uint8],
    reference: npt.NDArray[np.uint8],
    threshold: float = 30.0,
    *,
    use_sam: bool = True,
    sam_model: str = "fal-ai/sam-3/image",
    sam_object_name: str = "object",
    sam_max_masks: int = 5,
) -> tuple[npt.NDArray[np.bool_], dict[str, npt.NDArray[np.bool_]]]:
    diff_mask = _segment_by_difference(query, reference, threshold=threshold)
    debug_masks: dict[str, npt.NDArray[np.bool_]] = {"diff_mask": diff_mask}
    if not use_sam:
        log.info("Segmentation: diff mask only (%d px)", int(diff_mask.sum()))
        return diff_mask, debug_masks

    sam_mask = _segment_with_sam(
        query,
        diff_mask,
        sam_model=sam_model,
        sam_object_name=sam_object_name,
        sam_max_masks=sam_max_masks,
    )
    if sam_mask is None:
        log.info(
            "Segmentation: SAM unavailable/failed, using diff mask (%d px)",
            int(diff_mask.sum()),
        )
        return diff_mask, debug_masks

    debug_masks["sam_mask"] = sam_mask
    merged = _merge_diff_and_sam_masks(diff_mask, sam_mask)
    merged_u8 = _cleanup_binary_mask(merged.astype(np.uint8))
    final_mask = merged_u8.astype(bool)
    debug_masks["merged_mask"] = final_mask
    log.info(
        "Segmentation: diff=%d px, sam=%d px, merged=%d px",
        int(diff_mask.sum()),
        int(sam_mask.sum()),
        int(final_mask.sum()),
    )
    return final_mask, debug_masks


def _mask_bottom_center(mask: npt.NDArray[np.bool_]) -> tuple[float, float]:
    ys, xs = np.where(mask)
    y_bottom = int(ys.max())
    mask_h = ys.max() - ys.min()
    bottom_band = max(3, int(mask_h * 0.05))
    bottom_xs = xs[ys >= (y_bottom - bottom_band)]
    return float(bottom_xs.mean()), float(y_bottom)


def _mask_bbox(mask: npt.NDArray[np.bool_]) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


# ── Geometry: raycast, object extents, pose construction ────────────


def _raycast_scene(
    origin_scene: npt.NDArray[np.float64],
    direction_scene: npt.NDArray[np.float64],
    scene_concat_glb: trimesh.Trimesh,
) -> npt.NDArray[np.float64] | None:
    origin_glb = M_SCENE_TO_GLB[:3, :3] @ origin_scene + M_SCENE_TO_GLB[:3, 3]
    dir_glb = M_SCENE_TO_GLB[:3, :3] @ direction_scene
    locations, _, _ = scene_concat_glb.ray.intersects_location(
        ray_origins=[origin_glb], ray_directions=[dir_glb]
    )
    if len(locations) == 0:
        return None
    dists = np.linalg.norm(locations - origin_glb, axis=1)
    closest = locations[np.argmin(dists)]
    return M_GLB_TO_SCENE[:3, :3] @ closest + M_GLB_TO_SCENE[:3, 3]


def _get_object_extents_scene(obj: Mesh) -> npt.NDArray[np.float64]:
    ext = obj.extents
    return np.array([ext[0], ext[2], ext[1]])


def _get_object_bottom_glb_y(obj: Mesh) -> float:
    bounds = obj.bounds
    return float(bounds[0, 1])


def _get_mesh_vertices_glb(mesh: Mesh) -> npt.NDArray[np.float64]:
    """Return all mesh vertices in GLB coordinates."""
    if isinstance(mesh, trimesh.Scene):
        chunks: list[npt.NDArray[np.float64]] = []
        for node_name in mesh.graph.nodes_geometry:
            transform, geom_name = mesh.graph[node_name]
            geom = mesh.geometry[geom_name]
            verts = np.asarray(geom.vertices, dtype=np.float64)
            if verts.size == 0:
                continue
            ones = np.ones((verts.shape[0], 1), dtype=np.float64)
            verts_h = np.hstack([verts, ones])
            transformed = (np.asarray(transform, dtype=np.float64) @ verts_h.T).T[:, :3]
            chunks.append(transformed)
        if chunks:
            return np.vstack(chunks)
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(mesh.vertices, dtype=np.float64)


def _make_pose_glb(
    position_scene: npt.NDArray[np.float64],
    yaw_deg: float,
    scale: float,
) -> npt.NDArray[np.float64]:
    """Build 4x4 GLB transform from scene-coords position, yaw, scale."""
    yaw_rad = np.radians(yaw_deg)
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
    R_scene = np.array(
        [
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    T_scene = np.eye(4, dtype=np.float64)
    T_scene[:3, :3] = R_scene * scale
    T_scene[:3, 3] = position_scene
    return M_SCENE_TO_GLB @ T_scene @ M_GLB_TO_SCENE


def _sample_bottom_points_scene(
    object_vertices_glb: npt.NDArray[np.float64],
    pose_glb: npt.NDArray[np.float64],
    *,
    max_points: int = 96,
) -> npt.NDArray[np.float64]:
    """Sample transformed object bottom points in scene coordinates."""
    if object_vertices_glb.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    ys = object_vertices_glb[:, 1]
    y_min = float(ys.min())
    y_span = float(max(ys.max() - y_min, 1e-6))
    bottom_band = max(0.002, y_span * 0.03)
    bottom_idx = np.where(ys <= (y_min + bottom_band))[0]
    if bottom_idx.size == 0:
        bottom_idx = np.argsort(ys)[: min(max_points, len(ys))]

    if bottom_idx.size > max_points:
        pick = np.linspace(0, bottom_idx.size - 1, num=max_points, dtype=int)
        bottom_idx = bottom_idx[pick]

    pts_glb = object_vertices_glb[bottom_idx]
    pts_h = np.hstack([pts_glb, np.ones((pts_glb.shape[0], 1), dtype=np.float64)])
    pts_world_glb = (pose_glb @ pts_h.T).T[:, :3]
    pts_scene = (M_GLB_TO_SCENE[:3, :3] @ pts_world_glb.T).T + M_GLB_TO_SCENE[:3, 3]
    return pts_scene


def _resolve_vertical_collision(
    candidate: _PoseCandidate,
    *,
    scene_concat_glb: trimesh.Trimesh,
    object_vertices_glb: npt.NDArray[np.float64],
    clearance_m: float = 0.002,
    max_lift_m: float = 0.25,
    sample_points: int = 96,
) -> tuple[_PoseCandidate, float]:
    """
    Lift object minimally along +Z if bottom points penetrate the scene.

    This is a pragmatic collision guard without FCL dependency:
    for sampled bottom points, raycast scene directly downward and lift just
    enough to enforce a small vertical clearance.
    """
    pose_glb = _make_pose_glb(candidate.position, candidate.yaw, candidate.scale)
    bottom_pts_scene = _sample_bottom_points_scene(
        object_vertices_glb,
        pose_glb,
        max_points=sample_points,
    )
    if bottom_pts_scene.size == 0:
        return candidate, 0.0

    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_origin_offset = np.array([0.0, 0.0, 0.2], dtype=np.float64)

    required_lift = 0.0
    for p in bottom_pts_scene:
        hit = _raycast_scene(p + ray_origin_offset, ray_dir, scene_concat_glb)
        if hit is None:
            continue
        lift_here = float(hit[2] + clearance_m - p[2])
        required_lift = max(required_lift, lift_here)

    if required_lift <= 0.0:
        return candidate, 0.0

    applied_lift = min(required_lift, max_lift_m)
    lifted_pos = candidate.position.copy()
    lifted_pos[2] += applied_lift
    lifted = _PoseCandidate(
        yaw=candidate.yaw,
        scale=candidate.scale,
        position=lifted_pos,
        score=candidate.score,
    )
    return lifted, float(applied_lift)


def _collect_bottom_support_gaps(
    candidate: _PoseCandidate,
    *,
    scene_concat_glb: trimesh.Trimesh,
    object_vertices_glb: npt.NDArray[np.float64],
    sample_points: int = 96,
) -> npt.NDArray[np.float64]:
    """Return bottom-point vertical gaps to nearest supporting surface."""
    pose_glb = _make_pose_glb(candidate.position, candidate.yaw, candidate.scale)
    bottom_pts_scene = _sample_bottom_points_scene(
        object_vertices_glb,
        pose_glb,
        max_points=sample_points,
    )
    if bottom_pts_scene.size == 0:
        return np.empty((0,), dtype=np.float64)

    ray_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ray_origin_offset = np.array([0.0, 0.0, 0.2], dtype=np.float64)

    gaps: list[float] = []
    for p in bottom_pts_scene:
        hit = _raycast_scene(p + ray_origin_offset, ray_dir, scene_concat_glb)
        if hit is None:
            continue
        gap = float(p[2] - hit[2])
        if np.isfinite(gap):
            gaps.append(gap)
    if not gaps:
        return np.empty((0,), dtype=np.float64)
    return np.asarray(gaps, dtype=np.float64)


def _enforce_surface_contact(
    candidate: _PoseCandidate,
    *,
    scene_concat_glb: trimesh.Trimesh,
    object_vertices_glb: npt.NDArray[np.float64],
    clearance_m: float = 0.002,
    float_tolerance_m: float = 0.008,
    max_adjust_m: float = 0.25,
    sample_points: int = 96,
) -> tuple[_PoseCandidate, float, float]:
    """
    Enforce strong support prior: contact with a surface, but no collision.

    Returns (adjusted_candidate, applied_lift_m, applied_drop_m).
    """
    adjusted, lift = _resolve_vertical_collision(
        candidate,
        scene_concat_glb=scene_concat_glb,
        object_vertices_glb=object_vertices_glb,
        clearance_m=clearance_m,
        max_lift_m=max_adjust_m,
        sample_points=sample_points,
    )

    gaps = _collect_bottom_support_gaps(
        adjusted,
        scene_concat_glb=scene_concat_glb,
        object_vertices_glb=object_vertices_glb,
        sample_points=sample_points,
    )
    if gaps.size < 8:
        return adjusted, lift, 0.0

    # Use low percentile for robust "closest support" estimate.
    support_gap = float(np.percentile(gaps, 10.0))
    if support_gap <= clearance_m + float_tolerance_m:
        return adjusted, lift, 0.0

    requested_drop = min(support_gap - clearance_m, max_adjust_m)
    dropped_pos = adjusted.position.copy()
    dropped_pos[2] -= requested_drop
    dropped = _PoseCandidate(
        yaw=adjusted.yaw,
        scale=adjusted.scale,
        position=dropped_pos,
        score=adjusted.score,
    )

    # Re-apply collision guard after drop.
    no_collision, relift = _resolve_vertical_collision(
        dropped,
        scene_concat_glb=scene_concat_glb,
        object_vertices_glb=object_vertices_glb,
        clearance_m=clearance_m,
        max_lift_m=max_adjust_m,
        sample_points=sample_points,
    )
    total_lift = float(lift + relift)
    effective_drop = float(max(requested_drop - relift, 0.0))
    return no_collision, total_lift, effective_drop


# ── Future refinement hook ──────────────────────────────────────────


def _experimental_pose_refinement_stub(
    candidate: _PoseCandidate,
    *,
    query_image: npt.NDArray[np.uint8],
    target_mask: npt.NDArray[np.bool_],
    scene_mesh: Mesh,
    object_mesh: Mesh,
    camera: CameraParams,
) -> _PoseCandidate:
    """
    Placeholder for future post-alignment refinement (currently no-op).

    Potential ideas for a future experimental stage:
    - Local full-resolution optimization on (x, y, yaw, scale) with robust RGB loss
      constrained near the silhouette solution.
    - Edge tangent alignment around high-curvature silhouette regions (e.g. handles,
      legs) to stabilize yaw.
    - Depth-consistency penalty using monocular depth and scene depth reprojection.
    - Learnable pose prior conditioned on object class and support surface type.
    """
    _ = (query_image, target_mask, scene_mesh, object_mesh, camera)
    return candidate


# ── PoseRenderer ────────────────────────────────────────────────────


class _PoseRenderer:
    """Renders object at various poses efficiently by reusing the scene."""

    def __init__(
        self,
        scene_mesh: Mesh,
        object_mesh: Mesh,
        cam_config: CameraParams,
        width: int,
        height: int,
    ) -> None:
        self.width = width
        self.height = height

        pos_scene = np.array(cam_config["position"], dtype=np.float64)
        look_scene = np.array(cam_config["look_at"], dtype=np.float64)
        pos_glb = M_SCENE_TO_GLB[:3, :3] @ pos_scene + M_SCENE_TO_GLB[:3, 3]
        tgt_glb = M_SCENE_TO_GLB[:3, :3] @ look_scene + M_SCENE_TO_GLB[:3, 3]
        K = _camera_intrinsics(cam_config, width, height)
        cam_pose = _build_camera_pose_opengl(pos_glb, tgt_glb)

        self.pr_scene = pyrender.Scene(
            ambient_light=[0.3, 0.3, 0.3],
            bg_color=[0.85, 0.88, 0.92, 1.0],
        )

        if isinstance(scene_mesh, trimesh.Scene):
            for node_name in scene_mesh.graph.nodes_geometry:
                transform, geom_name = scene_mesh.graph[node_name]
                geom = scene_mesh.geometry[geom_name]
                try:
                    self.pr_scene.add(
                        pyrender.Mesh.from_trimesh(geom, smooth=False), pose=transform
                    )
                except Exception:  # noqa: BLE001 — pyrender raises generic errors for unsupported geometry
                    log.debug("Skipped scene geometry node %s", node_name)

        camera = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            znear=0.05,
            zfar=50.0,
        )
        self.pr_scene.add(camera, pose=cam_pose)
        self.pr_scene.add(
            pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.0), pose=cam_pose
        )
        top_glb = M_SCENE_TO_GLB[:3, :3] @ np.array([2, 4, 2.7])
        top_tgt = M_SCENE_TO_GLB[:3, :3] @ np.array([3, 4, 0.75])
        self.pr_scene.add(
            pyrender.DirectionalLight(color=[0.8, 0.8, 0.9], intensity=2.0),
            pose=_build_camera_pose_opengl(top_glb, top_tgt),
        )

        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.base_color: npt.NDArray[np.uint8]
        render_result = self.renderer.render(self.pr_scene)
        assert render_result is not None
        self.base_color = cast("npt.NDArray[np.uint8]", render_result[0])

        self.object_nodes: list[tuple[pyrender.Node, npt.NDArray[np.float64]]] = []
        self._add_object_mesh(object_mesh)

    def _add_object_mesh(self, object_mesh: Mesh) -> None:
        if isinstance(object_mesh, trimesh.Scene):
            for node_name in object_mesh.graph.nodes_geometry:
                transform, geom_name = object_mesh.graph[node_name]
                geom = object_mesh.geometry[geom_name]
                try:
                    pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
                    node = self.pr_scene.add(pr_mesh, pose=transform)
                    self.object_nodes.append((node, transform))
                except Exception:  # noqa: BLE001
                    log.debug("Skipped object geometry node %s", node_name)
        else:
            pr_mesh = pyrender.Mesh.from_trimesh(object_mesh, smooth=False)
            node = self.pr_scene.add(pr_mesh, pose=np.eye(4))
            self.object_nodes.append((node, np.eye(4)))

    def set_object_pose(self, pose_glb: npt.NDArray[np.float64]) -> None:
        for node, local_transform in self.object_nodes:
            node.matrix = (pose_glb @ local_transform).astype(np.float32)

    def render(self) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
        result = self.renderer.render(self.pr_scene)
        assert result is not None
        return cast("npt.NDArray[np.uint8]", result[0]), cast(
            "npt.NDArray[np.float32]", result[1]
        )

    def get_object_silhouette(
        self,
        color: npt.NDArray[np.uint8] | None = None,
    ) -> npt.NDArray[np.bool_]:
        if color is None:
            color, _ = self.render()
        diff = np.linalg.norm(
            color.astype(np.float32) - self.base_color.astype(np.float32),
            axis=-1,
        )
        return diff > 10

    def close(self) -> None:
        self.renderer.delete()


# ── Metrics ─────────────────────────────────────────────────────────


def _compute_iou(
    mask_a: npt.NDArray[np.bool_],
    mask_b: npt.NDArray[np.bool_],
) -> float:
    if mask_a.shape != mask_b.shape:
        mask_b = cv2.resize(
            mask_b.astype(np.uint8),
            (mask_a.shape[1], mask_a.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    inter = (mask_a & mask_b).sum()
    union = (mask_a | mask_b).sum()
    return float(inter / max(union, 1))


def _extract_contour(
    mask: npt.NDArray[np.bool_], dilate_px: int = 1
) -> npt.NDArray[np.bool_]:
    mask_u8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(mask_u8, kernel, iterations=dilate_px)
    eroded = cv2.erode(mask_u8, kernel, iterations=dilate_px)
    return (dilated - eroded).astype(bool)


def _compute_edge_score(
    target_mask: npt.NDArray[np.bool_],
    rendered_sil: npt.NDArray[np.bool_],
    target_dt: npt.NDArray[np.float64] | None = None,
) -> float:
    rendered_contour = _extract_contour(rendered_sil)
    target_contour = _extract_contour(target_mask)

    if rendered_contour.sum() < 5 or target_contour.sum() < 5:
        return 0.0

    if target_dt is None:
        target_dt = cast(
            "npt.NDArray[np.float64]", ndimage.distance_transform_edt(~target_contour)
        )
    rendered_dt = cast(
        "npt.NDArray[np.float64]", ndimage.distance_transform_edt(~rendered_contour)
    )

    fwd_dist = target_dt[rendered_contour].mean()
    bwd_dist = rendered_dt[target_contour].mean()
    mean_chamfer = (fwd_dist + bwd_dist) / 2.0

    diag = np.sqrt(target_mask.shape[0] ** 2 + target_mask.shape[1] ** 2)
    normalized = mean_chamfer / (diag * 0.1)
    return float(1.0 / (1.0 + normalized))


# ── Debug visualization ─────────────────────────────────────────────


def _make_debug_panel(
    query: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    rendered: npt.NDArray[np.uint8],
    silhouette: npt.NDArray[np.bool_],
    output_path: Path,
) -> None:
    h, w = query.shape[:2]
    target_w = 960
    scale = target_w / w
    sh, sw = int(h * scale), target_w

    def resize(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        if img.ndim == 2:
            return cast(
                "npt.NDArray[np.uint8]",
                cv2.resize(img, (sw, sh), interpolation=cv2.INTER_NEAREST),
            )
        return cast(
            "npt.NDArray[np.uint8]",
            cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR),
        )

    q = resize(query)
    r = resize(rendered)
    m = resize(mask.astype(np.uint8) * np.uint8(255))
    s = resize(silhouette.astype(np.uint8) * np.uint8(255))

    overlay = q.copy().astype(np.float32) * 0.5
    overlay[:, :, 1] += m.astype(np.float32) * 0.5
    overlay[:, :, 0] += s.astype(np.float32) * 0.5
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    top = np.hstack([q, r])
    bot = np.hstack([overlay, np.stack([s, s, s], axis=-1)])
    panel = np.vstack([top, bot])
    Image.fromarray(panel).save(str(output_path))


# ── CMA-ES objective ───────────────────────────────────────────────


def _make_combined_objective(
    renderer: _PoseRenderer,
    target_mask: npt.NDArray[np.bool_],
    init_position: npt.NDArray[np.float64],
    init_scale: float,
    obj_bottom_glb_y: float,
    hit_z: float,
    *,
    w_iou: float = 0.7,
    w_edge: float = 0.3,
) -> Callable[[list[float]], float]:
    target_contour = _extract_contour(target_mask)
    target_dt = cast(
        "npt.NDArray[np.float64]", ndimage.distance_transform_edt(~target_contour)
    )

    def objective(params: list[float]) -> float:
        yaw_deg = params[0]
        scale = init_scale * np.exp(params[1])
        dx, dy, dz = params[2], params[3], params[4]
        position = init_position + np.array([dx, dy, 0.0])
        position[2] = hit_z - obj_bottom_glb_y * scale + dz

        pose_glb = _make_pose_glb(position, yaw_deg, scale)
        renderer.set_object_pose(pose_glb)
        color, _ = renderer.render()
        silhouette = renderer.get_object_silhouette(color)

        iou = _compute_iou(target_mask, silhouette)
        edge = _compute_edge_score(target_mask, silhouette, target_dt)
        return -(w_iou * iou + w_edge * edge)

    return objective


# ── CMA-ES helpers ──────────────────────────────────────────────────


def _run_cma(
    objective: Callable[[list[float]], float],
    x0: list[float],
    sigma0: float,
    stds: list[float],
    max_evals: int,
    tolx: float,
    tolfun: float,
) -> cma.CMAEvolutionStrategy:
    opts = cma.CMAOptions()
    opts.set({
        "maxfevals": max_evals,
        "tolx": tolx,
        "tolfun": tolfun,
        "verbose": -9,
        "CMA_stds": stds,
    })
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
    while not es.stop():
        solutions = es.ask()
        es.tell(solutions, [objective(s) for s in solutions])
    return es


def _decode_cma_result(
    es: cma.CMAEvolutionStrategy,
    base_position: npt.NDArray[np.float64],
    base_scale: float,
    obj_bottom_glb_y: float,
    hit_z: float,
) -> _PoseCandidate:
    best_x = cast("npt.NDArray[np.float64]", es.result.xbest)
    yaw = float(best_x[0])
    scale = float(base_scale * np.exp(best_x[1]))
    pos = base_position + np.array([best_x[2], best_x[3], 0.0])
    pos[2] = hit_z - obj_bottom_glb_y * scale + best_x[4]
    return _PoseCandidate(
        yaw=yaw, scale=scale, position=pos, score=float(-es.result.fbest)
    )


# ── Diverse start selection ─────────────────────────────────────────


def _select_diverse_starts(
    hits: list[_TemplateHit],
    max_starts: int,
    *,
    min_yaw_gap: float = 30.0,
    min_sf_gap: float = 0.2,
) -> list[_TemplateHit]:
    selected = [hits[0]]
    for h in hits[1:]:
        if all(
            min(abs(h.yaw - s.yaw), 360 - abs(h.yaw - s.yaw)) >= min_yaw_gap
            or abs(h.scale_factor - s.scale_factor) >= min_sf_gap
            for s in selected
        ):
            selected.append(h)
            if len(selected) >= max_starts:
                break
    return selected


# ── Main pipeline ───────────────────────────────────────────────────


def estimate_pose(
    scene: str | Path,
    obj: str | Path,
    reference: str | Path,
    query: str | Path,
    camera: str | Path,
    *,
    output_dir: str | Path | None = None,
    debug: bool = False,
    use_depth: bool = False,
    use_sam: bool = True,
    sam_model: str = "fal-ai/sam-3/image",
    sam_object_name: str = "object",
    sam_max_masks: int = 5,
    config: AlignmentConfig | None = None,
) -> PoseResult:
    """
    Estimate the 6-DoF pose of an object in a scene via silhouette alignment.

    Args:
        scene: Path to scene GLB file.
        obj: Path to object GLB file.
        reference: Path to reference image (scene without object).
        query: Path to query image (scene with object).
        camera: Path to camera JSON configuration.
        output_dir: If set, save rendered image + result JSON here.
        debug: If ``True`` (and *output_dir* set), save debug images.
        use_depth: If ``True``, use Depth Anything V2 for initialization.
        use_sam: If ``True``, refine segmentation with Fal SAM-3.
        sam_model: Fal SAM model id.
        sam_object_name: Object name prompt for SAM segmentation.
        sam_max_masks: Max number of SAM mask candidates.
        config: Advanced tuning parameters.

    Returns:
        PoseResult with position, yaw, scale, IoU, pose matrix, rendered image.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If segmentation mask is too small.
        RuntimeError: If raycast fails to hit the scene.

    """
    t_start = time.time()
    cfg = config or AlignmentConfig()

    # Validate inputs
    paths = {
        "scene": Path(scene),
        "obj": Path(obj),
        "reference": Path(reference),
        "query": Path(query),
        "camera": Path(camera),
    }
    for label, p in paths.items():
        if not p.exists():
            msg = f"{label}: {p}"
            raise FileNotFoundError(msg)

    out = Path(output_dir) if output_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load ───────────────────────────────────────────
    t0 = time.time()
    cam = _load_camera(paths["camera"])
    full_w, full_h = cam["resolution"]
    ref_img = np.array(Image.open(paths["reference"]))[:, :, :3]
    query_img = np.array(Image.open(paths["query"]))[:, :, :3]
    scene_mesh = cast("Mesh", trimesh.load(str(paths["scene"])))
    object_mesh = cast("Mesh", trimesh.load(str(paths["obj"])))

    scene_concat: trimesh.Trimesh
    if isinstance(scene_mesh, trimesh.Scene):
        scene_concat = cast("trimesh.Trimesh", scene_mesh.to_geometry())
    else:
        scene_concat = cast("trimesh.Trimesh", scene_mesh)

    obj_extents = _get_object_extents_scene(object_mesh)
    obj_bottom_glb_y = _get_object_bottom_glb_y(object_mesh)
    obj_vertices_glb = _get_mesh_vertices_glb(object_mesh)
    log.info(
        "Load: %.1fs | extents=%s bottom_y=%.4f",
        time.time() - t0,
        obj_extents,
        obj_bottom_glb_y,
    )

    # ── Stage 2: Segment ────────────────────────────────────────
    t0 = time.time()
    mask, seg_debug = _segment_object(
        query_img,
        ref_img,
        threshold=cfg.seg_threshold,
        use_sam=use_sam,
        sam_model=sam_model,
        sam_object_name=sam_object_name,
        sam_max_masks=sam_max_masks,
    )
    if mask.sum() < 100:
        msg = f"Segmentation mask too small ({mask.sum()} px)"
        raise ValueError(msg)

    bx, by = _mask_bottom_center(mask)
    x0, y0, x1, y1 = _mask_bbox(mask)
    mask_w, mask_h = x1 - x0, y1 - y0
    if debug and out is not None:
        Image.fromarray((mask * 255).astype(np.uint8)).save(str(out / "debug_mask.png"))
        if "diff_mask" in seg_debug:
            Image.fromarray((seg_debug["diff_mask"] * 255).astype(np.uint8)).save(
                str(out / "debug_mask_diff.png")
            )
        if "sam_mask" in seg_debug:
            Image.fromarray((seg_debug["sam_mask"] * 255).astype(np.uint8)).save(
                str(out / "debug_mask_sam.png")
            )
        if "merged_mask" in seg_debug:
            Image.fromarray((seg_debug["merged_mask"] * 255).astype(np.uint8)).save(
                str(out / "debug_mask_merged.png")
            )
    log.info(
        "Segment: %.1fs | %d px, bbox=%dx%d",
        time.time() - t0,
        mask.sum(),
        mask_w,
        mask_h,
    )

    # ── Stage 3: Initialize ─────────────────────────────────────
    t0 = time.time()
    origin_b, dir_b = _pixel_to_ray_scene(bx, by, cam, full_w, full_h)
    hit_bottom = _raycast_scene(origin_b, dir_b, scene_concat)
    if hit_bottom is None:
        cx_px, cy_px = float((x0 + x1) / 2), float((y0 + y1) / 2)
        origin_c, dir_c = _pixel_to_ray_scene(cx_px, cy_px, cam, full_w, full_h)
        hit_bottom = _raycast_scene(origin_c, dir_c, scene_concat)
        if hit_bottom is None:
            msg = "Raycast failed to hit the scene geometry"
            raise RuntimeError(msg)

    K = _camera_intrinsics(cam, full_w, full_h)
    fx: float = float(K[0, 0])
    cam_pos = np.array(cam["position"], dtype=np.float64)
    distance = float(np.linalg.norm(hit_bottom - cam_pos))
    native_max = float(max(obj_extents[0], obj_extents[2]))
    mask_max_px = max(mask_w, mask_h)
    real_size = mask_max_px * distance / fx
    init_scale = float(np.clip(real_size / max(native_max, 0.01), 0.01, 5.0))

    hit_z = float(hit_bottom[2])
    init_position = hit_bottom.copy()
    init_position[2] = hit_z - obj_bottom_glb_y * init_scale

    if use_depth:
        from .depth import (  # noqa: PLC0415
            DepthEstimator,
            align_depth,
            depth_to_position,
            estimate_object_depth,
            render_scene_depth,
        )

        depth_estimator = DepthEstimator()
        scene_depth = render_scene_depth(scene_mesh, cam, full_w, full_h)
        mono_depth = depth_estimator.estimate(query_img)
        aligned_depth, _a, _b = align_depth(scene_depth, mono_depth, mask)
        object_depth = estimate_object_depth(aligned_depth, mask)

        cx_px = float((x0 + x1) / 2)
        cy_px = float((y0 + y1) / 2)
        depth_position = depth_to_position(
            cx_px, cy_px, object_depth, cam, full_w, full_h
        )

        depth_real_size = mask_max_px * object_depth / fx
        depth_scale = float(np.clip(depth_real_size / max(native_max, 0.01), 0.01, 5.0))

        init_position = depth_position.copy()
        init_position[2] = hit_z - obj_bottom_glb_y * depth_scale
        init_scale = depth_scale

        log.info(
            "Init (depth): %.1fs | scale=%.4f depth=%.3fm",
            time.time() - t0,
            init_scale,
            object_depth,
        )
    else:
        log.info(
            "Init (raycast): %.1fs | scale=%.4f dist=%.3fm",
            time.time() - t0,
            init_scale,
            distance,
        )

    # ── Stage 4: Template Search ────────────────────────────────
    t0 = time.time()
    sw_c, sh_c = cfg.coarse_res
    mask_coarse = cv2.resize(
        mask.astype(np.uint8), (sw_c, sh_c), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    cam_coarse: CameraParams = {**cam, "resolution": [sw_c, sh_c]}

    renderer_coarse = _PoseRenderer(scene_mesh, object_mesh, cam_coarse, sw_c, sh_c)

    yaw_candidates = np.linspace(0, 360, cfg.n_yaw + 1)[:-1]
    template_hits: list[_TemplateHit] = []

    for sf in cfg.scale_range:
        scale = init_scale * sf
        pos = init_position.copy()
        pos[2] = hit_z - obj_bottom_glb_y * scale
        for yaw in yaw_candidates:
            pose_glb = _make_pose_glb(pos, float(yaw), scale)
            renderer_coarse.set_object_pose(pose_glb)
            color, _ = renderer_coarse.render()
            sil = renderer_coarse.get_object_silhouette(color)
            iou = _compute_iou(mask_coarse, sil)
            template_hits.append(
                _TemplateHit(
                    yaw=float(yaw),
                    scale_factor=float(sf),
                    scale=float(scale),
                    iou=iou,
                )
            )

    template_hits.sort(key=lambda h: h.iou, reverse=True)
    best_hit = template_hits[0]
    log.info(
        "Template search: %.1fs | %d candidates, best IoU=%.4f (yaw=%.1f, sf=%.2f)",
        time.time() - t0,
        len(template_hits),
        best_hit.iou,
        best_hit.yaw,
        best_hit.scale_factor,
    )

    # ── Stage 5: Coarse CMA-ES ──────────────────────────────────
    t0 = time.time()
    selected = _select_diverse_starts(template_hits, cfg.coarse_restarts)
    coarse_candidates: list[_PoseCandidate] = []

    for i, start in enumerate(selected):
        scale_start = init_scale * start.scale_factor
        pos_start = init_position.copy()
        pos_start[2] = hit_z - obj_bottom_glb_y * scale_start

        objective = _make_combined_objective(
            renderer_coarse,
            mask_coarse,
            pos_start,
            scale_start,
            obj_bottom_glb_y,
            hit_z,
            w_iou=cfg.w_iou,
            w_edge=cfg.w_edge,
        )

        es = _run_cma(
            objective,
            [start.yaw, 0.0, 0.0, 0.0, 0.0],
            sigma0=_COARSE_SIGMA0,
            stds=_COARSE_STDS,
            max_evals=cfg.coarse_evals,
            tolx=1e-3,
            tolfun=1e-4,
        )
        cand = _decode_cma_result(es, pos_start, scale_start, obj_bottom_glb_y, hit_z)
        coarse_candidates.append(cand)

        log.info(
            "  Coarse #%d (yaw=%.1f, sf=%.2f) -> score=%.4f  yaw=%.1f  scale=%.4f",
            i + 1,
            start.yaw,
            start.scale_factor,
            cand.score,
            cand.yaw,
            cand.scale,
        )

    renderer_coarse.close()
    log.info("Coarse CMA-ES: %.1fs | %d starts", time.time() - t0, len(selected))

    # ── Stage 6: Fine CMA-ES ────────────────────────────────────
    t0 = time.time()
    sw_f, sh_f = cfg.fine_res
    mask_fine = cv2.resize(
        mask.astype(np.uint8), (sw_f, sh_f), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    cam_fine: CameraParams = {**cam, "resolution": [sw_f, sh_f]}

    renderer_fine = _PoseRenderer(scene_mesh, object_mesh, cam_fine, sw_f, sh_f)

    coarse_candidates.sort(key=lambda c: c.score, reverse=True)
    top_coarse = coarse_candidates[: cfg.fine_restarts]

    fine_candidates: list[_PoseCandidate] = []

    for i, cand in enumerate(top_coarse):
        objective_fine = _make_combined_objective(
            renderer_fine,
            mask_fine,
            cand.position,
            cand.scale,
            obj_bottom_glb_y,
            hit_z,
            w_iou=cfg.w_iou,
            w_edge=cfg.w_edge,
        )

        es = _run_cma(
            objective_fine,
            [cand.yaw, 0.0, 0.0, 0.0, 0.0],
            sigma0=_FINE_SIGMA0,
            stds=_FINE_STDS,
            max_evals=cfg.fine_evals,
            tolx=1e-4,
            tolfun=1e-5,
        )
        fine = _decode_cma_result(
            es, cand.position, cand.scale, obj_bottom_glb_y, hit_z
        )
        fine_candidates.append(fine)

        log.info(
            "  Fine #%d: score=%.4f  yaw=%.1f  scale=%.4f",
            i + 1,
            fine.score,
            fine.yaw,
            fine.scale,
        )

    renderer_fine.close()
    log.info(
        "Fine CMA-ES: %.1fs | %d candidates", time.time() - t0, len(fine_candidates)
    )

    # ── Stage 7: Multi-hypothesis Verification (full res) ───────
    t0 = time.time()
    full_renderer = _PoseRenderer(scene_mesh, object_mesh, cam, full_w, full_h)

    best_full_iou = -1.0
    best_cand = fine_candidates[0]
    best_color_full: npt.NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
    best_sil_full: npt.NDArray[np.bool_] = np.empty(0, dtype=bool)

    for j, cand in enumerate(fine_candidates):
        adjusted_cand, lift_m, drop_m = _enforce_surface_contact(
            cand,
            scene_concat_glb=scene_concat,
            object_vertices_glb=obj_vertices_glb,
        )

        pose_glb = _make_pose_glb(
            adjusted_cand.position, adjusted_cand.yaw, adjusted_cand.scale
        )
        full_renderer.set_object_pose(pose_glb)
        color, _ = full_renderer.render()
        sil = full_renderer.get_object_silhouette(color)
        full_iou = _compute_iou(mask, sil)

        tag = ""
        if full_iou > best_full_iou:
            best_full_iou = full_iou
            best_cand = adjusted_cand
            best_color_full = color.copy()
            best_sil_full = sil.copy()
            tag = " <- BEST"

        log.info(
            "  Verify #%d: full_IoU=%.4f  yaw=%.1f  scale=%.4f  lift=%.4fm drop=%.4fm%s",
            j + 1,
            full_iou,
            adjusted_cand.yaw,
            adjusted_cand.scale,
            lift_m,
            drop_m,
            tag,
        )

    best_cand = _experimental_pose_refinement_stub(
        best_cand,
        query_image=query_img,
        target_mask=mask,
        scene_mesh=scene_mesh,
        object_mesh=object_mesh,
        camera=cam,
    )
    pose_glb_best = _make_pose_glb(best_cand.position, best_cand.yaw, best_cand.scale)
    full_renderer.set_object_pose(pose_glb_best)
    best_color_full, _ = full_renderer.render()
    best_sil_full = full_renderer.get_object_silhouette(best_color_full)
    best_full_iou = _compute_iou(mask, best_sil_full)
    full_renderer.close()
    log.info("Verification: %.1fs", time.time() - t0)

    # ── Stage 8: Build result ───────────────────────────────────
    pose_glb = _make_pose_glb(best_cand.position, best_cand.yaw, best_cand.scale)

    if out is not None:
        Image.fromarray(best_color_full).save(str(out / "query_reconstructed.png"))

        if debug:
            _make_debug_panel(
                query_img, mask, best_color_full, best_sil_full, out / "debug_panel.png"
            )

        combined = (
            scene_mesh.copy()
            if isinstance(scene_mesh, trimesh.Scene)
            else trimesh.Scene(scene_mesh)
        )
        if isinstance(object_mesh, trimesh.Scene):
            for node_name in object_mesh.graph.nodes_geometry:
                transform, geom_name = object_mesh.graph[node_name]
                geom = object_mesh.geometry[geom_name]
                combined.add_geometry(geom, transform=pose_glb @ transform)
        else:
            combined.add_geometry(object_mesh, transform=pose_glb)
        combined.export(str(out / "scene_with_object.glb"))

    elapsed = time.time() - t_start
    result = PoseResult(
        position=best_cand.position,
        yaw_deg=best_cand.yaw,
        scale=best_cand.scale,
        iou=best_full_iou,
        pose_glb=pose_glb,
        rendered_image=best_color_full,
        elapsed_seconds=elapsed,
    )

    if out is not None:
        with (out / "result.json").open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

    log.info(
        "DONE: IoU=%.4f  yaw=%.1f  scale=%.4f  (%.1fs total)",
        best_full_iou,
        best_cand.yaw,
        best_cand.scale,
        elapsed,
    )

    return result
