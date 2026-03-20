"""Marigold depth helpers for silhouette/CMA-ES pose initialization."""

from __future__ import annotations

import io
import logging
import urllib.request

import cv2
import numpy as np
import numpy.typing as npt
import pyrender  # pyright: ignore[reportMissingImports]
import trimesh
from PIL import Image  # pyright: ignore[reportMissingImports]

from ._fal import require_fal_client
from ._fal import subscribe as fal_subscribe
from ._fal import upload_image as fal_upload_image
from .pose import (  # pyright: ignore[reportPrivateUsage]
    M_SCENE_TO_GLB,
    CameraParams,
    _build_camera_pose_opengl,
    _camera_intrinsics,
)

log = logging.getLogger("blender_cli.alignment.depth")

_FAL_MODEL_ID = "fal-ai/imageutils/marigold-depth"


class DepthEstimator:
    """Monocular depth estimation using Marigold via fal.ai."""

    def __init__(
        self,
        *,
        ensemble_size: int = 10,
        num_inference_steps: int = 10,
    ) -> None:
        require_fal_client("Depth initialization")
        self._ensemble_size = ensemble_size
        self._num_inference_steps = num_inference_steps
        log.info(
            "DepthEstimator ready (Marigold via fal.ai, ensemble=%d, steps=%d)",
            ensemble_size,
            num_inference_steps,
        )

    def estimate(self, image_np: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """
        Estimate relative depth from an RGB image.

        Returns a depth map (H, W) in [0, 1] float range where
        higher values = farther from camera.

        """
        h, w = image_np.shape[:2]

        pil_img = Image.fromarray(image_np)
        image_url = fal_upload_image(pil_img, fmt="png")
        log.info("Uploaded query image to fal CDN, calling Marigold...")

        result = fal_subscribe(
            _FAL_MODEL_ID,
            arguments={
                "image_url": image_url,
                "ensemble_size": self._ensemble_size,
                "num_inference_steps": self._num_inference_steps,
            },
            with_logs=True,
            logger=log,
            log_prefix="  fal: ",
        )

        # Download and decode the depth map PNG, preserving bit depth.
        # Marigold may return 8-bit or 16-bit PNG.
        depth_url: str = result["image"]["url"]
        with urllib.request.urlopen(depth_url) as resp:  # noqa: S310
            depth_bytes = resp.read()
        depth_img = Image.open(io.BytesIO(depth_bytes))

        # Convert to float preserving full dynamic range
        if depth_img.mode == "I;16":
            depth_np = np.asarray(depth_img, dtype=np.float32) / 65535.0
        elif depth_img.mode in {"I", "F"}:
            depth_np = np.asarray(depth_img, dtype=np.float32)
            dmax = depth_np.max()
            if dmax > 0:
                depth_np /= dmax
        else:
            # 8-bit (L or RGB)
            depth_np = np.asarray(depth_img.convert("L"), dtype=np.float32) / 255.0

        # Resize to original resolution
        resized: npt.NDArray[np.float32] = np.asarray(
            cv2.resize(depth_np, (w, h), interpolation=cv2.INTER_LINEAR),
            dtype=np.float32,
        )
        log.info(
            "Marigold depth map received (%dx%d, range=[%.3f, %.3f])",
            w,
            h,
            resized.min(),
            resized.max(),
        )
        return resized


def render_scene_depth(
    scene_mesh: trimesh.Scene | trimesh.Trimesh,
    cam_config: CameraParams,
    width: int,
    height: int,
) -> npt.NDArray[np.float32]:
    """Render depth buffer of the scene (without object) using pyrender."""
    pos_scene = np.array(cam_config["position"], dtype=np.float64)
    look_scene = np.array(cam_config["look_at"], dtype=np.float64)
    pos_glb = M_SCENE_TO_GLB[:3, :3] @ pos_scene + M_SCENE_TO_GLB[:3, 3]
    tgt_glb = M_SCENE_TO_GLB[:3, :3] @ look_scene + M_SCENE_TO_GLB[:3, 3]
    K = _camera_intrinsics(cam_config, width, height)
    cam_pose = _build_camera_pose_opengl(pos_glb, tgt_glb)

    pr_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    if isinstance(scene_mesh, trimesh.Scene):
        for node_name in scene_mesh.graph.nodes_geometry:
            transform, geom_name = scene_mesh.graph[node_name]
            geom = scene_mesh.geometry[geom_name]
            try:
                pr_scene.add(
                    pyrender.Mesh.from_trimesh(geom, smooth=False),
                    pose=transform,
                )
            except Exception:  # noqa: BLE001 - pyrender raises generic errors for unsupported geometry
                log.debug("Skipped scene depth geometry node %s", node_name)

    camera = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=0.05,
        zfar=50.0,
    )
    pr_scene.add(camera, pose=cam_pose)

    r = pyrender.OffscreenRenderer(width, height)
    result = r.render(pr_scene)
    assert result is not None
    _, depth = result
    r.delete()
    return depth


def align_depth(
    scene_depth: npt.NDArray[np.float32],
    mono_depth: npt.NDArray[np.float32],
    mask: npt.NDArray[np.bool_],
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Align monocular (relative) depth to absolute scene depth.

    Fits an affine model ``scene = a * mono + b`` on background pixels
    (where the rendered scene depth is valid and the object mask is off).
    Automatically tries both depth and inverse-depth conventions and
    picks the one with lower residual.

    """
    valid = (scene_depth > 0.1) & (~mask)
    if valid.sum() < 100:
        log.warning("Too few valid background pixels for depth alignment")
        return mono_depth.astype(np.float64), 1.0, 0.0

    sd = scene_depth[valid].astype(np.float64)
    md = mono_depth[valid].astype(np.float64)

    # Fit 1: direct linear  scene = a*mono + b
    A_lin = np.column_stack([md, np.ones_like(md)])
    res_lin = np.linalg.lstsq(A_lin, sd, rcond=None)
    a_lin, b_lin = res_lin[0]
    pred_lin = a_lin * md + b_lin
    err_lin = float(np.mean((pred_lin - sd) ** 2))

    # Fit 2: inverse  scene = a/mono + b  (handles disparity-like maps)
    md_inv = 1.0 / (md + 1e-6)
    A_inv = np.column_stack([md_inv, np.ones_like(md_inv)])
    res_inv = np.linalg.lstsq(A_inv, sd, rcond=None)
    a_inv, b_inv = res_inv[0]
    pred_inv = a_inv * md_inv + b_inv
    err_inv = float(np.mean((pred_inv - sd) ** 2))

    # Pick the better fit
    if a_lin > 0 and err_lin <= err_inv:
        aligned = a_lin * mono_depth.astype(np.float64) + b_lin
        log.info(
            "Depth alignment: linear (a=%.4f, b=%.4f, mse=%.6f)", a_lin, b_lin, err_lin
        )
        return np.clip(aligned, 0, 50), float(a_lin), float(b_lin)

    aligned = a_inv / (mono_depth.astype(np.float64) + 1e-6) + b_inv
    log.info(
        "Depth alignment: inverse (a=%.4f, b=%.4f, mse=%.6f)", a_inv, b_inv, err_inv
    )
    return np.clip(aligned, 0, 50), float(a_inv), float(b_inv)


def estimate_object_depth(
    aligned_depth: npt.NDArray[np.float64],
    mask: npt.NDArray[np.bool_],
) -> float:
    """Estimate the depth of the object using the aligned depth map."""
    object_depths = aligned_depth[mask]
    if len(object_depths) == 0:
        return 1.0
    return float(np.median(object_depths))


def depth_to_position(
    pixel_u: float,
    pixel_v: float,
    depth: float,
    cam_config: CameraParams,
    width: int,
    height: int,
) -> npt.NDArray[np.float64]:
    """Back-project a pixel at given depth to 3D scene coordinates."""
    K = _camera_intrinsics(cam_config, width, height)
    K_inv = np.linalg.inv(K)

    p_h = np.array([pixel_u, pixel_v, 1.0], dtype=np.float64)
    d_cam = K_inv @ p_h
    d_cam /= np.linalg.norm(d_cam)

    ray_distance = depth / abs(d_cam[2])

    pos = np.array(cam_config["position"], dtype=np.float64)
    target = np.array(cam_config["look_at"], dtype=np.float64)
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

    return pos + R_wc @ (d_cam * ray_distance)
