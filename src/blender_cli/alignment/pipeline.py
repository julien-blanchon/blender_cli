"""High-level alignment pipeline using silhouette/CMA-ES pose estimation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .generation import generate_alignment_assets
from .integration import compose_from_directory
from .pose import (
    M_GLB_TO_SCENE,
    AlignmentConfig,
    PoseResult,
    estimate_pose,
)
from .types import (
    AlignmentPipelineResult,
    ComposeOptions,
    ComposeResult,
    GenerationOptions,
    GenerationResult,
    PoseEstimationOptions,
    PoseEstimationResult,
)

logger = logging.getLogger(__name__)


def _require_paths(paths: list[tuple[Path, str]]) -> None:
    for path, label in paths:
        if not path.exists():
            msg = f"{label} not found: {path}"
            raise FileNotFoundError(msg)


def _resolve_sam_object_name(base_dir: Path, opts: PoseEstimationOptions) -> str:
    provenance_path = base_dir / "provenance.json"
    if provenance_path.exists():
        try:
            payload = json.loads(provenance_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        candidate = payload.get("object_name")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    # Fallback for estimate-only runs without provenance.
    return Path(opts.object_name).stem.strip() or "object"


def _build_payload(
    result: PoseResult,
    *,
    scale_method: str,
) -> dict[str, object]:
    # compose/integration expects a scene-space matrix under T_world_obj.
    t_world_obj = M_GLB_TO_SCENE @ result.pose_glb
    return {
        "success": True,
        "pipeline": "silhouette_cmaes",
        "position": result.position.tolist(),
        "yaw_deg": float(result.yaw_deg),
        "scale": float(result.scale),
        "iou": float(result.iou),
        "pose_glb": result.pose_glb.tolist(),
        "T_world_obj": t_world_obj.tolist(),
        "scale_method": scale_method,
        "elapsed_seconds": float(result.elapsed_seconds),
    }


def estimate_pose_from_directory(
    directory: str | Path,
    options: PoseEstimationOptions | None = None,
) -> PoseEstimationResult:
    """Estimate object pose from an example directory."""
    opts = options or PoseEstimationOptions()
    base_dir = Path(directory).resolve()
    if not base_dir.is_dir():
        msg = f"Directory not found: {base_dir}"
        raise FileNotFoundError(msg)

    mesh_path = base_dir / opts.object_name
    query_path = base_dir / opts.query_name
    reference_path = base_dir / opts.reference_name
    scene_path = base_dir / opts.scene_name
    camera_path = base_dir / opts.camera_name
    output_dir = base_dir / opts.output_name
    pose_result_path = output_dir / "pose_result.json"

    _require_paths([
        (mesh_path, "object mesh"),
        (query_path, "query image"),
        (reference_path, "reference image"),
        (scene_path, "scene mesh"),
        (camera_path, "camera config"),
    ])
    output_dir.mkdir(parents=True, exist_ok=True)

    if opts.scale_method == "raycast":
        use_depth = False
    elif opts.scale_method in {"depth", "marigold"}:
        use_depth = True
    else:
        msg = (
            f"Unsupported scale_method '{opts.scale_method}'. "
            "Use 'raycast', 'depth', or 'marigold'."
        )
        raise ValueError(msg)

    config = AlignmentConfig(
        n_yaw=opts.n_yaw,
        scale_range=tuple(opts.scale_range),
        coarse_res=(opts.coarse_res[0], opts.coarse_res[1]),
        fine_res=(opts.fine_res[0], opts.fine_res[1]),
        coarse_restarts=opts.coarse_restarts,
        fine_restarts=opts.fine_restarts,
        coarse_evals=opts.coarse_evals,
        fine_evals=opts.fine_evals,
        w_iou=opts.w_iou,
        w_edge=opts.w_edge,
        seg_threshold=float(opts.seg_threshold),
    )
    sam_object_name = _resolve_sam_object_name(base_dir, opts)

    try:
        core_result = estimate_pose(
            scene=scene_path,
            obj=mesh_path,
            reference=reference_path,
            query=query_path,
            camera=camera_path,
            output_dir=output_dir,
            debug=opts.debug,
            use_depth=use_depth,
            use_sam=opts.use_sam,
            sam_model=opts.sam_model,
            sam_object_name=sam_object_name,
            sam_max_masks=opts.sam_max_masks,
            config=config,
        )
    except (RuntimeError, ValueError) as exc:
        payload = {
            "success": False,
            "pipeline": "silhouette_cmaes",
            "reason": str(exc),
        }
        pose_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return PoseEstimationResult(
            directory=base_dir,
            output_dir=output_dir,
            pose_result_path=pose_result_path,
            success=False,
            result=payload,
            options=opts,
        )

    payload = _build_payload(core_result, scale_method=opts.scale_method)
    pose_result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return PoseEstimationResult(
        directory=base_dir,
        output_dir=output_dir,
        pose_result_path=pose_result_path,
        success=True,
        result=payload,
        options=opts,
    )


def run_alignment_pipeline(
    directory: str | Path,
    *,
    generation_options: GenerationOptions | None = None,
    pose_options: PoseEstimationOptions | None = None,
    compose: bool = True,
    compose_options: ComposeOptions | None = None,
) -> AlignmentPipelineResult:
    """Run generation (optional), pose estimation, then composition (optional)."""
    generation_result: GenerationResult | None = None
    if generation_options is not None:
        generation_result = generate_alignment_assets(directory, generation_options)

    estimation_result = estimate_pose_from_directory(directory, options=pose_options)

    composition_result: ComposeResult | None = None
    if compose and estimation_result.success:
        compose_opts = compose_options or ComposeOptions()
        try:
            composition_result = compose_from_directory(directory, options=compose_opts)
        except FileNotFoundError:
            logger.warning(
                "Skipping composition: scene/object/pose files were not all present"
            )

    return AlignmentPipelineResult(
        generation=generation_result,
        estimation=estimation_result,
        composition=composition_result,
    )
