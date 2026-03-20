"""Alignment pipeline CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

import click

from blender_cli.alignment import (
    ComposeOptions,
    GenerationOptions,
    PoseEstimationOptions,
    compose_from_directory,
    estimate_pose_from_directory,
    generate_alignment_assets,
    run_alignment_pipeline,
)


def _parse_resolution(value: str) -> tuple[int, int]:
    raw = value.strip().lower().replace(" ", "")
    for sep in ("x", ","):
        if sep in raw:
            w_raw, h_raw = raw.split(sep, 1)
            break
    else:
        msg = f"Invalid resolution '{value}'. Use WIDTHxHEIGHT (example: 320x180)."
        raise click.BadParameter(msg)

    try:
        width = int(w_raw)
        height = int(h_raw)
    except ValueError as exc:
        msg = f"Invalid resolution '{value}'. Width/height must be integers."
        raise click.BadParameter(msg) from exc

    if width <= 0 or height <= 0:
        msg = f"Invalid resolution '{value}'. Width/height must be > 0."
        raise click.BadParameter(msg)
    return width, height


def _parse_scale_range(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        msg = f"Invalid --scale-range '{value}'. Use comma-separated floats."
        raise click.BadParameter(msg) from exc
    if not parsed:
        msg = "--scale-range must contain at least one value"
        raise click.BadParameter(msg)
    return parsed


def _missing_for_estimate(base_dir: Path, options: PoseEstimationOptions) -> list[str]:
    required = {
        "object": base_dir / options.object_name,
        "query": base_dir / options.query_name,
        "reference": base_dir / options.reference_name,
        "scene": base_dir / options.scene_name,
        "camera": base_dir / options.camera_name,
    }
    return [key for key, value in required.items() if not value.exists()]


@click.group()
def align() -> None:
    """Alignment pipeline: generate, estimate, and compose."""


@align.command("generate")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--object", "object_name", required=True, help="Object description prompt."
)
@click.option("--placement", required=True, help="Placement phrase in the scene.")
@click.option(
    "--reference", "reference_name", default="reference.png", show_default=True
)
@click.option("--query", "query_name", default="query.png", show_default=True)
@click.option(
    "--object-reference",
    "object_reference_name",
    default="object_reference.png",
    show_default=True,
)
@click.option(
    "--object-file", "object_name_on_disk", default="object.glb", show_default=True
)
@click.option("--skip-3d", is_flag=True, help="Skip image-to-3D conversion.")
@click.option(
    "--dry-run", is_flag=True, help="Validate options only, no network calls."
)
def align_generate(
    directory: str,
    object_name: str,
    placement: str,
    reference_name: str,
    query_name: str,
    object_reference_name: str,
    object_name_on_disk: str,
    skip_3d: bool,
    dry_run: bool,
) -> None:
    """Run AI generation steps in an example directory."""
    base_dir = Path(directory).resolve()
    options = GenerationOptions(
        object_name=object_name,
        placement=placement,
        reference_name=reference_name,
        query_name=query_name,
        object_reference_name=object_reference_name,
        object_name_on_disk=object_name_on_disk,
        skip_3d=skip_3d,
    )
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "action": "align_generate",
                    "dry_run": True,
                    "directory": str(base_dir),
                    "reference_exists": (base_dir / reference_name).exists(),
                    "options": options.to_dict(),
                },
                indent=2,
            )
        )
        return

    try:
        result = generate_alignment_assets(base_dir, options=options)
    except Exception as exc:  # pragma: no cover - runtime/dependency/network errors
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "align_generate",
                "result": result.to_dict(),
            },
            indent=2,
        )
    )


@align.command("estimate")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--object", "object_name", default="object.glb", show_default=True)
@click.option("--query", "query_name", default="query.png", show_default=True)
@click.option(
    "--reference", "reference_name", default="reference.png", show_default=True
)
@click.option("--scene", "scene_name", default="scene.glb", show_default=True)
@click.option("--camera", "camera_name", default="camera.json", show_default=True)
@click.option("--output", "output_name", default="output", show_default=True)
@click.option("--n-yaw", default=36, type=int, show_default=True)
@click.option(
    "--scale-range",
    default="0.5,0.75,1.0,1.3,2.0",
    show_default=True,
    help="Comma-separated relative scale factors.",
)
@click.option("--coarse-res", default="320x180", show_default=True)
@click.option("--fine-res", default="640x360", show_default=True)
@click.option("--coarse-restarts", default=5, type=int, show_default=True)
@click.option("--fine-restarts", default=2, type=int, show_default=True)
@click.option("--coarse-evals", default=300, type=int, show_default=True)
@click.option("--fine-evals", default=200, type=int, show_default=True)
@click.option("--w-iou", default=0.7, type=float, show_default=True)
@click.option("--w-edge", default=0.3, type=float, show_default=True)
@click.option("--seg-threshold", default=30.0, type=float, show_default=True)
@click.option(
    "--sam/--no-sam",
    "use_sam",
    default=True,
    show_default=True,
    help="Use Fal SAM-3 segmentation and intersect with diff mask.",
)
@click.option(
    "--sam-model",
    default="fal-ai/sam-3/image",
    show_default=True,
    help="Fal model id for SAM segmentation.",
)
@click.option("--sam-max-masks", default=5, type=int, show_default=True)
@click.option(
    "--scale-method",
    type=click.Choice(["raycast", "depth", "marigold"]),
    default="raycast",
    show_default=True,
)
@click.option("--debug", is_flag=True, help="Save debug artifacts.")
@click.option(
    "--dry-run", is_flag=True, help="Validate options and required files only."
)
def align_estimate(
    directory: str,
    object_name: str,
    query_name: str,
    reference_name: str,
    scene_name: str,
    camera_name: str,
    output_name: str,
    n_yaw: int,
    scale_range: str,
    coarse_res: str,
    fine_res: str,
    coarse_restarts: int,
    fine_restarts: int,
    coarse_evals: int,
    fine_evals: int,
    w_iou: float,
    w_edge: float,
    seg_threshold: float,
    use_sam: bool,
    sam_model: str,
    sam_max_masks: int,
    scale_method: str,
    debug: bool,
    dry_run: bool,
) -> None:
    """Estimate object pose from query/reference images."""
    base_dir = Path(directory).resolve()
    options = PoseEstimationOptions(
        object_name=object_name,
        query_name=query_name,
        reference_name=reference_name,
        scene_name=scene_name,
        camera_name=camera_name,
        output_name=output_name,
        n_yaw=n_yaw,
        scale_range=_parse_scale_range(scale_range),
        coarse_res=_parse_resolution(coarse_res),
        fine_res=_parse_resolution(fine_res),
        coarse_restarts=coarse_restarts,
        fine_restarts=fine_restarts,
        coarse_evals=coarse_evals,
        fine_evals=fine_evals,
        w_iou=w_iou,
        w_edge=w_edge,
        seg_threshold=seg_threshold,
        use_sam=use_sam,
        sam_model=sam_model,
        sam_max_masks=sam_max_masks,
        scale_method=scale_method,  # type: ignore[arg-type]
        debug=debug,
    )
    if dry_run:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "action": "align_estimate",
                    "dry_run": True,
                    "directory": str(base_dir),
                    "missing_required": _missing_for_estimate(base_dir, options),
                    "options": options.to_dict(),
                },
                indent=2,
            )
        )
        return

    try:
        result = estimate_pose_from_directory(base_dir, options=options)
    except Exception as exc:  # pragma: no cover - runtime/dependency/model errors
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": "ok" if result.success else "warning",
                "action": "align_estimate",
                "result": result.to_dict(),
            },
            indent=2,
        )
    )


@align.command("compose")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--scene", "scene_name", default="scene.glb", show_default=True)
@click.option("--object", "object_name", default="object.glb", show_default=True)
@click.option("--output", "output_dir_name", default="output", show_default=True)
@click.option(
    "--pose-result",
    "pose_result_name",
    default="pose_result.json",
    show_default=True,
)
@click.option("--out", "output_name", default="combined.glb", show_default=True)
@click.option("--dry-run", is_flag=True, help="Validate paths only.")
def align_compose(
    directory: str,
    scene_name: str,
    object_name: str,
    output_dir_name: str,
    pose_result_name: str,
    output_name: str,
    dry_run: bool,
) -> None:
    """Compose the object into the scene from a saved pose result."""
    base_dir = Path(directory).resolve()
    options = ComposeOptions(
        output_name=output_name,
        pose_result_name=pose_result_name,
        output_dir_name=output_dir_name,
        scene_name=scene_name,
        object_name=object_name,
    )
    output_dir = base_dir / options.output_dir_name
    scene_path = base_dir / options.scene_name
    object_path = base_dir / options.object_name
    pose_path = output_dir / options.pose_result_name
    out_path = output_dir / options.output_name

    if dry_run:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "action": "align_compose",
                    "dry_run": True,
                    "scene_exists": scene_path.exists(),
                    "object_exists": object_path.exists(),
                    "pose_result_exists": pose_path.exists(),
                    "output_path": str(out_path),
                },
                indent=2,
            )
        )
        return

    try:
        result = compose_from_directory(base_dir, options=options)
    except Exception as exc:  # pragma: no cover - runtime/dependency/path errors
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": "ok",
                "action": "align_compose",
                "result": result.to_dict(),
            },
            indent=2,
        )
    )


@align.command("run")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--generate/--no-generate",
    default=False,
    show_default=True,
    help="Include AI generation before pose estimation.",
)
@click.option(
    "--object", "object_name", default=None, help="Object prompt (for --generate)."
)
@click.option("--placement", default=None, help="Placement prompt (for --generate).")
@click.option("--mesh-file", "mesh_file", default="object.glb", show_default=True)
@click.option("--query-file", "query_file", default="query.png", show_default=True)
@click.option(
    "--reference-file", "reference_file", default="reference.png", show_default=True
)
@click.option("--scene-file", "scene_file", default="scene.glb", show_default=True)
@click.option("--camera-file", "camera_file", default="camera.json", show_default=True)
@click.option("--output-dir", "output_dir", default="output", show_default=True)
@click.option("--n-yaw", default=36, type=int, show_default=True)
@click.option("--scale-range", default="0.5,0.75,1.0,1.3,2.0", show_default=True)
@click.option("--coarse-res", default="320x180", show_default=True)
@click.option("--fine-res", default="640x360", show_default=True)
@click.option("--coarse-restarts", default=5, type=int, show_default=True)
@click.option("--fine-restarts", default=2, type=int, show_default=True)
@click.option("--coarse-evals", default=300, type=int, show_default=True)
@click.option("--fine-evals", default=200, type=int, show_default=True)
@click.option("--w-iou", default=0.7, type=float, show_default=True)
@click.option("--w-edge", default=0.3, type=float, show_default=True)
@click.option("--seg-threshold", default=30.0, type=float, show_default=True)
@click.option(
    "--sam/--no-sam",
    "use_sam",
    default=True,
    show_default=True,
    help="Use Fal SAM-3 segmentation and intersect with diff mask.",
)
@click.option(
    "--sam-model",
    default="fal-ai/sam-3/image",
    show_default=True,
    help="Fal model id for SAM segmentation.",
)
@click.option("--sam-max-masks", default=5, type=int, show_default=True)
@click.option(
    "--scale-method",
    type=click.Choice(["raycast", "depth", "marigold"]),
    default="raycast",
    show_default=True,
)
@click.option("--skip-3d", is_flag=True, help="Skip image-to-3D stage.")
@click.option("--debug", is_flag=True, help="Save debug artifacts during estimation.")
@click.option("--compose/--no-compose", default=True, show_default=True)
@click.option("--dry-run", is_flag=True, help="Only print execution plan.")
def align_run(
    directory: str,
    generate: bool,
    object_name: str | None,
    placement: str | None,
    mesh_file: str,
    query_file: str,
    reference_file: str,
    scene_file: str,
    camera_file: str,
    output_dir: str,
    n_yaw: int,
    scale_range: str,
    coarse_res: str,
    fine_res: str,
    coarse_restarts: int,
    fine_restarts: int,
    coarse_evals: int,
    fine_evals: int,
    w_iou: float,
    w_edge: float,
    seg_threshold: float,
    use_sam: bool,
    sam_model: str,
    sam_max_masks: int,
    scale_method: str,
    skip_3d: bool,
    debug: bool,
    compose: bool,
    dry_run: bool,
) -> None:
    """Run optional generation + pose estimation + composition in one command."""
    base_dir = Path(directory).resolve()
    generation_options = None
    if generate:
        if not object_name or not placement:
            msg = "--object and --placement are required when --generate is enabled"
            raise click.ClickException(msg)
        generation_options = GenerationOptions(
            object_name=object_name,
            placement=placement,
            skip_3d=skip_3d,
        )

    pose_options = PoseEstimationOptions(
        object_name=mesh_file,
        query_name=query_file,
        reference_name=reference_file,
        scene_name=scene_file,
        camera_name=camera_file,
        output_name=output_dir,
        n_yaw=n_yaw,
        scale_range=_parse_scale_range(scale_range),
        coarse_res=_parse_resolution(coarse_res),
        fine_res=_parse_resolution(fine_res),
        coarse_restarts=coarse_restarts,
        fine_restarts=fine_restarts,
        coarse_evals=coarse_evals,
        fine_evals=fine_evals,
        w_iou=w_iou,
        w_edge=w_edge,
        seg_threshold=seg_threshold,
        use_sam=use_sam,
        sam_model=sam_model,
        sam_max_masks=sam_max_masks,
        scale_method=scale_method,  # type: ignore[arg-type]
        debug=debug,
    )
    compose_options = ComposeOptions(
        output_dir_name=output_dir,
        scene_name=scene_file,
        object_name=mesh_file,
    )

    if dry_run:
        click.echo(
            json.dumps(
                {
                    "status": "ok",
                    "action": "align_run",
                    "dry_run": True,
                    "directory": str(base_dir),
                    "generate": generate,
                    "compose": compose,
                    "missing_required_for_estimate": _missing_for_estimate(
                        base_dir, pose_options
                    ),
                    "generation_options": generation_options.to_dict()
                    if generation_options
                    else None,
                    "pose_options": pose_options.to_dict(),
                },
                indent=2,
            )
        )
        return

    try:
        result = run_alignment_pipeline(
            base_dir,
            generation_options=generation_options,
            pose_options=pose_options,
            compose=compose,
            compose_options=compose_options,
        )
    except Exception as exc:  # pragma: no cover - runtime/dependency/network errors
        raise click.ClickException(str(exc)) from exc
    click.echo(
        json.dumps(
            {
                "status": "ok" if result.estimation.success else "warning",
                "action": "align_run",
                "result": result.to_dict(),
            },
            indent=2,
        )
    )
