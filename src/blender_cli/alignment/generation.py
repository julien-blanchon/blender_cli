"""AI generation stage: scene edit + object isolation + image-to-3D."""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from pathlib import Path

from PIL import Image

from ._fal import subscribe as fal_subscribe
from ._fal import upload_file as fal_upload_file
from .types import GenerationOptions, GenerationResult

logger = logging.getLogger(__name__)


def _require_fal_key() -> None:
    if os.environ.get("FAL_KEY", "").strip():
        return
    msg = (
        "FAL_KEY is required for alignment generation. "
        "Set it before running generation commands."
    )
    raise RuntimeError(msg)


def _download_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - explicit external API call
        return resp.read()


def _resize_to_reference_size(reference_path: Path, image_path: Path) -> None:
    with Image.open(reference_path) as ref, Image.open(image_path) as generated:
        if generated.size == ref.size:
            return
        resized = generated.resize(ref.size, Image.Resampling.LANCZOS)
    resized.save(image_path)


def _fal_timeout_seconds(max_attempts: int, interval_sec: int) -> float:
    return float(max(1, max_attempts * interval_sec))


def _run_fal_model(
    model_id: str,
    *,
    arguments: dict[str, object],
    step_name: str,
    max_attempts: int,
    interval_sec: int,
) -> dict[str, object]:
    return fal_subscribe(
        model_id,
        arguments=arguments,
        with_logs=True,
        logger=logger,
        log_prefix=f"[{step_name}] ",
        client_timeout=_fal_timeout_seconds(max_attempts, interval_sec),
    )


def _extract_url(value: object, *, label: str) -> str:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        url = value.get("url")
        if isinstance(url, str) and url:
            return url
    msg = f"Fal result missing {label} URL"
    raise KeyError(msg)


def _extract_image_url(payload: dict[str, object]) -> str:
    images = payload.get("images")
    if isinstance(images, list) and images:
        return _extract_url(images[0], label="generated image")
    msg = "Fal result missing generated images"
    raise KeyError(msg)


def _extract_glb_url(payload: dict[str, object]) -> str:
    direct_model = payload.get("model_glb")
    if direct_model is not None:
        return _extract_url(direct_model, label="GLB model")

    model_urls = payload.get("model_urls")
    if isinstance(model_urls, dict):
        return _extract_url(model_urls.get("glb"), label="GLB model")

    msg = "Fal result missing GLB model output"
    raise KeyError(msg)


def _step_edit_scene(
    *,
    reference_path: Path,
    output_path: Path,
    options: GenerationOptions,
) -> None:
    logger.info("Generation step 1/3: scene edit (%s)", options.object_name)
    image_url = fal_upload_file(reference_path)
    prompt = (
        f"Add {options.object_name} {options.placement}. "
        "The object should look realistic and match scene lighting. "
        "Keep everything else exactly the same: same room, furniture, lighting, and camera angle."
    )
    payload = _run_fal_model(
        options.scene_edit_model,
        arguments={
            "prompt": prompt,
            "image_urls": [image_url],
            "output_format": options.output_format,
        },
        step_name="scene-edit",
        max_attempts=options.edit_max_attempts,
        interval_sec=options.edit_poll_interval_sec,
    )
    result_url = _extract_image_url(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_download_bytes(result_url))
    _resize_to_reference_size(reference_path, output_path)


def _step_isolate_object(
    *,
    query_path: Path,
    output_path: Path,
    options: GenerationOptions,
) -> None:
    logger.info("Generation step 2/3: object isolation")
    image_url = fal_upload_file(query_path)
    prompt = (
        f"Create a clean product photo of {options.object_name} on a plain white background. "
        "Show only the object, centered, with soft studio lighting. "
        "Remove all room and furniture context."
    )
    payload = _run_fal_model(
        options.scene_edit_model,
        arguments={
            "prompt": prompt,
            "image_urls": [image_url],
            "output_format": options.output_format,
        },
        step_name="object-isolation",
        max_attempts=options.edit_max_attempts,
        interval_sec=options.edit_poll_interval_sec,
    )
    result_url = _extract_image_url(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(_download_bytes(result_url))


def _step_image_to_3d(
    *,
    object_reference_path: Path,
    object_path: Path,
    options: GenerationOptions,
) -> None:
    logger.info("Generation step 3/3: image-to-3D")
    image_url = fal_upload_file(object_reference_path)
    payload = _run_fal_model(
        options.image_to_3d_model,
        arguments={
            "image_url": image_url,
            "topology": "triangle",
            "target_polycount": 30000,
            "should_remesh": True,
            "should_texture": True,
            "enable_pbr": True,
        },
        step_name="image-to-3d",
        max_attempts=options.mesh_max_attempts,
        interval_sec=options.mesh_poll_interval_sec,
    )
    glb_url = _extract_glb_url(payload)
    object_path.parent.mkdir(parents=True, exist_ok=True)
    object_path.write_bytes(_download_bytes(glb_url))
    thumbnail_value = payload.get("thumbnail")
    if thumbnail_value is not None:
        thumbnail_url = _extract_url(thumbnail_value, label="thumbnail")
        thumb_path = object_path.with_name("object_thumbnail.png")
        thumb_path.write_bytes(_download_bytes(thumbnail_url))


def generate_alignment_assets(
    directory: str | Path,
    options: GenerationOptions,
) -> GenerationResult:
    """Run generation pipeline in an example directory."""
    base_dir = Path(directory).resolve()
    if not base_dir.is_dir():
        msg = f"Directory not found: {base_dir}"
        raise FileNotFoundError(msg)

    _require_fal_key()
    reference_path = base_dir / options.reference_name
    if not reference_path.is_file():
        msg = f"Reference image not found: {reference_path}"
        raise FileNotFoundError(msg)

    query_path = base_dir / options.query_name
    object_reference_path = base_dir / options.object_reference_name
    object_path = base_dir / options.object_name_on_disk

    _step_edit_scene(
        reference_path=reference_path,
        output_path=query_path,
        options=options,
    )
    _step_isolate_object(
        query_path=query_path,
        output_path=object_reference_path,
        options=options,
    )
    if options.skip_3d:
        logger.info("Generation step 3/3 skipped (--skip-3d)")
        resolved_object_path: Path | None = (
            object_path if object_path.exists() else None
        )
    else:
        _step_image_to_3d(
            object_reference_path=object_reference_path,
            object_path=object_path,
            options=options,
        )
        resolved_object_path = object_path

    provenance_path = base_dir / "provenance.json"
    provenance = {
        "object_name": options.object_name,
        "placement": options.placement,
        "scene_edit_model": options.scene_edit_model,
        "image_to_3d_model": options.image_to_3d_model,
        "skip_3d": options.skip_3d,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    return GenerationResult(
        directory=base_dir,
        query_path=query_path,
        object_reference_path=object_reference_path,
        object_path=resolved_object_path,
        provenance_path=provenance_path,
        options=options,
    )
