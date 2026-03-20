"""Public configuration and result types for the alignment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from pathlib import Path

ScaleMethod = Literal["raycast", "depth", "marigold"]


@dataclass(slots=True)
class GenerationOptions:
    """Options for AI image-editing and image-to-3D generation."""

    object_name: str
    placement: str
    reference_name: str = "reference.png"
    query_name: str = "query.png"
    object_reference_name: str = "object_reference.png"
    object_name_on_disk: str = "object.glb"
    scene_edit_model: str = "fal-ai/nano-banana-pro/edit"
    image_to_3d_model: str = "fal-ai/meshy/v6/image-to-3d"
    skip_3d: bool = False
    output_format: str = "png"
    edit_max_attempts: int = 12
    edit_poll_interval_sec: int = 10
    mesh_max_attempts: int = 40
    mesh_poll_interval_sec: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PoseEstimationOptions:
    """Options for silhouette-based pose estimation from an example directory."""

    object_name: str = "object.glb"
    query_name: str = "query.png"
    reference_name: str = "reference.png"
    scene_name: str = "scene.glb"
    camera_name: str = "camera.json"
    output_name: str = "output"
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
    use_sam: bool = True
    sam_model: str = "fal-ai/sam-3/image"
    sam_max_masks: int = 5
    scale_method: ScaleMethod = "raycast"
    debug: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ComposeOptions:
    """Options for composing the predicted object into the scene GLB."""

    output_name: str = "combined.glb"
    pose_result_name: str = "pose_result.json"
    output_dir_name: str = "output"
    scene_name: str = "scene.glb"
    object_name: str = "object.glb"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GenerationResult:
    """Output paths from the generation stage."""

    directory: Path
    query_path: Path
    object_reference_path: Path
    object_path: Path | None
    provenance_path: Path
    options: GenerationOptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "query_path": str(self.query_path),
            "object_reference_path": str(self.object_reference_path),
            "object_path": str(self.object_path)
            if self.object_path is not None
            else None,
            "provenance_path": str(self.provenance_path),
            "options": self.options.to_dict(),
        }


@dataclass(slots=True)
class PoseEstimationResult:
    """Pose estimation status and outputs."""

    directory: Path
    output_dir: Path
    pose_result_path: Path
    success: bool
    result: dict[str, Any]
    options: PoseEstimationOptions

    def to_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "output_dir": str(self.output_dir),
            "pose_result_path": str(self.pose_result_path),
            "success": self.success,
            "result": self.result,
            "options": self.options.to_dict(),
        }


@dataclass(slots=True)
class ComposeResult:
    """Scene composition output."""

    output_scene_path: Path
    pose_result_path: Path
    scene_path: Path
    object_path: Path
    transform: list[list[float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_scene_path": str(self.output_scene_path),
            "pose_result_path": str(self.pose_result_path),
            "scene_path": str(self.scene_path),
            "object_path": str(self.object_path),
            "transform": self.transform,
        }


@dataclass(slots=True)
class AlignmentPipelineResult:
    """Result of the full optional generation + estimation + composition pipeline."""

    generation: GenerationResult | None
    estimation: PoseEstimationResult
    composition: ComposeResult | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation.to_dict()
            if self.generation is not None
            else None,
            "estimation": self.estimation.to_dict(),
            "composition": self.composition.to_dict()
            if self.composition is not None
            else None,
        }
