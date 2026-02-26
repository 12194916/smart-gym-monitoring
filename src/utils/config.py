"""YAML configuration loader with typed dataclasses."""

import yaml
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class FaceIDConfig:
    model_name: str = "buffalo_l"
    gallery_path: str = "data/gallery.pkl"
    enrollment_dir: str = "data/enrollment"
    det_size: Tuple[int, int] = (640, 640)
    similarity_threshold: float = 0.4
    max_identification_frames: int = 60
    providers: List[str] = field(default_factory=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])


@dataclass
class PersonDetectionConfig:
    model_path: str = "data/models/yolov8n.pt"
    confidence_threshold: float = 0.5
    center_weight: float = 0.4
    size_weight: float = 0.6
    bbox_padding: float = 0.15
    iou_threshold: float = 0.3
    max_missing_frames: int = 5


@dataclass
class PoseConfig:
    model_path: str = "data/models/pose_landmarker_heavy.task"
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_poses: int = 3
    person_size_weight: float = 0.6
    person_center_weight: float = 0.4


@dataclass
class ExerciseConfig:
    classifier_model_path: str = "data/models/exercise_classifier.pkl"
    classification_interval_frames: int = 15
    window_size: int = 30
    min_classification_score: float = 0.3
    stabilizer_window: int = 5
    stabilizer_ema_alpha: float = 0.15
    stabilizer_hysteresis_margin: float = 0.15
    stabilizer_min_hold_frames: int = 15


@dataclass
class RepCounterConfig:
    min_rep_duration_ms: int = 800
    angle_smoothing_window: int = 5


@dataclass
class OutputConfig:
    annotated_video_codec: str = "mp4v"
    json_indent: int = 2
    draw_skeleton: bool = True
    draw_angles: bool = True
    draw_labels: bool = True


@dataclass
class PipelineConfig:
    face: FaceIDConfig
    person_detection: PersonDetectionConfig
    pose: PoseConfig
    exercise: ExerciseConfig
    rep_counter: RepCounterConfig
    output: OutputConfig


def load_config(path: str = "config.yaml") -> PipelineConfig:
    """Load YAML config file into typed PipelineConfig dataclass."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return PipelineConfig(
        face=FaceIDConfig(**raw.get("face_id", {})),
        person_detection=PersonDetectionConfig(**raw.get("person_detection", {})),
        pose=PoseConfig(**raw.get("pose", {})),
        exercise=ExerciseConfig(**raw.get("exercise", {})),
        rep_counter=RepCounterConfig(**raw.get("rep_counter", {})),
        output=OutputConfig(**raw.get("output", {})),
    )
