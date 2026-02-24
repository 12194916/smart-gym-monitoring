"""MediaPipe PoseLandmarker wrapper (Tasks API)."""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float
    visibility: float


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker for frame-by-frame pose extraction."""

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        raise NotImplementedError

    def process(self, frame: np.ndarray, timestamp_ms: int) -> Optional[Dict[str, LandmarkPoint]]:
        """Process a single frame and return landmarks dict, or None if no pose detected."""
        raise NotImplementedError

    def close(self):
        """Release resources."""
        raise NotImplementedError
