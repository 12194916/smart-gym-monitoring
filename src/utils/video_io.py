"""OpenCV video read/write helpers."""

import cv2
import numpy as np
from typing import Generator, Tuple


def read_video_frames(video_path: str) -> Generator[Tuple[int, np.ndarray, int], None, None]:
    """Yield (frame_index, frame, timestamp_ms) tuples from a video file."""
    raise NotImplementedError


def create_video_writer(output_path: str, fps: float, width: int, height: int, codec: str = "mp4v") -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for the output annotated video."""
    raise NotImplementedError


def get_video_metadata(video_path: str) -> dict:
    """Return video metadata: fps, width, height, total_frames, duration_seconds."""
    raise NotImplementedError
