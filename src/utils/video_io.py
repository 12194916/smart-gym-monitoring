"""OpenCV video read/write helpers."""

import cv2
import numpy as np
from typing import Generator, Tuple


def read_video_frames(video_path: str) -> Generator[Tuple[int, np.ndarray, int], None, None]:
    """Yield (frame_index, frame, timestamp_ms) tuples from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        yield frame_idx, frame, timestamp_ms
        frame_idx += 1

    cap.release()


def create_video_writer(output_path: str, fps: float, width: int, height: int, codec: str = "mp4v") -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for the output annotated video."""
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def get_video_metadata(video_path: str) -> dict:
    """Return video metadata: fps, width, height, total_frames, duration_seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    metadata = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    metadata["duration_seconds"] = metadata["total_frames"] / metadata["fps"] if metadata["fps"] > 0 else 0
    cap.release()
    return metadata
