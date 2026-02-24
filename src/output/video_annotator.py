"""Draw pose skeleton and overlays on video frames."""

import numpy as np


class VideoAnnotator:
    """Draws skeleton, labels, and rep counter on video frames."""

    def draw(self, frame: np.ndarray, landmarks: dict, exercise: str,
             rep_count: int, user_id: str = None, angles: dict = None) -> np.ndarray:
        """Draw all overlays on a frame and return the annotated copy."""
        raise NotImplementedError
