"""Per-frame face recognition against a known gallery."""

import numpy as np
from typing import Optional, Tuple


class FaceRecognizer:
    """Detects and identifies faces by matching against a pre-built gallery."""

    def __init__(self, gallery_path: str, similarity_threshold: float = 0.4):
        raise NotImplementedError

    def identify(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify the person in a frame. Returns (user_id, confidence) or (None, 0.0)."""
        raise NotImplementedError
