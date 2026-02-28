"""Per-frame face recognition against a known gallery."""

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Detects and identifies faces by matching against a pre-built gallery.

    Uses InsightFace for face detection + embedding extraction, then compares
    against the enrolled gallery using cosine similarity.

    Typical usage in a pipeline:
        recognizer = FaceRecognizer.from_config(config)
        for frame in video:
            name, confidence = recognizer.identify(frame)
    """

    def __init__(
        self,
        gallery: Dict[str, np.ndarray],
        similarity_threshold: float = 0.4,
        model_name: str = "buffalo_l",
        det_size: Tuple[int, int] = (640, 640),
        providers: list = None,
    ):
        self._gallery = gallery
        self._threshold = similarity_threshold

        # Pre-compute gallery matrix for vectorized comparison
        if gallery:
            self._names = list(gallery.keys())
            self._embeddings = np.stack([gallery[n] for n in self._names])
        else:
            self._names = []
            self._embeddings = np.empty((0, 512))

        # Initialize InsightFace
        self._app = FaceAnalysis(
            name=model_name,
            providers=providers or ["CPUExecutionProvider"],
        )
        self._app.prepare(ctx_id=-1, det_size=det_size)

    @classmethod
    def from_config(cls, config) -> "FaceRecognizer":
        """Create FaceRecognizer from PipelineConfig."""
        face_cfg = config.face
        gallery_path = Path(face_cfg.gallery_path)

        if not gallery_path.exists():
            raise FileNotFoundError(
                f"Gallery not found: {gallery_path}. "
                "Run enrollment first (python scripts/enroll_faces.py)."
            )

        with open(gallery_path, "rb") as f:
            gallery = pickle.load(f)

        return cls(
            gallery=gallery,
            similarity_threshold=face_cfg.similarity_threshold,
            model_name=face_cfg.model_name,
            det_size=tuple(face_cfg.det_size),
            providers=face_cfg.providers,
        )

    def identify(self, frame: np.ndarray) -> Tuple[Optional[str], float]:
        """Identify the person in a frame.

        Detects the largest face, computes its embedding, and compares
        against all gallery entries using cosine similarity.

        Args:
            frame: BGR image (numpy array).

        Returns:
            (person_name, similarity) if match found above threshold,
            (None, 0.0) otherwise.
        """
        if len(self._names) == 0:
            return None, 0.0

        faces = self._app.get(frame)
        if not faces:
            return None, 0.0

        # Pick the largest detected face
        best_face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        # L2-normalize the query embedding
        query = best_face.embedding
        query = query / np.linalg.norm(query)

        # Cosine similarity against all gallery entries
        similarities = self._embeddings @ query
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= self._threshold:
            return self._names[best_idx], best_sim

        return None, best_sim
