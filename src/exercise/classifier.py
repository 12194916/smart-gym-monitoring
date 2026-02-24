"""Exercise classifier using trained model on pose angle features.

Uses a sliding window of recent JointAngles to compute aggregate statistics
(mean, std, min, max per angle), then feeds the same 37 features to the
trained sklearn pipeline for classification.
"""

import pickle
from collections import deque
from typing import Optional, Tuple

import numpy as np

from src.exercise.definitions import Exercise
from src.pose.angles import JointAngles

ANGLE_NAMES = [
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "torso_inclination",
]

# Map raw model label strings (folder names from training data) to Exercise enum.
# Labels not listed here default to Exercise.UNKNOWN.
LABEL_TO_EXERCISE = {
    "barbell biceps curl": Exercise.BICEP_CURL,
    "hammer curl": Exercise.BICEP_CURL,
    "bench press": Exercise.BENCH_PRESS,
    "incline bench press": Exercise.BENCH_PRESS,
    "decline bench press": Exercise.BENCH_PRESS,
    "deadlift": Exercise.DEADLIFT,
    "romanian deadlift": Exercise.DEADLIFT,
    "push-up": Exercise.PUSHUP,
    "pull Up": Exercise.PULLUP,
    "lat pulldown": Exercise.PULLUP,
    "squat": Exercise.SQUAT,
    "lunge": Exercise.LUNGE,
    "shoulder press": Exercise.SHOULDER_PRESS,
    "lateral raise": Exercise.SHOULDER_PRESS,
}


class ExerciseClassifier:
    """Classifies exercises from joint angle features using a trained sklearn model."""

    def __init__(self, model_path: str, window_size: int = 30):
        with open(model_path, "rb") as f:
            saved = pickle.load(f)
        self.pipeline = saved["pipeline"]
        self.label_encoder = saved["label_encoder"]
        self.feature_columns = saved["feature_columns"]
        self.window_size = window_size
        self.buffer: deque = deque(maxlen=window_size)
        self._last_raw_label = "unknown"

    def update(self, angles: JointAngles):
        """Add angles to the sliding window buffer. Call this every frame."""
        self.buffer.append(angles)

    def classify(self, angles: Optional[JointAngles] = None) -> Tuple[Exercise, float]:
        """Classify exercise from current sliding window.

        If angles is provided, it's added to the buffer first (for convenience
        when calling classify without a separate update call).

        Returns (exercise, confidence). Returns (UNKNOWN, 0.0) if buffer
        has fewer than 5 frames.
        """
        if angles is not None:
            self.buffer.append(angles)

        if len(self.buffer) < 5:
            return Exercise.UNKNOWN, 0.0

        # Compute aggregate features from sliding window
        features = self._compute_features()

        # Build feature vector in exact column order the model expects
        feature_vector = [features[col] for col in self.feature_columns]
        X = np.array([feature_vector])

        # Predict
        pred_idx = self.pipeline.predict(X)[0]
        raw_label = self.label_encoder.inverse_transform([pred_idx])[0]
        self._last_raw_label = raw_label

        # Get confidence via predict_proba if available
        confidence = 0.0
        if hasattr(self.pipeline, "predict_proba"):
            proba = self.pipeline.predict_proba(X)[0]
            confidence = float(proba[pred_idx])

        # Map raw label to Exercise enum
        exercise = LABEL_TO_EXERCISE.get(raw_label, Exercise.UNKNOWN)

        return exercise, confidence

    @property
    def last_raw_label(self) -> str:
        """The raw model label string from the most recent classification."""
        return self._last_raw_label

    def _compute_features(self) -> dict:
        """Compute mean/std/min/max from the sliding window buffer."""
        features = {}
        for name in ANGLE_NAMES:
            values = np.array([getattr(a, name) for a in self.buffer])
            features[f"{name}_mean"] = float(np.mean(values))
            features[f"{name}_std"] = float(np.std(values))
            features[f"{name}_min"] = float(np.min(values))
            features[f"{name}_max"] = float(np.max(values))
        features["num_frames"] = len(self.buffer)
        return features

    def reset(self):
        """Clear the sliding window buffer."""
        self.buffer.clear()
        self._last_raw_label = "unknown"
