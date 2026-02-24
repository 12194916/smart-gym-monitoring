"""Exercise classifier using trained model on pose angle features."""

from typing import Tuple
from src.exercise.definitions import Exercise
from src.pose.angles import JointAngles


class ExerciseClassifier:
    """Classifies exercises from joint angle features using a trained sklearn model."""

    def __init__(self, model_path: str, window_size: int = 30):
        raise NotImplementedError

    def classify(self, angles: JointAngles) -> Tuple[Exercise, float]:
        """Add angles to buffer and classify using sliding window. Returns (exercise, confidence)."""
        raise NotImplementedError
