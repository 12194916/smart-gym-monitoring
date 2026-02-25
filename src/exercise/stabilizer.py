"""Classification stabilizer using majority voting over recent predictions.

Prevents exercise label flickering by requiring a new exercise to win
a majority vote over a sliding window of recent classification results
before the label changes.
"""

from collections import deque, Counter
from typing import Tuple

from src.exercise.definitions import Exercise


class ClassificationStabilizer:
    """Smooths exercise classification output using majority voting."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        self._stable_exercise: Exercise = Exercise.UNKNOWN
        self._stable_confidence: float = 0.0

    def update(self, exercise: Exercise, confidence: float) -> Tuple[Exercise, float]:
        """Add a new prediction and return the stabilized result.

        The new exercise must win >50% of votes in the history window
        to become the stable label. UNKNOWN votes are excluded.
        """
        self._history.append((exercise, confidence))

        # Count votes (exclude UNKNOWN)
        votes = Counter()
        confidences = {}
        for ex, conf in self._history:
            if ex != Exercise.UNKNOWN:
                votes[ex] += 1
                if ex not in confidences or conf > confidences[ex]:
                    confidences[ex] = conf

        if not votes:
            return Exercise.UNKNOWN, 0.0

        winner, winner_count = votes.most_common(1)[0]

        # Only switch if the new exercise has a strict majority
        if winner != self._stable_exercise:
            if winner_count > len(self._history) / 2:
                self._stable_exercise = winner
                self._stable_confidence = confidences[winner]
        else:
            # Update confidence for the current stable exercise
            if winner in confidences:
                self._stable_confidence = confidences[winner]

        return self._stable_exercise, self._stable_confidence

    def reset(self):
        """Clear voting history."""
        self._history.clear()
        self._stable_exercise = Exercise.UNKNOWN
        self._stable_confidence = 0.0
