"""Classification stabilizer with EMA smoothing, hysteresis, and minimum hold.

Prevents exercise label flickering using three mechanisms:
1. Exponential moving average (EMA) per class for temporal smoothing
2. Hysteresis margin: challenger must exceed current label by a margin to switch
3. Minimum hold time: label is locked for N frames after each switch
"""

from typing import Dict, Optional, Tuple

from src.exercise.definitions import Exercise


class ClassificationStabilizer:
    """Smooths exercise classification using EMA + hysteresis + min hold."""

    def __init__(
        self,
        ema_alpha: float = 0.15,
        hysteresis_margin: float = 0.15,
        min_hold_frames: int = 15,
        min_confidence: float = 0.3,
    ):
        self.ema_alpha = ema_alpha
        self.hysteresis_margin = hysteresis_margin
        self.min_hold_frames = min_hold_frames
        self.min_confidence = min_confidence

        # Per-class EMA scores
        self._ema: Dict[str, float] = {}

        # Current stable state
        self._stable_label: Optional[str] = None
        self._stable_confidence: float = 0.0

        # Hold timer: frames since last label switch
        self._frames_since_switch: int = 0
        self._initialized: bool = False

    def update_with_probabilities(
        self, class_probabilities: Dict[str, float]
    ) -> Tuple[Optional[str], float]:
        """Update with full probability distribution from predict_proba.

        Args:
            class_probabilities: Dict mapping class name -> probability.
                E.g., {"squat": 0.85, "deadlift": 0.10, "pushup": 0.05}

        Returns:
            (stable_label, confidence) -- the smoothed, stable prediction.
        """
        self._frames_since_switch += 1

        # Confidence gate: if max prob is below threshold, skip this frame
        max_prob = max(class_probabilities.values()) if class_probabilities else 0.0
        if max_prob < self.min_confidence:
            return self._stable_label, self._stable_confidence

        # Update EMA for all classes present in prediction
        for cls, prob in class_probabilities.items():
            if cls in self._ema:
                self._ema[cls] = (
                    self.ema_alpha * prob
                    + (1 - self.ema_alpha) * self._ema[cls]
                )
            else:
                self._ema[cls] = prob

        # Decay classes not in current prediction toward zero
        for cls in list(self._ema.keys()):
            if cls not in class_probabilities:
                self._ema[cls] *= (1 - self.ema_alpha)

        # Find the EMA leader
        if not self._ema:
            return self._stable_label, self._stable_confidence

        ema_leader = max(self._ema, key=self._ema.get)
        ema_leader_score = self._ema[ema_leader]

        # First label commitment (no hold constraint)
        if not self._initialized:
            if ema_leader_score > self.min_confidence:
                self._stable_label = ema_leader
                self._stable_confidence = ema_leader_score
                self._initialized = True
                self._frames_since_switch = 0
            return self._stable_label, self._stable_confidence

        # Current label's EMA score
        current_score = self._ema.get(self._stable_label, 0.0)

        # Check if leader is different from current stable label
        if ema_leader != self._stable_label:
            # Must pass BOTH hysteresis AND minimum hold
            hold_elapsed = self._frames_since_switch >= self.min_hold_frames
            margin_exceeded = (ema_leader_score - current_score) > self.hysteresis_margin

            if hold_elapsed and margin_exceeded:
                self._stable_label = ema_leader
                self._stable_confidence = ema_leader_score
                self._frames_since_switch = 0
        else:
            # Same label — just update confidence
            self._stable_confidence = ema_leader_score

        return self._stable_label, self._stable_confidence

    def update(self, exercise: Exercise, confidence: float) -> Tuple[Exercise, float]:
        """Legacy interface: single prediction + confidence.

        Constructs an approximate probability distribution and delegates
        to update_with_probabilities.
        """
        if exercise == Exercise.UNKNOWN:
            label = self._stable_label
            if label is None:
                return Exercise.UNKNOWN, 0.0
            try:
                return Exercise(label), self._stable_confidence
            except ValueError:
                return Exercise.UNKNOWN, self._stable_confidence

        # Approximate distribution: assign confidence to predicted class,
        # distribute remainder equally among other known classes
        probs = {}
        other_classes = [
            e.value for e in Exercise
            if e != Exercise.UNKNOWN and e != exercise
        ]
        remaining = 1.0 - confidence
        per_other = remaining / max(len(other_classes), 1)
        for cls in other_classes:
            probs[cls] = per_other
        probs[exercise.value] = confidence

        label, conf = self.update_with_probabilities(probs)

        if label is None:
            return Exercise.UNKNOWN, 0.0
        try:
            return Exercise(label), conf
        except ValueError:
            return Exercise.UNKNOWN, conf

    def reset(self):
        """Clear all state."""
        self._ema.clear()
        self._stable_label = None
        self._stable_confidence = 0.0
        self._frames_since_switch = 0
        self._initialized = False
