"""Tests for ClassificationStabilizer (EMA + hysteresis + min hold)."""

import pytest
from src.exercise.stabilizer import ClassificationStabilizer
from src.exercise.definitions import Exercise


class TestClassificationStabilizer:

    def test_initial_state_returns_none(self):
        s = ClassificationStabilizer()
        assert s._stable_label is None
        assert s._stable_confidence == 0.0

    def test_first_prediction_commits(self):
        """First high-confidence prediction should commit within a few frames."""
        s = ClassificationStabilizer(ema_alpha=0.3, min_confidence=0.3)
        probs = {"squat": 0.9, "deadlift": 0.05, "pushup": 0.05}
        for _ in range(5):
            label, conf = s.update_with_probabilities(probs)
        assert label == "squat"
        assert conf > 0.3

    def test_resists_transient_flicker(self):
        """A few wrong frames should NOT switch the label."""
        s = ClassificationStabilizer(
            ema_alpha=0.15, hysteresis_margin=0.15, min_hold_frames=10
        )
        # Establish squat over 20 frames
        for _ in range(20):
            s.update_with_probabilities({"squat": 0.9, "deadlift": 0.05, "pushup": 0.05})

        # 3 frames of deadlift (simulating bottom-of-squat confusion)
        for _ in range(3):
            label, _ = s.update_with_probabilities(
                {"squat": 0.2, "deadlift": 0.7, "pushup": 0.1}
            )

        assert label == "squat", "Label should NOT switch after 3 transient frames"

    def test_allows_genuine_transition(self):
        """Sustained new exercise should eventually switch the label."""
        s = ClassificationStabilizer(
            ema_alpha=0.15, hysteresis_margin=0.15, min_hold_frames=10
        )
        # Establish squat
        for _ in range(20):
            s.update_with_probabilities({"squat": 0.9, "deadlift": 0.05, "pushup": 0.05})

        # Genuine switch to deadlift for many frames
        for _ in range(60):
            label, _ = s.update_with_probabilities(
                {"squat": 0.05, "deadlift": 0.9, "pushup": 0.05}
            )

        assert label == "deadlift", "Label should switch after sustained new exercise"

    def test_min_hold_prevents_rapid_oscillation(self):
        """Cannot switch before min_hold_frames even with high confidence."""
        s = ClassificationStabilizer(
            ema_alpha=0.5, hysteresis_margin=0.0, min_hold_frames=30
        )
        # Establish squat with just enough frames to commit (few frames)
        for _ in range(5):
            s.update_with_probabilities({"squat": 0.9, "deadlift": 0.1})
        assert s._stable_label == "squat"

        # Immediately push high deadlift — still within the 30-frame hold window
        # (only ~5 frames since initial commit)
        for _ in range(10):
            label, _ = s.update_with_probabilities({"squat": 0.1, "deadlift": 0.9})

        assert label == "squat", "Hold time should prevent switching"

    def test_confidence_gate_skips_low_confidence(self):
        """Frames below min_confidence should not affect the EMA."""
        s = ClassificationStabilizer(min_confidence=0.3, ema_alpha=0.15)
        # Establish squat
        for _ in range(20):
            s.update_with_probabilities({"squat": 0.9, "deadlift": 0.05, "pushup": 0.05})

        # Low-confidence garbage frames (max is 0.15, below threshold)
        for _ in range(10):
            label, _ = s.update_with_probabilities(
                {"squat": 0.1, "deadlift": 0.15, "pushup": 0.05}
            )

        assert label == "squat", "Low-confidence frames should be ignored"

    def test_reset_clears_state(self):
        """After reset, stabilizer should behave as freshly constructed."""
        s = ClassificationStabilizer()
        for _ in range(10):
            s.update_with_probabilities({"squat": 0.9, "deadlift": 0.1})

        s.reset()
        assert s._stable_label is None
        assert s._ema == {}
        assert s._frames_since_switch == 0
        assert s._initialized is False

    def test_legacy_update_interface(self):
        """The legacy update(Exercise, float) interface should work."""
        s = ClassificationStabilizer(ema_alpha=0.3, min_confidence=0.3)
        for _ in range(10):
            ex, conf = s.update(Exercise.SQUAT, 0.9)
        assert ex == Exercise.SQUAT
        assert conf > 0.3

    def test_legacy_update_unknown_passes_through(self):
        """Updating with UNKNOWN should not crash and should return current state."""
        s = ClassificationStabilizer(ema_alpha=0.3)
        # Establish squat first
        for _ in range(10):
            s.update(Exercise.SQUAT, 0.9)
        ex, conf = s.update(Exercise.UNKNOWN, 0.0)
        assert ex == Exercise.SQUAT

    def test_hysteresis_prevents_marginal_switch(self):
        """A challenger that barely exceeds the current should not switch."""
        s = ClassificationStabilizer(
            ema_alpha=0.3, hysteresis_margin=0.2, min_hold_frames=0
        )
        # Establish squat
        for _ in range(20):
            s.update_with_probabilities({"squat": 0.6, "deadlift": 0.4})

        # Deadlift slightly winning but within margin
        for _ in range(20):
            label, _ = s.update_with_probabilities({"squat": 0.45, "deadlift": 0.55})

        assert label == "squat", "Hysteresis should prevent marginal switches"

    def test_ema_decays_absent_classes(self):
        """Classes not in prediction should decay toward zero."""
        s = ClassificationStabilizer(ema_alpha=0.3)
        # Feed squat + deadlift
        for _ in range(10):
            s.update_with_probabilities({"squat": 0.5, "deadlift": 0.5})

        # Now only squat, deadlift absent
        for _ in range(20):
            s.update_with_probabilities({"squat": 0.9})

        assert s._ema["deadlift"] < 0.05, "Absent class EMA should decay"
