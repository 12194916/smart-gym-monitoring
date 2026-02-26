"""Unit tests for the peak-valley rep counter."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest
from dataclasses import dataclass
from src.rep_counter.counter import ExerciseRepCounter, RepCounterManager
from src.rep_counter.rep_rules import REP_RULES, AngleRepRule, TwistRepRule, HoldRule


@dataclass
class FakeLandmark:
    """Minimal landmark mock with x, y, visibility."""
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.99


def make_squat_landmarks(knee_angle_approx):
    """Create body landmarks that produce approximately the given knee angle."""
    hip = FakeLandmark(x=100, y=100)
    ankle = FakeLandmark(x=100, y=300)
    knee_y = 200

    target_cos = math.cos(math.radians(knee_angle_approx))
    if abs(1 - target_cos) < 1e-10:
        d = 0
    else:
        d_sq = 10000 * (1 + target_cos) / (1 - target_cos)
        d = math.sqrt(max(0, d_sq))

    knee = FakeLandmark(x=100 + d, y=knee_y)

    return {
        "left_hip": hip, "right_hip": hip,
        "left_knee": knee, "right_knee": knee,
        "left_ankle": ankle, "right_ankle": ankle,
        "left_shoulder": FakeLandmark(x=100, y=50),
        "right_shoulder": FakeLandmark(x=100, y=50),
        "left_elbow": FakeLandmark(x=100, y=150, visibility=0.1),
        "right_elbow": FakeLandmark(x=100, y=150, visibility=0.1),
        "left_wrist": FakeLandmark(x=100, y=200, visibility=0.1),
        "right_wrist": FakeLandmark(x=100, y=200, visibility=0.1),
    }


def make_elbow_landmarks(elbow_angle_approx):
    """Create body landmarks that produce approximately the given elbow angle."""
    shoulder = FakeLandmark(x=100, y=100)
    elbow_y = 200

    target_cos = math.cos(math.radians(elbow_angle_approx))
    if abs(1 - target_cos) < 1e-10:
        d = 0
    else:
        d_sq = 10000 * (1 + target_cos) / (1 - target_cos)
        d = math.sqrt(max(0, d_sq))

    elbow = FakeLandmark(x=100 + d, y=elbow_y)
    wrist = FakeLandmark(x=100, y=300)

    return {
        "left_shoulder": shoulder, "right_shoulder": shoulder,
        "left_elbow": elbow, "right_elbow": elbow,
        "left_wrist": wrist, "right_wrist": wrist,
        "left_hip": FakeLandmark(x=100, y=250, visibility=0.1),
        "right_hip": FakeLandmark(x=100, y=250, visibility=0.1),
        "left_knee": FakeLandmark(x=100, y=350, visibility=0.1),
        "right_knee": FakeLandmark(x=100, y=350, visibility=0.1),
        "left_ankle": FakeLandmark(x=100, y=450, visibility=0.1),
        "right_ankle": FakeLandmark(x=100, y=450, visibility=0.1),
    }


# -----------------------------------------------------------------------
# Core peak-valley detection tests
# -----------------------------------------------------------------------

class TestSquatRepCounter:
    """Test squat rep counting with knee angle peak-valley detection."""

    def test_no_reps_initial(self):
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)
        assert counter.rep_count == 0

    def test_one_full_squat_rep(self):
        """Angle goes up → down → up = peak → valley → peak = 1 rep."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        # Standing at 160° (establishes initial direction)
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        # Squat to 80° (drops 80° from peak of 160° → reversal confirmed)
        for _ in range(3):
            counter.update(make_squat_landmarks(80), frame_number=frame)
            frame += 1

        assert counter.rep_count == 0  # only first reversal (half rep)

        # Stand back to 160° (rises 80° from valley of 80° → second reversal)
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        assert counter.rep_count == 1

    def test_multiple_reps(self):
        """3 full up-down-up cycles = 3 reps."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        for rep in range(3):
            for _ in range(3):
                counter.update(make_squat_landmarks(160), frame_number=frame)
                frame += 1
            for _ in range(3):
                counter.update(make_squat_landmarks(80), frame_number=frame)
                frame += 1
            for _ in range(3):
                counter.update(make_squat_landmarks(160), frame_number=frame)
                frame += 1

        assert counter.rep_count == 3

    def test_small_wobble_no_false_rep(self):
        """Small fluctuations (< min_amplitude) should NOT trigger reversals."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        # Stand at 160°
        for _ in range(5):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        # Small wobble: 160 → 150 → 160 (amplitude 10° < min_amplitude 25°)
        for _ in range(3):
            counter.update(make_squat_landmarks(150), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        assert counter.rep_count == 0

    def test_rep_events_have_timestamps(self):
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(80), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        assert len(counter.reps) == 1
        rep = counter.reps[0]
        assert rep.rep_number == 1
        assert rep.timestamp_ms > 0
        assert rep.frame_number > 0

    def test_works_with_different_absolute_angles(self):
        """Peak-valley should work regardless of the absolute angle range.

        A squat filmed from a weird angle might show 130°→100° instead of
        160°→80°, but the 30° amplitude still exceeds min_amplitude=25°.
        """
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        # Narrower range: 130° → 100° → 130° (amplitude 30° > 25°)
        for _ in range(3):
            counter.update(make_squat_landmarks(130), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(100), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(130), frame_number=frame)
            frame += 1

        assert counter.rep_count == 1


class TestMinRepDuration:
    """Test minimum rep duration enforcement."""

    def test_fast_reversals_blocked(self):
        """Reversals faster than min_rep_duration/2 should be ignored."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=800,
                                     angle_smoothing_window=1)

        # Stand (frame 0), squat (frame 1), stand (frame 2) — too fast
        counter.update(make_squat_landmarks(160), frame_number=0)
        counter.update(make_squat_landmarks(80), frame_number=1)
        counter.update(make_squat_landmarks(160), frame_number=2)

        assert counter.rep_count == 0

    def test_slow_reversals_counted(self):
        """Reversals with enough spacing should be counted."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=800,
                                     angle_smoothing_window=1)

        counter.update(make_squat_landmarks(160), frame_number=0)
        counter.update(make_squat_landmarks(80), frame_number=15)
        counter.update(make_squat_landmarks(160), frame_number=30)

        assert counter.rep_count == 1


class TestAngleSmoothing:
    """Test that smoothing prevents noise-triggered reversals."""

    def test_single_spike_ignored(self):
        """One extreme outlier in a smooth signal should be smoothed away."""
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=5)

        frame = 0
        # 10 frames at 160°
        for _ in range(10):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        # 1 spike to 80° → average of [160,160,160,160,80] = 144° → only 16° drop
        counter.update(make_squat_landmarks(80), frame_number=frame)
        frame += 1

        # Continue at 160°
        for _ in range(5):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        assert counter.rep_count == 0


class TestBicepsCurl:
    """Test elbow angle tracking for biceps curl."""

    def test_one_curl_rep(self):
        counter = ExerciseRepCounter("barbell biceps curl", fps=30,
                                     min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        # Arms straight ~155°
        for _ in range(3):
            counter.update(make_elbow_landmarks(155), frame_number=frame)
            frame += 1
        # Arms curled ~45° (drop of 110° >> min_amplitude 25°)
        for _ in range(3):
            counter.update(make_elbow_landmarks(45), frame_number=frame)
            frame += 1
        # Arms straight again
        for _ in range(3):
            counter.update(make_elbow_landmarks(155), frame_number=frame)
            frame += 1

        assert counter.rep_count == 1


class TestRussianTwist:
    """Test peak-valley on horizontal wrist offset."""

    def _make_twist_landmarks(self, wrist_offset_x):
        shoulder_width = 100
        shoulder_cx = 300

        ls = FakeLandmark(x=shoulder_cx - shoulder_width / 2, y=100)
        rs = FakeLandmark(x=shoulder_cx + shoulder_width / 2, y=100)

        wrist_x = shoulder_cx + wrist_offset_x
        lw = FakeLandmark(x=wrist_x - 10, y=200)
        rw = FakeLandmark(x=wrist_x + 10, y=200)

        return {
            "left_shoulder": ls, "right_shoulder": rs,
            "left_wrist": lw, "right_wrist": rw,
            "left_hip": FakeLandmark(x=300, y=250),
            "right_hip": FakeLandmark(x=300, y=250),
            "left_elbow": FakeLandmark(x=300, y=150, visibility=0.1),
            "right_elbow": FakeLandmark(x=300, y=150, visibility=0.1),
            "left_knee": FakeLandmark(x=300, y=350, visibility=0.1),
            "right_knee": FakeLandmark(x=300, y=350, visibility=0.1),
            "left_ankle": FakeLandmark(x=300, y=450, visibility=0.1),
            "right_ankle": FakeLandmark(x=300, y=450, visibility=0.1),
        }

    def test_one_twist_rep(self):
        """Full cycle left → right → left → right = 2 reversals = 1 rep."""
        counter = ExerciseRepCounter("russian twist", fps=30,
                                     min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        # Start left
        for _ in range(3):
            counter.update(self._make_twist_landmarks(-50), frame_number=frame)
            frame += 1
        # Go right (peak tracked)
        for _ in range(3):
            counter.update(self._make_twist_landmarks(50), frame_number=frame)
            frame += 1
        # Go left (reversal #1: peak→valley, half rep)
        for _ in range(3):
            counter.update(self._make_twist_landmarks(-50), frame_number=frame)
            frame += 1
        # Go right again (reversal #2: valley→peak, full rep)
        for _ in range(3):
            counter.update(self._make_twist_landmarks(50), frame_number=frame)
            frame += 1

        assert counter.rep_count == 1

    def test_half_cycle_no_rep(self):
        """Only one reversal (left → right → left) = 0 reps."""
        counter = ExerciseRepCounter("russian twist", fps=30,
                                     min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        for _ in range(3):
            counter.update(self._make_twist_landmarks(-50), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(self._make_twist_landmarks(50), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(self._make_twist_landmarks(-50), frame_number=frame)
            frame += 1

        assert counter.rep_count == 0


class TestPlank:
    """Test plank hold duration tracking."""

    def test_hold_duration(self):
        counter = ExerciseRepCounter("plank", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        landmarks = make_squat_landmarks(170)

        # frames 0..90 = 91 frames, hold = 90 frames difference, 90/30 = 3s
        for i in range(91):
            counter.update(landmarks, frame_number=i)

        assert counter.rep_count == 3  # 3 seconds


class TestUnknownExercise:
    """Test that unsupported exercise names return 0 reps."""

    def test_unknown_exercise(self):
        counter = ExerciseRepCounter("some_unknown_exercise", fps=30)
        landmarks = make_squat_landmarks(160)
        for i in range(20):
            counter.update(landmarks, frame_number=i)
        assert counter.rep_count == 0


class TestReset:
    """Test counter reset."""

    def test_reset_clears_state(self):
        counter = ExerciseRepCounter("squat", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)

        frame = 0
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(80), frame_number=frame)
            frame += 1
        for _ in range(3):
            counter.update(make_squat_landmarks(160), frame_number=frame)
            frame += 1

        assert counter.rep_count == 1

        counter.reset()
        assert counter.rep_count == 0
        assert len(counter.reps) == 0
        assert counter.current_angle is None


class TestRepCounterManager:
    """Test the multi-exercise manager."""

    def test_switches_exercise(self):
        manager = RepCounterManager(fps=30, min_rep_duration_ms=0,
                                    angle_smoothing_window=1)

        frame = 0
        # 1 squat rep
        for _ in range(3):
            manager.update("squat", make_squat_landmarks(160), frame_number=frame)
            frame += 1
        for _ in range(3):
            manager.update("squat", make_squat_landmarks(80), frame_number=frame)
            frame += 1
        for _ in range(3):
            count, _ = manager.update("squat", make_squat_landmarks(160), frame_number=frame)
            frame += 1
        assert count == 1

        # Switch to curls
        for _ in range(3):
            manager.update("barbell biceps curl", make_elbow_landmarks(155), frame_number=frame)
            frame += 1
        for _ in range(3):
            manager.update("barbell biceps curl", make_elbow_landmarks(45), frame_number=frame)
            frame += 1
        for _ in range(3):
            count, _ = manager.update("barbell biceps curl", make_elbow_landmarks(155), frame_number=frame)
            frame += 1
        assert count == 1

        # Switch back — squat count preserved
        count, _ = manager.update("squat", make_squat_landmarks(160), frame_number=frame)
        assert count == 1

    def test_none_exercise_returns_last(self):
        manager = RepCounterManager(fps=30, min_rep_duration_ms=0,
                                    angle_smoothing_window=1)

        frame = 0
        for _ in range(3):
            manager.update("squat", make_squat_landmarks(160), frame_number=frame)
            frame += 1
        for _ in range(3):
            manager.update("squat", make_squat_landmarks(80), frame_number=frame)
            frame += 1
        for _ in range(3):
            manager.update("squat", make_squat_landmarks(160), frame_number=frame)
            frame += 1

        count, _ = manager.update(None, {}, frame_number=frame)
        assert count == 1

    def test_reset_clears_all(self):
        manager = RepCounterManager(fps=30, min_rep_duration_ms=0,
                                    angle_smoothing_window=1)
        manager.update("squat", make_squat_landmarks(160), frame_number=0)
        manager.reset()
        assert manager.current_counter is None


class TestAllExercisesHaveRules:
    """Verify all 22 model classes have rep counting rules."""

    EXPECTED_CLASSES = [
        "barbell biceps curl", "bench press", "chest fly machine",
        "deadlift", "decline bench press", "hammer curl", "hip thrust",
        "incline bench press", "lat pulldown", "lateral raises",
        "leg extension", "leg raises", "plank", "pull up", "push up",
        "romanian deadlift", "russian twist", "shoulder press", "squat",
        "t bar row", "tricep dips", "tricep pushdown",
    ]

    def test_all_classes_have_rules(self):
        for cls in self.EXPECTED_CLASSES:
            assert cls in REP_RULES, f"Missing rep rule for: {cls}"

    def test_rule_types_correct(self):
        for cls, rule in REP_RULES.items():
            assert isinstance(rule, (AngleRepRule, TwistRepRule, HoldRule)), \
                f"Invalid rule type for {cls}: {type(rule)}"

    def test_min_amplitude_positive(self):
        for cls, rule in REP_RULES.items():
            if isinstance(rule, (AngleRepRule, TwistRepRule)):
                assert rule.min_amplitude > 0, \
                    f"{cls}: min_amplitude must be positive"


class TestNameNormalization:
    """Test that hyphen/case variations in exercise names are handled."""

    def test_hyphen_normalized(self):
        """'push-up' (hyphen) should find the 'push up' (space) rule."""
        counter = ExerciseRepCounter("push-up", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)
        assert counter.rule is not None

    def test_case_normalized(self):
        """'Push Up' should find the 'push up' rule."""
        counter = ExerciseRepCounter("Push Up", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)
        assert counter.rule is not None

    def test_exact_match_works(self):
        """'push up' (exact) should find the rule."""
        counter = ExerciseRepCounter("push up", fps=30, min_rep_duration_ms=0,
                                     angle_smoothing_window=1)
        assert counter.rule is not None
