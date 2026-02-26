"""Peak-valley rep counter for gym exercises.

Counts reps by detecting direction reversals in the tracked joint angle.
Unlike fixed-threshold approaches, this works regardless of camera angle
because it tracks RELATIVE changes, not absolute values.

Algorithm (for angle-based exercises):
  1. Smooth the raw angle with a moving average
  2. Track whether angle is currently rising ("up") or falling ("down")
  3. When rising and angle drops >= min_amplitude below the peak → reversal
  4. When falling and angle rises >= min_amplitude above the valley → reversal
  5. Every two reversals (peak→valley→peak or valley→peak→valley) = 1 rep

Special cases:
  - Russian twist: same peak-valley on horizontal wrist offset
  - Plank: static hold, reports duration in seconds
"""

import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.utils.geometry import calculate_angle
from src.rep_counter.rep_rules import (
    REP_RULES, AngleRepRule, TwistRepRule, HoldRule,
)


@dataclass
class RepEvent:
    """A single counted rep with metadata."""
    rep_number: int
    frame_number: int
    timestamp_ms: float
    angle_at_count: float


class ExerciseRepCounter:
    """Rep counter using peak-valley detection.

    Tracks the smoothed angle signal and detects direction reversals.
    A reversal is confirmed when the signal moves >= min_amplitude away
    from the last extreme point. Two reversals = 1 rep.

    This is robust to different camera angles because it only cares about
    the AMPLITUDE of the angle change, not the absolute angle values.
    """

    def __init__(self, exercise: str, fps: float = 30.0,
                 min_rep_duration_ms: float = 800,
                 angle_smoothing_window: int = 5):
        self.exercise = exercise.lower().replace("-", " ").strip()
        self.fps = fps

        self.rule = REP_RULES.get(self.exercise)

        # Peak-valley state
        self._direction: Optional[str] = None  # "up" or "down"
        self._extreme_angle: float = 0.0       # last peak or valley value
        self._half_rep_done: bool = False
        self._rep_count: int = 0
        self._reps: List[RepEvent] = []

        # Angle smoothing (moving average)
        self._angle_buffer: deque = deque(maxlen=angle_smoothing_window)

        # Timing
        self._frame_count: int = 0
        self._last_reversal_frame: int = 0
        self._min_half_rep_frames: int = max(
            1, int(min_rep_duration_ms / 2000.0 * fps)
        )

        # Plank hold state
        self._hold_start_frame: Optional[int] = None

    @property
    def rep_count(self) -> int:
        return self._rep_count

    @property
    def reps(self) -> List[RepEvent]:
        return list(self._reps)

    @property
    def current_angle(self) -> Optional[float]:
        """Current smoothed angle (or offset for twist)."""
        if self._angle_buffer:
            return float(np.mean(self._angle_buffer))
        return None

    def update(self, body_landmarks: Dict,
               frame_number: Optional[int] = None) -> int:
        """Process one frame and return current rep count."""
        if frame_number is not None:
            self._frame_count = frame_number
        else:
            self._frame_count += 1

        if self.rule is None:
            return self._rep_count

        if isinstance(self.rule, AngleRepRule):
            return self._update_angle(body_landmarks)
        elif isinstance(self.rule, TwistRepRule):
            return self._update_peak_valley_signal(
                self._get_twist_signal(body_landmarks)
            )
        elif isinstance(self.rule, HoldRule):
            return self._update_hold(body_landmarks)

        return self._rep_count

    # ------------------------------------------------------------------
    # Angle computation
    # ------------------------------------------------------------------

    def _compute_angle(self, body_landmarks: Dict,
                       joints: Tuple[str, str, str]) -> Optional[float]:
        """Compute angle at the middle joint. None if landmarks not visible."""
        a_name, b_name, c_name = joints

        for name in (a_name, b_name, c_name):
            if name not in body_landmarks:
                return None

        a = body_landmarks[a_name]
        b = body_landmarks[b_name]
        c = body_landmarks[c_name]

        if a.visibility < 0.3 or b.visibility < 0.3 or c.visibility < 0.3:
            return None

        pt_a = np.array([a.x, a.y])
        pt_b = np.array([b.x, b.y])
        pt_c = np.array([c.x, c.y])

        return calculate_angle(pt_a, pt_b, pt_c)

    def _get_tracked_angle(self, body_landmarks: Dict) -> Optional[float]:
        """Average left/right side angles. Falls back to whichever is visible."""
        rule = self.rule
        left = self._compute_angle(body_landmarks, rule.left_joints)
        right = self._compute_angle(body_landmarks, rule.right_joints)

        if left is not None and right is not None:
            return (left + right) / 2.0
        return left if left is not None else right

    # ------------------------------------------------------------------
    # Peak-valley detection (used by angle and twist exercises)
    # ------------------------------------------------------------------

    def _update_peak_valley_signal(self, raw_value: Optional[float]) -> int:
        """Core peak-valley detection on any 1D signal.

        Tracks direction and detects reversals when the signal moves
        >= min_amplitude from the last extreme point.
        """
        if raw_value is None:
            return self._rep_count

        min_amp = self.rule.min_amplitude

        # Smooth
        self._angle_buffer.append(raw_value)
        value = float(np.mean(self._angle_buffer))

        # Initialize direction on first value
        if self._direction is None:
            self._extreme_angle = value
            self._direction = "up"  # arbitrary, corrects itself
            self._last_reversal_frame = self._frame_count
            return self._rep_count

        if self._direction == "up":
            # Track the peak
            if value > self._extreme_angle:
                self._extreme_angle = value
            # Check for reversal: dropped enough from peak?
            elif self._extreme_angle - value >= min_amp:
                self._on_reversal(value)
                self._direction = "down"
                self._extreme_angle = value

        elif self._direction == "down":
            # Track the valley
            if value < self._extreme_angle:
                self._extreme_angle = value
            # Check for reversal: risen enough from valley?
            elif value - self._extreme_angle >= min_amp:
                self._on_reversal(value)
                self._direction = "up"
                self._extreme_angle = value

        return self._rep_count

    def _on_reversal(self, angle: float):
        """Handle a confirmed direction reversal."""
        # Enforce minimum time between reversals
        if (self._frame_count - self._last_reversal_frame) < self._min_half_rep_frames:
            return

        if self._half_rep_done:
            # Second reversal → count rep
            self._rep_count += 1
            self._reps.append(RepEvent(
                rep_number=self._rep_count,
                frame_number=self._frame_count,
                timestamp_ms=self._frame_count / self.fps * 1000,
                angle_at_count=angle,
            ))
            self._half_rep_done = False
        else:
            # First reversal
            self._half_rep_done = True

        self._last_reversal_frame = self._frame_count

    # ------------------------------------------------------------------
    # Angle-based exercises (20 of 22)
    # ------------------------------------------------------------------

    def _update_angle(self, body_landmarks: Dict) -> int:
        raw_angle = self._get_tracked_angle(body_landmarks)
        return self._update_peak_valley_signal(raw_angle)

    # ------------------------------------------------------------------
    # Russian twist (position-based)
    # ------------------------------------------------------------------

    def _get_twist_signal(self, body_landmarks: Dict) -> Optional[float]:
        """Compute normalized horizontal wrist offset from shoulder center."""
        required = ["left_wrist", "right_wrist", "left_shoulder", "right_shoulder"]
        if not all(name in body_landmarks for name in required):
            return None

        lw = body_landmarks["left_wrist"]
        rw = body_landmarks["right_wrist"]
        ls = body_landmarks["left_shoulder"]
        rs = body_landmarks["right_shoulder"]

        if any(lm.visibility < 0.3 for lm in [lw, rw, ls, rs]):
            return None

        wrist_mid_x = (lw.x + rw.x) / 2.0
        shoulder_mid_x = (ls.x + rs.x) / 2.0
        shoulder_width = abs(ls.x - rs.x)

        if shoulder_width < 1e-6:
            return None

        return (wrist_mid_x - shoulder_mid_x) / shoulder_width

    # ------------------------------------------------------------------
    # Plank (hold duration)
    # ------------------------------------------------------------------

    def _update_hold(self, body_landmarks: Dict) -> int:
        """Returns hold duration in whole seconds as rep_count."""
        if body_landmarks:
            if self._hold_start_frame is None:
                self._hold_start_frame = self._frame_count
            hold_frames = self._frame_count - self._hold_start_frame
            self._rep_count = max(1, int(hold_frames / self.fps))
        return self._rep_count

    # ------------------------------------------------------------------

    def reset(self):
        self._direction = None
        self._extreme_angle = 0.0
        self._half_rep_done = False
        self._rep_count = 0
        self._reps.clear()
        self._angle_buffer.clear()
        self._frame_count = 0
        self._last_reversal_frame = 0
        self._hold_start_frame = None


class RepCounterManager:
    """Manages rep counters across exercise changes.

    When the detected exercise label changes, switches to the counter
    for that exercise (creating one if needed). Old counters are preserved
    so reps accumulate if the label switches back.
    """

    def __init__(self, fps: float = 30.0, min_rep_duration_ms: float = 800,
                 angle_smoothing_window: int = 5):
        self.fps = fps
        self.min_rep_duration_ms = min_rep_duration_ms
        self.angle_smoothing_window = angle_smoothing_window

        self._current_exercise: Optional[str] = None
        self._counter: Optional[ExerciseRepCounter] = None
        self._history: Dict[str, ExerciseRepCounter] = {}

    def update(self, exercise: Optional[str], body_landmarks: Dict,
               frame_number: Optional[int] = None) -> Tuple[int, Optional[float]]:
        """Update with current exercise label and landmarks.

        Returns:
            (rep_count, current_smoothed_angle) for the current exercise.
        """
        if exercise is None or exercise == "...":
            if self._counter is not None:
                return self._counter.rep_count, self._counter.current_angle
            return 0, None

        # Exercise changed → switch counter
        if exercise != self._current_exercise:
            if self._current_exercise is not None and self._counter is not None:
                self._history[self._current_exercise] = self._counter

            if exercise in self._history:
                self._counter = self._history[exercise]
            else:
                self._counter = ExerciseRepCounter(
                    exercise=exercise,
                    fps=self.fps,
                    min_rep_duration_ms=self.min_rep_duration_ms,
                    angle_smoothing_window=self.angle_smoothing_window,
                )
            self._current_exercise = exercise

        if self._counter is not None:
            self._counter.update(body_landmarks, frame_number)
            return self._counter.rep_count, self._counter.current_angle

        return 0, None

    @property
    def current_counter(self) -> Optional[ExerciseRepCounter]:
        return self._counter

    def reset(self):
        self._current_exercise = None
        self._counter = None
        self._history.clear()
