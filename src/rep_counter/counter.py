"""State-machine repetition counter.

Two-state cycle: EXTENDED -> CONTRACTED -> EXTENDED = 1 rep.
Each exercise defines which angle to track and the thresholds
for each state transition (see rep_rules.py).
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from src.exercise.definitions import Exercise
from src.pose.angles import JointAngles
from src.rep_counter.rep_rules import REP_RULES, get_tracked_angle


class RepState(Enum):
    EXTENDED = "extended"
    CONTRACTED = "contracted"


@dataclass
class RepEvent:
    rep_number: int
    start_ms: int
    end_ms: int
    duration_ms: int


class RepCounter:
    """Counts repetitions using a two-state machine (EXTENDED <-> CONTRACTED)."""

    def __init__(self):
        self.state = RepState.EXTENDED
        self.count: int = 0
        self.reps: List[RepEvent] = []
        self.current_exercise: Optional[Exercise] = None
        self._rep_start_ms: Optional[int] = None
        self._last_tracked_value: Optional[float] = None

    def update(self, exercise: Exercise, angles: JointAngles,
               timestamp_ms: int) -> Optional[RepEvent]:
        """Feed new angle data to the state machine.

        Automatically resets if the exercise changes. Returns a RepEvent
        when a full rep cycle completes, None otherwise.
        """
        # Auto-reset on exercise change
        if exercise != self.current_exercise:
            self.reset(exercise)

        # No rep rules for this exercise (e.g. UNKNOWN, plank)
        if exercise not in REP_RULES:
            return None

        rule = REP_RULES[exercise]
        value = get_tracked_angle(angles, rule["track_angle"])
        self._last_tracked_value = value

        extended_thresh = rule["extended_threshold"]
        contracted_thresh = rule["contracted_threshold"]
        min_duration = rule["min_rep_duration_ms"]

        event = None

        if self.state == RepState.EXTENDED:
            if value < contracted_thresh:
                # Transition to CONTRACTED — start of a potential rep
                self.state = RepState.CONTRACTED
                self._rep_start_ms = timestamp_ms

        elif self.state == RepState.CONTRACTED:
            if value > extended_thresh:
                # Transition back to EXTENDED — rep completed if long enough
                if self._rep_start_ms is not None:
                    duration = timestamp_ms - self._rep_start_ms
                    if duration >= min_duration:
                        self.count += 1
                        event = RepEvent(
                            rep_number=self.count,
                            start_ms=self._rep_start_ms,
                            end_ms=timestamp_ms,
                            duration_ms=duration,
                        )
                        self.reps.append(event)
                self.state = RepState.EXTENDED
                self._rep_start_ms = None

        return event

    def reset(self, exercise: Optional[Exercise] = None):
        """Reset the counter, optionally for a new exercise."""
        self.state = RepState.EXTENDED
        self.count = 0
        self.reps = []
        self.current_exercise = exercise
        self._rep_start_ms = None
        self._last_tracked_value = None

    @property
    def tracked_value(self) -> Optional[float]:
        """Last tracked angle value in degrees, for display purposes."""
        return self._last_tracked_value
