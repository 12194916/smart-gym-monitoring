"""State-machine repetition counter."""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from src.exercise.definitions import Exercise
from src.pose.angles import JointAngles


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

    def update(self, exercise: Exercise, angles: JointAngles, timestamp_ms: int):
        """Feed new angle data to the state machine. Increments count on full rep cycle."""
        raise NotImplementedError

    def reset(self, exercise: Optional[Exercise] = None):
        """Reset the counter, optionally for a new exercise."""
        raise NotImplementedError
