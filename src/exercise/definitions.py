"""Exercise enum and classification definitions."""

from enum import Enum


class Exercise(Enum):
    SQUAT = "squat"
    PUSHUP = "pushup"
    DEADLIFT = "deadlift"
    BICEP_CURL = "bicep_curl"
    LUNGE = "lunge"
    SHOULDER_PRESS = "shoulder_press"
    PULLUP = "pullup"
    BENCH_PRESS = "bench_press"
    UNKNOWN = "unknown"
