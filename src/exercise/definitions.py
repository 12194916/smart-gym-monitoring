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
    LEG_RAISES = "leg_raises"
    LEG_EXTENSION = "leg_extension"
    HIP_THRUST = "hip_thrust"
    CHEST_FLY = "chest_fly"
    TRICEP_PUSHDOWN = "tricep_pushdown"
    TRICEP_DIPS = "tricep_dips"
    T_BAR_ROW = "t_bar_row"
    UNKNOWN = "unknown"
