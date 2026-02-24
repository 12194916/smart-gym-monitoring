"""Per-exercise repetition detection rules and thresholds."""

from src.exercise.definitions import Exercise

REP_RULES = {
    Exercise.SQUAT: {
        "track_angle": "avg_knee",
        "extended_threshold": 160,
        "contracted_threshold": 100,
        "min_rep_duration_ms": 800,
    },
    Exercise.BICEP_CURL: {
        "track_angle": "avg_elbow",
        "extended_threshold": 150,
        "contracted_threshold": 60,
        "min_rep_duration_ms": 600,
    },
    Exercise.PUSHUP: {
        "track_angle": "avg_elbow",
        "extended_threshold": 155,
        "contracted_threshold": 80,
        "min_rep_duration_ms": 700,
    },
    Exercise.DEADLIFT: {
        "track_angle": "avg_hip",
        "extended_threshold": 165,
        "contracted_threshold": 100,
        "min_rep_duration_ms": 1000,
    },
    Exercise.SHOULDER_PRESS: {
        "track_angle": "avg_elbow",
        "extended_threshold": 160,
        "contracted_threshold": 80,
        "min_rep_duration_ms": 700,
    },
    Exercise.LUNGE: {
        "track_angle": "min_knee",
        "extended_threshold": 155,
        "contracted_threshold": 95,
        "min_rep_duration_ms": 900,
    },
    Exercise.PULLUP: {
        "track_angle": "avg_elbow",
        "extended_threshold": 160,
        "contracted_threshold": 70,
        "min_rep_duration_ms": 800,
    },
    Exercise.BENCH_PRESS: {
        "track_angle": "avg_elbow",
        "extended_threshold": 160,
        "contracted_threshold": 75,
        "min_rep_duration_ms": 800,
    },
}
