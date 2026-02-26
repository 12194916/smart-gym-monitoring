"""Per-exercise rep counting rules.

Uses peak-valley detection instead of fixed absolute thresholds.
This works regardless of camera angle because it tracks RELATIVE
angle changes, not absolute values.

Algorithm:
  - Track the smoothed joint angle over time
  - Detect direction reversals (peaks and valleys)
  - A reversal is confirmed when the angle changes by >= min_amplitude
    from the last extreme point
  - Two confirmed reversals (peak → valley → peak) = 1 rep

Each exercise defines:
  - Which joint angle to track (3-point chain, angle at middle point)
  - min_amplitude: minimum angle change (degrees) to confirm a reversal
    Larger = fewer false positives, smaller = more sensitive

MediaPipe body landmarks used:
  11/12: left/right shoulder    13/14: left/right elbow
  15/16: left/right wrist       23/24: left/right hip
  25/26: left/right knee        27/28: left/right ankle
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Union


@dataclass(frozen=True)
class AngleRepRule:
    """Angle-based rep rule — tracks peak-valley cycles in joint angle.

    min_amplitude: the angle must change by at least this many degrees
    from the last peak/valley to confirm a direction reversal.
    """
    left_joints: Tuple[str, str, str]   # (a, b, c) — angle measured at b
    right_joints: Tuple[str, str, str]
    min_amplitude: float                # degrees — minimum change to confirm reversal


@dataclass(frozen=True)
class TwistRepRule:
    """Position-based rule for rotational exercises (Russian twist).

    Tracks horizontal wrist-midpoint offset relative to shoulder center,
    normalized by shoulder width. Uses same peak-valley detection.
    """
    min_amplitude: float = 0.3  # normalized offset change to confirm reversal


@dataclass(frozen=True)
class HoldRule:
    """Static hold exercise (plank). Reports hold duration, not reps."""
    pass


RepRuleType = Union[AngleRepRule, TwistRepRule, HoldRule]

# ---------------------------------------------------------------------------
# All 22 exercise classes
#
# min_amplitude guide:
#   - Large ROM exercises (squat, curl, deadlift): 25-35°
#   - Medium ROM exercises (bench press, shoulder press): 20-30°
#   - Small ROM / machine exercises (leg ext, fly, lateral raise): 12-20°
# ---------------------------------------------------------------------------

REP_RULES: Dict[str, RepRuleType] = {

    # ===== LEG EXERCISES =====
    # Squat: knee angle (hip → knee → ankle)
    # Full ROM ~60° change, min_amplitude catches partial reps too
    "squat": AngleRepRule(
        left_joints=("left_hip", "left_knee", "left_ankle"),
        right_joints=("right_hip", "right_knee", "right_ankle"),
        min_amplitude=25,
    ),

    # Leg extension: knee angle (hip → knee → ankle)
    # Machine-limited ROM, smaller angle change in 2D
    "leg extension": AngleRepRule(
        left_joints=("left_hip", "left_knee", "left_ankle"),
        right_joints=("right_hip", "right_knee", "right_ankle"),
        min_amplitude=15,
    ),

    # Hip thrust: hip angle (shoulder → hip → knee)
    "hip thrust": AngleRepRule(
        left_joints=("left_shoulder", "left_hip", "left_knee"),
        right_joints=("right_shoulder", "right_hip", "right_knee"),
        min_amplitude=20,
    ),

    # Leg raises: hip flexion angle (shoulder → hip → ankle)
    "leg raises": AngleRepRule(
        left_joints=("left_shoulder", "left_hip", "left_ankle"),
        right_joints=("right_shoulder", "right_hip", "right_ankle"),
        min_amplitude=20,
    ),

    # ===== HIP HINGE / BACK =====
    # Deadlift: hip angle (shoulder → hip → knee)
    "deadlift": AngleRepRule(
        left_joints=("left_shoulder", "left_hip", "left_knee"),
        right_joints=("right_shoulder", "right_hip", "right_knee"),
        min_amplitude=25,
    ),

    # Romanian deadlift: same joints, slightly less ROM
    "romanian deadlift": AngleRepRule(
        left_joints=("left_shoulder", "left_hip", "left_knee"),
        right_joints=("right_shoulder", "right_hip", "right_knee"),
        min_amplitude=20,
    ),

    # T-bar row: elbow angle (shoulder → elbow → wrist)
    "t bar row": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # ===== CHEST / PUSH =====
    # Bench press: elbow angle (shoulder → elbow → wrist)
    "bench press": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    "decline bench press": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    "incline bench press": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # Chest fly machine: shoulder abduction (hip → shoulder → elbow)
    # Smaller ROM in 2D projection
    "chest fly machine": AngleRepRule(
        left_joints=("left_hip", "left_shoulder", "left_elbow"),
        right_joints=("right_hip", "right_shoulder", "right_elbow"),
        min_amplitude=12,
    ),

    # Push-up: elbow angle (shoulder → elbow → wrist)
    "push up": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # ===== SHOULDERS =====
    # Shoulder press: elbow angle (shoulder → elbow → wrist)
    "shoulder press": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # Lateral raise: shoulder abduction (hip → shoulder → wrist)
    "lateral raises": AngleRepRule(
        left_joints=("left_hip", "left_shoulder", "left_wrist"),
        right_joints=("right_hip", "right_shoulder", "right_wrist"),
        min_amplitude=15,
    ),

    # ===== ARMS — BICEPS =====
    # Barbell biceps curl: elbow angle (shoulder → elbow → wrist)
    "barbell biceps curl": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=25,
    ),

    "hammer curl": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=25,
    ),

    # ===== ARMS — TRICEPS =====
    # Tricep pushdown: elbow angle (shoulder → elbow → wrist)
    "tricep pushdown": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    "tricep dips": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # ===== BACK — PULL =====
    # Lat pulldown: elbow angle (shoulder → elbow → wrist)
    "lat pulldown": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # Pull up: elbow angle (shoulder → elbow → wrist)
    "pull up": AngleRepRule(
        left_joints=("left_shoulder", "left_elbow", "left_wrist"),
        right_joints=("right_shoulder", "right_elbow", "right_wrist"),
        min_amplitude=20,
    ),

    # ===== SPECIAL =====
    # Russian twist: wrist horizontal offset from torso center
    "russian twist": TwistRepRule(min_amplitude=0.3),

    # Plank: static hold — track duration, not reps
    "plank": HoldRule(),
}
