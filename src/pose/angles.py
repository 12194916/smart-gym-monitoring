"""Joint angle computation from pose landmarks.

Computes 9 key angles used for exercise classification and rep counting:
- left/right knee (hip-knee-ankle)
- left/right hip (shoulder-hip-knee)
- left/right elbow (shoulder-elbow-wrist)
- left/right shoulder (elbow-shoulder-hip)
- torso inclination (shoulder midpoint to hip midpoint vs vertical)
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict

from src.pose.estimator import LandmarkPoint
from src.utils.geometry import calculate_angle, torso_inclination


@dataclass
class JointAngles:
    """All 9 computed joint angles for a single frame."""
    left_knee: float
    right_knee: float
    left_hip: float
    right_hip: float
    left_elbow: float
    right_elbow: float
    left_shoulder: float
    right_shoulder: float
    torso_inclination: float
    timestamp_ms: int


def _to_xy(lm: LandmarkPoint) -> np.ndarray:
    """Extract [x, y] as numpy array from a landmark."""
    return np.array([lm.x, lm.y])


def compute_joint_angles(landmarks: Dict[str, LandmarkPoint], timestamp_ms: int) -> JointAngles:
    """Compute 9 key joint angles from body landmarks.

    Args:
        landmarks: Dict mapping landmark name to LandmarkPoint.
        timestamp_ms: Frame timestamp in milliseconds.

    Returns:
        JointAngles dataclass with all 9 angles in degrees.
    """
    # Extract points
    ls = _to_xy(landmarks["left_shoulder"])
    rs = _to_xy(landmarks["right_shoulder"])
    le = _to_xy(landmarks["left_elbow"])
    re = _to_xy(landmarks["right_elbow"])
    lw = _to_xy(landmarks["left_wrist"])
    rw = _to_xy(landmarks["right_wrist"])
    lh = _to_xy(landmarks["left_hip"])
    rh = _to_xy(landmarks["right_hip"])
    lk = _to_xy(landmarks["left_knee"])
    rk = _to_xy(landmarks["right_knee"])
    la = _to_xy(landmarks["left_ankle"])
    ra = _to_xy(landmarks["right_ankle"])

    return JointAngles(
        # Knee angles: hip-knee-ankle
        left_knee=calculate_angle(lh, lk, la),
        right_knee=calculate_angle(rh, rk, ra),
        # Hip angles: shoulder-hip-knee
        left_hip=calculate_angle(ls, lh, lk),
        right_hip=calculate_angle(rs, rh, rk),
        # Elbow angles: shoulder-elbow-wrist
        left_elbow=calculate_angle(ls, le, lw),
        right_elbow=calculate_angle(rs, re, rw),
        # Shoulder angles: elbow-shoulder-hip
        left_shoulder=calculate_angle(le, ls, lh),
        right_shoulder=calculate_angle(re, rs, rh),
        # Torso inclination
        torso_inclination=torso_inclination(
            shoulder_mid=(ls + rs) / 2,
            hip_mid=(lh + rh) / 2,
        ),
        timestamp_ms=timestamp_ms,
    )
