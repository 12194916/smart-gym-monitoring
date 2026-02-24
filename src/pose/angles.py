"""Joint angle computation from pose landmarks."""

from dataclasses import dataclass


@dataclass
class JointAngles:
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


def compute_joint_angles(landmarks: dict, timestamp_ms: int) -> JointAngles:
    """Compute 9 key joint angles from MediaPipe landmarks."""
    raise NotImplementedError
