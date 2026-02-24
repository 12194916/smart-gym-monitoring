"""Vector math and angle calculation utilities."""

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b formed by vectors ba and bc, in degrees.

    Args:
        a: First point (e.g. hip).
        b: Vertex point where angle is measured (e.g. knee).
        c: Third point (e.g. ankle).

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)


def torso_inclination(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Calculate torso inclination angle relative to vertical, in degrees.

    0 = perfectly upright, 90 = horizontal (bent over).

    Args:
        shoulder_mid: Midpoint between left and right shoulders [x, y].
        hip_mid: Midpoint between left and right hips [x, y].

    Returns:
        Inclination angle in degrees [0, 180].
    """
    torso_vec = shoulder_mid - hip_mid  # points upward from hips to shoulders
    vertical = np.array([0.0, -1.0])  # up in image coordinates (y decreases upward)

    cos_angle = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return float(angle)
