"""Vector math and angle calculation utilities."""

import numpy as np


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Calculate angle at point b formed by vectors ba and bc, in degrees."""
    raise NotImplementedError


def torso_inclination(shoulder_mid: np.ndarray, hip_mid: np.ndarray) -> float:
    """Calculate torso inclination angle relative to vertical, in degrees."""
    raise NotImplementedError
