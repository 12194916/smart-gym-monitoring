"""MediaPipe PoseLandmarker wrapper (Tasks API).

Extracts 12 body landmarks (skipping face, hands, feet) from a cropped
person frame. Landmarks are returned in pixel coordinates relative to
the full frame using the provided offset.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Dict

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)


@dataclass
class LandmarkPoint:
    """A single pose landmark in pixel coordinates."""
    x: float  # pixel x in full frame
    y: float  # pixel y in full frame
    z: float  # relative depth
    visibility: float  # confidence that landmark is visible


# MediaPipe landmark indices for the body landmarks we use.
# Skipping: face (0-10), hands (17-22), feet (29-32)
BODY_LANDMARK_INDICES = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


class PoseEstimator:
    """Wraps MediaPipe PoseLandmarker for frame-by-frame pose extraction.

    Uses IMAGE mode so each frame is processed independently — this is
    simpler and avoids tracking issues when the crop region shifts between
    frames.
    """

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray, offset_x: int = 0,
                offset_y: int = 0) -> Optional[Dict[str, LandmarkPoint]]:
        """Process a single frame and return body landmarks dict, or None.

        Args:
            frame: BGR frame (full frame or crop).
            offset_x: X offset to convert crop coords to full-frame coords.
            offset_y: Y offset to convert crop coords to full-frame coords.

        Returns:
            Dict mapping landmark name to LandmarkPoint in full-frame pixel
            coordinates, or None if no pose detected.
        """
        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None

        # Take the first (and only) detected pose
        pose = result.pose_landmarks[0]

        landmarks = {}
        for idx, name in BODY_LANDMARK_INDICES.items():
            lm = pose[idx]
            # Convert normalized [0,1] coords to pixel coords, then add offset
            px = lm.x * w + offset_x
            py = lm.y * h + offset_y
            landmarks[name] = LandmarkPoint(
                x=px, y=py, z=lm.z, visibility=lm.visibility,
            )

        return landmarks

    def close(self):
        """Release resources."""
        self.landmarker.close()
