"""MediaPipe PoseLandmarker wrapper (Tasks API).

Extracts 33 pose landmarks from a frame. When multiple people are present,
selects the best candidate by body size and center proximity — no YOLO needed.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

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

    When num_poses > 1, selects the best person using:
    - Body size (bounding box area from visible landmarks) — prefers larger
    - Center proximity (distance of body center to frame center) — prefers closer
    """

    def __init__(self, model_path: str, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, num_poses: int = 1,
                 size_weight: float = 0.6, center_weight: float = 0.4):
        self.num_poses = num_poses
        self.size_weight = size_weight
        self.center_weight = center_weight

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            num_poses=num_poses,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _select_best_pose(self, poses: list) -> int:
        """Select the best pose by body size + center proximity.

        Uses normalized [0,1] coordinates from MediaPipe landmarks.
        Frame center is (0.5, 0.5) in normalized space.

        Returns the index of the best pose.
        """
        if len(poses) == 1:
            return 0

        best_idx = 0
        best_score = -1.0
        vis_threshold = 0.3

        areas = []
        for pose in poses:
            xs = [lm.x for lm in pose if (lm.visibility or 0) > vis_threshold]
            ys = [lm.y for lm in pose if (lm.visibility or 0) > vis_threshold]
            if xs and ys:
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            else:
                area = 0.0
            areas.append(area)

        max_area = max(areas) if areas else 1.0

        for i, pose in enumerate(poses):
            # Size score: normalized by largest detected pose
            size_score = areas[i] / max_area if max_area > 0 else 0.0

            # Center score: distance of body center to frame center (0.5, 0.5)
            xs = [lm.x for lm in pose if (lm.visibility or 0) > vis_threshold]
            ys = [lm.y for lm in pose if (lm.visibility or 0) > vis_threshold]
            if xs and ys:
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
            else:
                cx, cy = 0.5, 0.5

            # Max possible distance from center in normalized space is ~0.707
            dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)
            max_dist = np.sqrt(0.5)
            center_score = 1.0 - (dist / max_dist)

            score = self.size_weight * size_score + self.center_weight * center_score

            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

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

        best_idx = self._select_best_pose(result.pose_landmarks)
        pose = result.pose_landmarks[best_idx]

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

    def process_full_frame(self, frame: np.ndarray):
        """Process a full frame and return all 33 raw normalized landmarks.

        Returns a list of 33*4 floats (x, y, z, visibility) in MediaPipe's
        normalized [0,1] coordinate space, or None if no pose detected.
        Also returns the 12 body landmarks dict for angle computation.

        When multiple people are detected, selects the best candidate
        by body size and center proximity.
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.landmarker.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return None, None

        best_idx = self._select_best_pose(result.pose_landmarks)
        pose = result.pose_landmarks[best_idx]

        # All 33 raw normalized landmarks for classification
        raw_landmarks = []
        for lm in pose:
            raw_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility if lm.visibility else 0.0])

        # 12 body landmarks in pixel coords for angles and skeleton
        body_landmarks = {}
        for idx, name in BODY_LANDMARK_INDICES.items():
            lm = pose[idx]
            body_landmarks[name] = LandmarkPoint(
                x=lm.x * w, y=lm.y * h, z=lm.z, visibility=lm.visibility,
            )

        return raw_landmarks, body_landmarks

    def close(self):
        """Release resources."""
        self.landmarker.close()
