"""Draw pose skeleton and overlays on video frames.

Renders skeleton connections, joint dots, and a HUD panel with exercise
label, rep count, confidence, tracked angle, and FPS.
"""

import cv2
import numpy as np
from typing import Dict, Optional

from src.pose.estimator import LandmarkPoint

# Anatomical connections between the 12 body landmarks
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Colors (BGR)
SKELETON_COLOR = (0, 255, 128)
JOINT_COLOR = (0, 200, 255)
TEXT_BG_COLOR = (40, 40, 40)
TEXT_COLOR = (255, 255, 255)
ACCENT_COLOR = (0, 255, 255)
REP_COLOR = (50, 205, 50)
NO_DETECT_COLOR = (0, 0, 255)


class VideoAnnotator:
    """Draws skeleton, labels, and rep counter on video frames."""

    def __init__(self, draw_skeleton: bool = True, draw_angles: bool = True,
                 draw_labels: bool = True):
        self.draw_skeleton_flag = draw_skeleton
        self.draw_angles_flag = draw_angles
        self.draw_labels_flag = draw_labels

    def draw(self, frame: np.ndarray,
             landmarks: Optional[Dict[str, LandmarkPoint]] = None,
             exercise: str = "unknown",
             rep_count: int = 0,
             confidence: float = 0.0,
             tracked_angle_value: Optional[float] = None,
             tracked_angle_name: Optional[str] = None,
             fps: Optional[float] = None,
             user_id: Optional[str] = None) -> np.ndarray:
        """Draw all overlays on a frame and return the annotated copy."""
        annotated = frame.copy()

        if landmarks and self.draw_skeleton_flag:
            self._draw_skeleton(annotated, landmarks)

        if self.draw_labels_flag:
            self._draw_hud(annotated, exercise, rep_count, confidence,
                           tracked_angle_value, tracked_angle_name, fps)

        if landmarks is None:
            self._draw_no_detection(annotated)

        return annotated

    def _draw_skeleton(self, frame: np.ndarray,
                       landmarks: Dict[str, LandmarkPoint]):
        """Draw bone connections and joint circles."""
        # Draw lines between connected joints
        for start_name, end_name in SKELETON_CONNECTIONS:
            if start_name in landmarks and end_name in landmarks:
                pt1 = (int(landmarks[start_name].x), int(landmarks[start_name].y))
                pt2 = (int(landmarks[end_name].x), int(landmarks[end_name].y))
                cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2, cv2.LINE_AA)

        # Draw joint dots
        for name, lm in landmarks.items():
            center = (int(lm.x), int(lm.y))
            cv2.circle(frame, center, 5, JOINT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, center, 5, (0, 0, 0), 1, cv2.LINE_AA)

    def _draw_hud(self, frame: np.ndarray, exercise: str, rep_count: int,
                  confidence: float, tracked_angle_value: Optional[float],
                  tracked_angle_name: Optional[str], fps: Optional[float]):
        """Draw the heads-up display panel in the top-left corner."""
        margin = 15
        y = 30

        # Calculate panel height
        panel_h = 110
        if tracked_angle_value is not None:
            panel_h += 25
        if fps is not None:
            panel_h += 25

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (330, panel_h), TEXT_BG_COLOR, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Exercise label
        label_text = exercise.upper().replace("_", " ")
        cv2.putText(frame, label_text, (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT_COLOR, 2, cv2.LINE_AA)
        y += 30

        # Confidence
        cv2.putText(frame, f"Confidence: {confidence:.0%}", (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 25

        # Rep count
        cv2.putText(frame, f"Reps: {rep_count}", (margin, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, REP_COLOR, 2, cv2.LINE_AA)
        y += 28

        # Tracked angle
        if tracked_angle_value is not None and tracked_angle_name:
            cv2.putText(frame, f"{tracked_angle_name}: {tracked_angle_value:.0f} deg",
                        (margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        TEXT_COLOR, 1, cv2.LINE_AA)
            y += 25

        # FPS
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.0f}", (margin, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180),
                        1, cv2.LINE_AA)

    def _draw_no_detection(self, frame: np.ndarray):
        """Show a warning when no person is detected."""
        h, w = frame.shape[:2]
        text = "No person detected"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (w - text_size[0]) // 2
        y = h // 2
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    NO_DETECT_COLOR, 2, cv2.LINE_AA)
