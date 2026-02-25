"""Live / video exercise prediction using raw landmark classifier.

Runs the full pipeline: YOLO person detection → MediaPipe full-frame pose →
raw landmark classifier → smoothed label overlay.

Usage:
    python scripts/predict.py --source 0                    # webcam
    python scripts/predict.py --source data/videos/test/squat/video1.mp4
    python scripts/predict.py --source path/to/video.mp4 --no-yolo
"""

import argparse
import os
import sys
from collections import deque, Counter

import cv2
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.pose.estimator import PoseEstimator, BODY_LANDMARK_INDICES


# Skeleton connections for drawing (pairs of landmark names)
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


def draw_skeleton(frame, body_landmarks):
    """Draw pose skeleton on the frame."""
    if body_landmarks is None:
        return

    # Draw connections
    for name_a, name_b in SKELETON_CONNECTIONS:
        if name_a in body_landmarks and name_b in body_landmarks:
            a = body_landmarks[name_a]
            b = body_landmarks[name_b]
            if a.visibility > 0.5 and b.visibility > 0.5:
                pt_a = (int(a.x), int(a.y))
                pt_b = (int(b.x), int(b.y))
                cv2.line(frame, pt_a, pt_b, (0, 255, 0), 2)

    # Draw keypoints
    for name, lm in body_landmarks.items():
        if lm.visibility > 0.5:
            cv2.circle(frame, (int(lm.x), int(lm.y)), 4, (0, 0, 255), -1)


def draw_label(frame, label, confidence):
    """Draw exercise label and confidence on the frame."""
    text = f"{label} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background rectangle
    (tw, th), baseline = cv2.getTextSize(text, font, 1.0, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + th), font, 1.0, (0, 255, 255), 2)


def draw_bbox(frame, box):
    """Draw YOLO person bounding box."""
    cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), (255, 200, 0), 2)
    label = f"person {box.confidence:.0%}"
    cv2.putText(frame, label, (box.x1, box.y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)


def main():
    parser = argparse.ArgumentParser(description="Live exercise prediction")
    parser.add_argument("--source", default="0",
                        help="Video file path or 0 for webcam")
    parser.add_argument("--model", default=None,
                        help="Path to classifier model (default: from config)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--no-yolo", action="store_true",
                        help="Skip YOLO person detection (just show pose + label)")
    parser.add_argument("--stabilizer-window", type=int, default=None,
                        help="Smoothing window size (default: from config)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load classifier
    model_path = args.model or config.exercise.classifier_model_path
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Run: python scripts/experiment.py")
        sys.exit(1)

    print(f"Loading classifier: {model_path}")
    model = joblib.load(model_path)
    print(f"  Classes: {list(model.classes_)}")

    # Init pose estimator
    print(f"Loading pose model: {config.pose.model_path}")
    pose = PoseEstimator(
        model_path=config.pose.model_path,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
    )

    # Init YOLO person detector (optional)
    detector = None
    if not args.no_yolo:
        from src.pose.person_detector import PersonDetector
        print(f"Loading YOLO: {config.person_detection.model_path}")
        detector = PersonDetector(
            model_path=config.person_detection.model_path,
            confidence_threshold=config.person_detection.confidence_threshold,
            center_weight=config.person_detection.center_weight,
            size_weight=config.person_detection.size_weight,
            iou_threshold=config.person_detection.iou_threshold,
            max_missing_frames=config.person_detection.max_missing_frames,
        )

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = isinstance(source, int)
    print(f"Source: {args.source} ({'webcam' if is_webcam else f'{total_frames} frames, {fps:.0f} fps'})")

    # Smoothing: simple majority vote over recent predictions
    window_size = args.stabilizer_window or config.exercise.stabilizer_window
    history = deque(maxlen=window_size)
    stable_label = "..."
    stable_confidence = 0.0

    frame_count = 0
    print("\nPress 'q' to quit, 'r' to reset stabilizer\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                continue
            break

        frame_count += 1

        # YOLO person detection (for box overlay only)
        person_box = None
        if detector is not None:
            person_box = detector.detect(frame)

        # Full-frame pose estimation (matches training — no crop)
        raw_landmarks, body_landmarks = pose.process_full_frame(frame)

        if raw_landmarks is not None:
            # Classify
            X = np.array(raw_landmarks).reshape(1, -1)
            pred = model.predict(X)[0]
            proba = model.predict_proba(X).max()

            history.append((pred, proba))

            # Majority vote for stable label
            votes = Counter(p for p, _ in history)
            winner, count = votes.most_common(1)[0]
            winner_conf = max(c for p, c in history if p == winner)

            if count > len(history) / 2:
                stable_label = winner
                stable_confidence = winner_conf

        # Draw
        if body_landmarks is not None:
            draw_skeleton(frame, body_landmarks)
        if person_box is not None:
            draw_bbox(frame, person_box)
        draw_label(frame, stable_label, stable_confidence)

        # FPS counter
        cv2.putText(frame, f"Frame {frame_count}", (10, frame.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Exercise Prediction", frame)

        key = cv2.waitKey(1 if is_webcam else int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            history.clear()
            stable_label = "..."
            stable_confidence = 0.0
            if detector is not None:
                detector.reset_tracking()
            print("Reset stabilizer")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
