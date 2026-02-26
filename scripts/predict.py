"""Live / video exercise prediction with rep counting.

Runs the full pipeline: MediaPipe multi-pose detection (selects best person
by body size + center proximity) → raw landmark classifier → EMA-stabilized
label → joint angle rep counter → annotated overlay.

Usage:
    python scripts/predict.py --source 0                    # webcam
    python scripts/predict.py --source data/videos/test/squat/video1.mp4
"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.pose.estimator import PoseEstimator, BODY_LANDMARK_INDICES
from src.exercise.stabilizer import ClassificationStabilizer
from src.rep_counter.counter import RepCounterManager


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


def draw_rep_count(frame, rep_count, angle, is_hold=False):
    """Draw rep counter (or hold duration) below the exercise label."""
    if is_hold:
        text = f"Hold: {rep_count}s"
    else:
        text = f"Reps: {rep_count}"
    if angle is not None:
        text += f"  [{angle:.0f}]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 65
    (tw, th), baseline = cv2.getTextSize(text, font, 0.8, 2)
    cv2.rectangle(frame, (10, y_offset), (20 + tw, y_offset + 10 + th + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, y_offset + 5 + th), font, 0.8, (0, 255, 128), 2)


def main():
    parser = argparse.ArgumentParser(description="Live exercise prediction")
    parser.add_argument("--source", default="0",
                        help="Video file path or 0 for webcam")
    parser.add_argument("--model", default=None,
                        help="Path to classifier model (default: from config)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video to outputs/")
    parser.add_argument("--config", default="config.yaml")
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

    # Init pose estimator (handles multi-person selection internally)
    print(f"Loading pose model: {config.pose.model_path}")
    pose = PoseEstimator(
        model_path=config.pose.model_path,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
        num_poses=config.pose.max_poses,
        size_weight=config.pose.person_size_weight,
        center_weight=config.pose.person_center_weight,
    )
    print(f"  Multi-pose: up to {config.pose.max_poses} people, selecting by size({config.pose.person_size_weight}) + center({config.pose.person_center_weight})")

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

    # Video writer for --save
    video_writer = None
    save_path = None
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        if isinstance(source, int):
            src_name = "webcam"
        else:
            src_name = os.path.splitext(os.path.basename(args.source))[0]
        save_path = os.path.join("outputs", f"{src_name}_annotated.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*config.output.annotated_video_codec)
        video_writer = cv2.VideoWriter(save_path, codec, fps, (w, h))
        print(f"Saving annotated video → {save_path}")

    # Stabilizer: EMA + hysteresis + minimum hold
    stabilizer = ClassificationStabilizer(
        ema_alpha=config.exercise.stabilizer_ema_alpha,
        hysteresis_margin=config.exercise.stabilizer_hysteresis_margin,
        min_hold_frames=config.exercise.stabilizer_min_hold_frames,
        min_confidence=config.exercise.min_classification_score,
    )
    stable_label = "..."
    stable_confidence = 0.0

    # Rep counter
    rep_manager = RepCounterManager(
        fps=fps,
        min_rep_duration_ms=config.rep_counter.min_rep_duration_ms,
        angle_smoothing_window=config.rep_counter.angle_smoothing_window,
    )

    frame_count = 0
    print("\nPress 'q' to quit, 'r' to reset stabilizer + rep counter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                continue
            break

        frame_count += 1

        # Full-frame pose estimation (matches training — no crop)
        # Handles multi-person selection internally (largest + most centered)
        raw_landmarks, body_landmarks = pose.process_full_frame(frame)

        if raw_landmarks is not None:
            # Classify: get full probability distribution
            X = np.array(raw_landmarks).reshape(1, -1)
            proba_array = model.predict_proba(X)[0]
            class_probs = {
                cls: float(prob)
                for cls, prob in zip(model.classes_, proba_array)
            }

            label, conf = stabilizer.update_with_probabilities(class_probs)
            if label is not None:
                stable_label = label
                stable_confidence = conf

        # Update rep counter with current exercise + body landmarks
        rep_count, tracked_angle = rep_manager.update(
            exercise=stable_label if stable_label != "..." else None,
            body_landmarks=body_landmarks if body_landmarks else {},
            frame_number=frame_count,
        )

        # Draw
        if body_landmarks is not None:
            draw_skeleton(frame, body_landmarks)
        draw_label(frame, stable_label, stable_confidence)
        is_hold = stable_label == "plank"
        draw_rep_count(frame, rep_count, tracked_angle, is_hold=is_hold)

        # FPS counter
        cv2.putText(frame, f"Frame {frame_count}", (10, frame.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Exercise Prediction", frame)

        key = cv2.waitKey(1 if is_webcam else int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            stabilizer.reset()
            rep_manager.reset()
            stable_label = "..."
            stable_confidence = 0.0
            print("Reset stabilizer + rep counter")

    if video_writer is not None:
        video_writer.release()
        print(f"Saved annotated video → {save_path}")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\nProcessed {frame_count} frames")


if __name__ == "__main__":
    main()
