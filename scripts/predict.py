"""Predict exercise class from a video file or live webcam.

Usage:
    # From a video file:
    python scripts/predict.py data/videos/test/squat/some_video.mp4

    # Live webcam mode:
    python scripts/predict.py --live
    python scripts/predict.py --live --model data/models/exercise_classifier.pkl
"""

import argparse
import os
import sys
import time
import pickle
from collections import deque

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.video_io import read_video_frames, get_video_metadata
from src.pose.person_detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.pose.angles import compute_joint_angles
from src.exercise.classifier import ExerciseClassifier
from src.exercise.definitions import Exercise
from src.rep_counter.counter import RepCounter
from src.rep_counter.rep_rules import REP_RULES, get_tracked_angle
from src.output.video_annotator import VideoAnnotator

ANGLE_NAMES = [
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "torso_inclination",
]


def run_video(args, config):
    """Predict exercise from a pre-recorded video file (original behavior)."""
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Run scripts/experiment.py first")
        sys.exit(1)

    # Load model
    print("Loading model...")
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    pipe = saved["pipeline"]
    le = saved["label_encoder"]
    feature_columns = saved["feature_columns"]

    # Load config and init pose pipeline
    meta = get_video_metadata(args.video)
    print(f"Video: {meta['width']}x{meta['height']}, {meta['fps']:.1f} fps, "
          f"{meta['total_frames']} frames, {meta['duration_seconds']:.1f}s")

    print("Loading pose pipeline...")
    detector = PersonDetector(
        model_path=config.person_detection.model_path,
        confidence_threshold=config.person_detection.confidence_threshold,
        center_weight=config.person_detection.center_weight,
        size_weight=config.person_detection.size_weight,
    )
    estimator = PoseEstimator(
        model_path=config.pose.model_path,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
    )

    # Extract angles
    print("Processing frames...")
    angles_list = []
    for frame_idx, frame, timestamp_ms in read_video_frames(args.video):
        person = detector.detect(frame)
        if person is None:
            continue
        crop, cx, cy = detector.get_padded_crop(frame, person, padding=config.person_detection.bbox_padding)
        landmarks = estimator.process(crop, offset_x=cx, offset_y=cy)
        if landmarks is None:
            continue
        angles = compute_joint_angles(landmarks, timestamp_ms)
        angles_list.append(angles)

    estimator.close()

    if not angles_list:
        print("No poses detected in this video.")
        sys.exit(1)

    print(f"Poses extracted: {len(angles_list)}/{meta['total_frames']} frames")

    # Aggregate features
    features = {}
    for name in ANGLE_NAMES:
        values = [getattr(a, name) for a in angles_list]
        arr = np.array(values)
        features[f"{name}_mean"] = float(np.mean(arr))
        features[f"{name}_std"] = float(np.std(arr))
        features[f"{name}_min"] = float(np.min(arr))
        features[f"{name}_max"] = float(np.max(arr))
    features["num_frames"] = len(angles_list)

    # Predict
    feature_values = [features[col] for col in feature_columns]
    X = np.array([feature_values])

    pred_idx = pipe.predict(X)[0]
    pred_label = le.inverse_transform([pred_idx])[0]

    # Confidence (probability per class)
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X)[0]
        top_3 = sorted(zip(le.classes_, proba), key=lambda x: x[1], reverse=True)[:3]

        print(f"\n{'=' * 40}")
        print(f"  PREDICTION: {pred_label}")
        print(f"{'=' * 40}")
        print(f"\n  Top 3:")
        for label, prob in top_3:
            bar = "#" * int(prob * 30)
            print(f"    {label:<25} {prob:>5.1%} {bar}")
    else:
        print(f"\n{'=' * 40}")
        print(f"  PREDICTION: {pred_label}")
        print(f"{'=' * 40}")


def run_live(config, model_path: str):
    """Run real-time exercise monitoring from webcam."""
    print("Initializing live mode...")

    # Initialize pose pipeline
    print("Loading pose pipeline...")
    detector = PersonDetector(
        model_path=config.person_detection.model_path,
        confidence_threshold=config.person_detection.confidence_threshold,
        center_weight=config.person_detection.center_weight,
        size_weight=config.person_detection.size_weight,
    )
    estimator = PoseEstimator(
        model_path=config.pose.model_path,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
    )

    # Initialize classifier, rep counter, annotator
    print("Loading exercise classifier...")
    classifier = ExerciseClassifier(
        model_path=model_path,
        window_size=config.exercise.window_size,
    )
    rep_counter = RepCounter()
    annotator = VideoAnnotator(
        draw_skeleton=config.output.draw_skeleton,
        draw_angles=config.output.draw_angles,
        draw_labels=config.output.draw_labels,
    )

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_w}x{actual_h}")
    print()
    print("Controls:")
    print("  q - Quit")
    print("  r - Reset rep count and classifier buffer")
    print()

    frame_idx = 0
    interval = config.exercise.classification_interval_frames
    min_score = config.exercise.min_classification_score

    # Cached results for display
    last_landmarks = None
    last_exercise = Exercise.UNKNOWN
    last_confidence = 0.0
    last_tracked_value = None
    last_tracked_name = None

    # FPS tracking
    fps = 0.0
    frame_times = deque(maxlen=30)
    start_ms = int(time.time() * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)
        timestamp_ms = int(time.time() * 1000) - start_ms

        # FPS calculation
        now = time.time()
        frame_times.append(now)
        if len(frame_times) > 1:
            elapsed = frame_times[-1] - frame_times[0]
            if elapsed > 0:
                fps = (len(frame_times) - 1) / elapsed

        # --- POSE DETECTION (every frame) ---
        person = detector.detect(frame)
        if person is not None:
            crop, cx, cy = detector.get_padded_crop(
                frame, person, padding=config.person_detection.bbox_padding
            )
            landmarks = estimator.process(crop, offset_x=cx, offset_y=cy)
            if landmarks is not None:
                last_landmarks = landmarks
                angles = compute_joint_angles(landmarks, timestamp_ms)

                # Add angles to classifier buffer EVERY frame
                classifier.update(angles)

                # --- CLASSIFICATION (every Nth frame) ---
                if frame_idx % interval == 0:
                    exercise, confidence = classifier.classify()
                    if confidence >= min_score:
                        last_exercise = exercise
                        last_confidence = confidence

                # --- REP COUNTING (every frame) ---
                if last_exercise != Exercise.UNKNOWN:
                    rep_counter.update(last_exercise, angles, timestamp_ms)

                    # Get tracked angle for display
                    if last_exercise in REP_RULES:
                        rule = REP_RULES[last_exercise]
                        last_tracked_value = get_tracked_angle(angles, rule["track_angle"])
                        last_tracked_name = rule["track_angle"]
            else:
                last_landmarks = None
        else:
            last_landmarks = None

        # --- ANNOTATE & DISPLAY (every frame) ---
        annotated = annotator.draw(
            frame,
            landmarks=last_landmarks,
            exercise=last_exercise.value if last_exercise else "unknown",
            rep_count=rep_counter.count,
            confidence=last_confidence,
            tracked_angle_value=last_tracked_value,
            tracked_angle_name=last_tracked_name,
            fps=fps,
        )

        cv2.imshow("Gym Monitor - Live", annotated)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_counter.reset(last_exercise)
            classifier.reset()
            last_exercise = Exercise.UNKNOWN
            last_confidence = 0.0
            last_tracked_value = None
            last_tracked_name = None
            print("Reset: rep count and classifier buffer cleared")

        frame_idx += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    estimator.close()

    # Session summary
    print(f"\nSession summary:")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Last exercise: {last_exercise.value}")
    print(f"  Total reps: {rep_counter.count}")
    if rep_counter.reps:
        print(f"  Rep details:")
        for rep in rep_counter.reps:
            print(f"    Rep {rep.rep_number}: {rep.duration_ms}ms "
                  f"({rep.start_ms/1000:.1f}s - {rep.end_ms/1000:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Predict exercise from video or live webcam")
    parser.add_argument("video", nargs="?", default=None, help="Path to video file")
    parser.add_argument("--live", action="store_true", help="Use webcam for live monitoring")
    parser.add_argument("--model", default="data/models/exercise_classifier.pkl")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not args.live and args.video is None:
        parser.error("Provide a video path or use --live for webcam mode")

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Run scripts/experiment.py first to train a model")
        sys.exit(1)

    config = load_config(args.config)

    if args.live:
        run_live(config, args.model)
    else:
        run_video(args, config)


if __name__ == "__main__":
    main()
