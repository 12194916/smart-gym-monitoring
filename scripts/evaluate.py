"""Evaluate the trained classifier on unseen test videos.

Processes test videos through the full pipeline (YOLO -> MediaPipe -> angles -> classifier)
and reports accuracy against ground truth labels (folder names).

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model data/models/exercise_classifier.pkl --test-dir data/videos/test
"""

import argparse
import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.utils.video_io import read_video_frames, get_video_metadata
from src.pose.person_detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.pose.angles import compute_joint_angles

# Same as in extract_pose_features.py
ANGLE_NAMES = [
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "torso_inclination",
]


def collect_test_videos(test_dir: str) -> list:
    """Collect (video_path, true_label) from test directory."""
    videos = []
    for label in sorted(os.listdir(test_dir)):
        label_dir = os.path.join(test_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".mp4"):
                videos.append((os.path.join(label_dir, fname), label))
    return videos


def extract_features_from_video(video_path, detector, estimator, bbox_padding):
    """Extract aggregated angle features from a single video."""
    angles_list = []

    for frame_idx, frame, timestamp_ms in read_video_frames(video_path):
        person = detector.detect(frame)
        if person is None:
            continue
        crop, cx, cy = detector.get_padded_crop(frame, person, padding=bbox_padding)
        landmarks = estimator.process(crop, offset_x=cx, offset_y=cy)
        if landmarks is None:
            continue
        angles = compute_joint_angles(landmarks, timestamp_ms)
        angles_list.append(angles)

    if not angles_list:
        return None

    features = {}
    for name in ANGLE_NAMES:
        values = [getattr(a, name) for a in angles_list]
        arr = np.array(values)
        features[f"{name}_mean"] = float(np.mean(arr))
        features[f"{name}_std"] = float(np.std(arr))
        features[f"{name}_min"] = float(np.min(arr))
        features[f"{name}_max"] = float(np.max(arr))
    features["num_frames"] = len(angles_list)
    return features


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on test videos")
    parser.add_argument("--model", default="data/models/exercise_classifier.pkl",
                        help="Path to trained model pickle")
    parser.add_argument("--test-dir", default="data/videos/test",
                        help="Directory with test videos (folder name = label)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # Load trained model
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Run scripts/experiment.py first to train a model")
        sys.exit(1)

    print("Loading trained model...")
    with open(args.model, "rb") as f:
        saved = pickle.load(f)
    pipe = saved["pipeline"]
    le = saved["label_encoder"]
    feature_columns = saved["feature_columns"]
    print(f"  Classes: {le.classes_.tolist()}")
    print(f"  Features: {len(feature_columns)} columns")

    # Collect test videos — only classes the model was trained on
    all_test_videos = collect_test_videos(args.test_dir)
    train_labels = set(le.classes_)
    test_videos = [(p, l) for p, l in all_test_videos if l in train_labels]
    skipped_classes = set(l for _, l in all_test_videos) - train_labels

    print(f"\nFound {len(all_test_videos)} test videos total")
    print(f"  Using {len(test_videos)} videos (classes the model learned)")
    if skipped_classes:
        print(f"  Skipping {len(all_test_videos) - len(test_videos)} videos from unknown classes:")
        for cls in sorted(skipped_classes):
            count = sum(1 for _, l in all_test_videos if l == cls)
            print(f"    {cls}: {count} videos (not in training data)")

    # Initialize pose pipeline
    print("\nLoading pose pipeline...")
    config = load_config(args.config)
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

    # Process each test video
    print("\nProcessing test videos...")
    true_labels = []
    pred_labels = []
    skipped = 0
    results_per_video = []

    for video_path, true_label in tqdm(test_videos, desc="Evaluating"):
        features = extract_features_from_video(
            video_path, detector, estimator,
            bbox_padding=config.person_detection.bbox_padding,
        )

        if features is None:
            skipped += 1
            results_per_video.append({
                "video": video_path, "true": true_label,
                "pred": "SKIPPED", "correct": False,
            })
            continue

        # Build feature vector in the exact order the model was trained on
        feature_values = [features[col] for col in feature_columns]
        X_test = np.array([feature_values])

        # Predict
        y_pred = pipe.predict(X_test)[0]
        pred_label = le.inverse_transform([y_pred])[0]

        true_labels.append(true_label)
        pred_labels.append(pred_label)
        correct = true_label == pred_label

        results_per_video.append({
            "video": os.path.basename(video_path),
            "true": true_label, "pred": pred_label,
            "correct": correct,
        })

    estimator.close()

    # Results
    print(f"\n{'=' * 70}")
    print("EVALUATION RESULTS ON UNSEEN TEST VIDEOS")
    print(f"{'=' * 70}")
    print(f"  Total test videos:  {len(test_videos)}")
    print(f"  Evaluated:          {len(true_labels)}")
    print(f"  Skipped (no pose):  {skipped}")

    if true_labels:
        acc = accuracy_score(true_labels, pred_labels)
        print(f"\n  ACCURACY: {acc:.1%}")

        print(f"\n{classification_report(true_labels, pred_labels, digits=2)}")

        # Per-video results
        print("\nPer-video results:")
        for r in results_per_video:
            status = "OK" if r["correct"] else "WRONG" if r["pred"] != "SKIPPED" else "SKIP"
            print(f"  [{status:>5}] {r['true']:<25} -> {r['pred']:<25} ({r['video']})")

        # Save results
        results_path = "data/models/test_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "accuracy": acc,
                "total": len(test_videos),
                "evaluated": len(true_labels),
                "skipped": skipped,
                "per_video": results_per_video,
            }, f, indent=2)
        print(f"\nSaved results to: {results_path}")


if __name__ == "__main__":
    main()
