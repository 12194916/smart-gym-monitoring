"""Evaluate trained landmark classifier on test videos.

Extracts landmarks from test videos (same 15-frame sampling as training),
predicts per-frame, then majority-votes for a per-video label.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model data/models/exercise_classifier.pkl
    python scripts/evaluate.py --test-dir data/videos/test
"""

import argparse
import os
import sys
from collections import Counter

import cv2
import joblib
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config

# Same sampling fractions as training extraction and notebook
SAMPLE_FRACTIONS = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95,
]


def normalize_label(label: str) -> str:
    """Normalize label to match training convention."""
    label = label.lower().replace('-', ' ').replace('_', ' ').strip()
    corrections = {
        'lateral raise': 'lateral raises',
    }
    return corrections.get(label, label)


def extract_landmarks_from_video(video_path: str, detector: PoseLandmarker) -> list:
    """Extract raw landmarks from sampled frames (matching training extraction)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    if total_frames > 15:
        sample_indices = [int(total_frames * p) for p in SAMPLE_FRACTIONS]
    else:
        sample_indices = list(range(0, total_frames, 2))

    rows = []
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)

        if result.pose_landmarks:
            row = []
            for lm in result.pose_landmarks[0]:
                row.extend([lm.x, lm.y, lm.z, lm.visibility if lm.visibility else 0.0])
            rows.append(row)

    cap.release()
    return rows


def collect_test_videos(test_dir: str) -> list:
    """Collect (video_path, label) tuples from test directory."""
    videos = []
    for exercise_name in sorted(os.listdir(test_dir)):
        exercise_dir = os.path.join(test_dir, exercise_name)
        if not os.path.isdir(exercise_dir):
            continue
        for fname in sorted(os.listdir(exercise_dir)):
            if fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append((os.path.join(exercise_dir, fname), exercise_name))
    return videos


def predict_video(model, landmarks: list) -> tuple:
    """Predict exercise for a video using per-frame prediction + majority vote.

    Returns (predicted_label, confidence, per_frame_predictions).
    """
    if not landmarks:
        return "unknown", 0.0, []

    X = np.array(landmarks)
    frame_predictions = model.predict(X)

    # Majority vote
    vote_counts = Counter(frame_predictions)
    predicted_label = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[predicted_label] / len(frame_predictions)

    return predicted_label, confidence, list(frame_predictions)


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier on test videos")
    parser.add_argument("--model", default="data/models/exercise_classifier.pkl",
                        help="Path to trained model")
    parser.add_argument("--test-dir", default="data/videos/test",
                        help="Test video directory")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        print("Run: python scripts/experiment.py")
        sys.exit(1)

    if not os.path.isdir(args.test_dir):
        print(f"Error: Test directory not found: {args.test_dir}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {args.model}")
    model = joblib.load(args.model)

    # Init MediaPipe (matching training extraction — default options)
    config = load_config(args.config)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.pose.model_path),
        output_segmentation_masks=False,
    )
    detector = PoseLandmarker.create_from_options(options)

    # Collect test videos
    videos = collect_test_videos(args.test_dir)
    print(f"Found {len(videos)} test videos\n")

    # Get model's known classes
    known_classes = set(model.classes_)

    # Evaluate
    correct = 0
    total = 0
    skipped = 0
    per_class_correct = Counter()
    per_class_total = Counter()
    results = []

    print(f"{'Video':<60} {'True':<22} {'Predicted':<22} {'Conf':>5} {'':>4}")
    print("=" * 115)

    for video_path, raw_label in videos:
        true_label = normalize_label(raw_label)

        # Skip classes not in training set
        if true_label not in known_classes:
            fname = os.path.basename(video_path)
            print(f"{fname:<60} {true_label:<22} {'SKIP (not in model)':<22}")
            skipped += 1
            continue

        landmarks = extract_landmarks_from_video(video_path, detector)
        if not landmarks:
            fname = os.path.basename(video_path)
            print(f"{fname:<60} {true_label:<22} {'SKIP (no pose)':<22}")
            skipped += 1
            continue

        pred_label, confidence, frame_preds = predict_video(model, landmarks)

        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        total += 1

        per_class_total[true_label] += 1
        if is_correct:
            per_class_correct[true_label] += 1

        status = "OK" if is_correct else "WRONG"
        fname = os.path.basename(video_path)
        print(f"{fname:<60} {true_label:<22} {pred_label:<22} {confidence:>4.0%} {status:>5}")

        results.append({
            "video": video_path,
            "true": true_label,
            "predicted": pred_label,
            "confidence": confidence,
            "correct": is_correct,
            "frames_with_pose": len(landmarks),
            "frame_predictions": frame_preds,
        })

    detector.close()

    # Summary
    accuracy = correct / total if total > 0 else 0
    print(f"\n{'=' * 115}")
    print(f"RESULTS")
    print(f"{'=' * 115}")
    print(f"Total videos: {len(videos)}")
    print(f"Evaluated: {total}")
    print(f"Skipped: {skipped}")
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1%}")

    # Per-class breakdown
    print(f"\nPer-class accuracy:")
    for label in sorted(per_class_total.keys()):
        c = per_class_correct[label]
        t = per_class_total[label]
        acc = c / t if t > 0 else 0
        print(f"  {label:<25} {c}/{t} ({acc:.0%})")

    # Show wrong predictions grouped
    wrong = [r for r in results if not r["correct"]]
    if wrong:
        print(f"\nWrong predictions ({len(wrong)}):")
        for r in wrong:
            fname = os.path.basename(r["video"])
            # Show frame-level vote breakdown
            votes = Counter(r["frame_predictions"])
            vote_str = ", ".join(f"{k}: {v}" for k, v in votes.most_common())
            print(f"  {fname}")
            print(f"    True: {r['true']}, Predicted: {r['predicted']}")
            print(f"    Frame votes: {vote_str}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
