"""Evaluate trained landmark classifier on test videos.

Processes every frame of the first video per class through the same pipeline
as predict.py: per-frame predict_proba → EMA stabilizer → final label.
Reports both strict accuracy and family accuracy.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --model data/models/exercise_classifier.pkl
    python scripts/evaluate.py --test-dir data/videos/test
"""

import argparse
import os
import sys

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
from src.exercise.stabilizer import ClassificationStabilizer


# Exercise families: predictions within the same family count as correct
EXERCISE_FAMILIES = {
    "bench press":         "bench press",
    "decline bench press": "bench press",
    "incline bench press": "bench press",
    "deadlift":            "deadlift",
    "romanian deadlift":   "deadlift",
    "barbell biceps curl": "curl",
    "hammer curl":         "curl",
}


def get_family(label: str) -> str:
    """Return exercise family, or the label itself if it has no family."""
    return EXERCISE_FAMILIES.get(label, label)


def normalize_label(label: str) -> str:
    """Normalize folder name to match model class names."""
    label = label.lower().replace('-', ' ').replace('_', ' ').strip()
    corrections = {
        'lateral raise': 'lateral raises',
    }
    return corrections.get(label, label)


def collect_test_videos(test_dir: str) -> list:
    """Collect first video per class from test directory.

    Returns list of (video_path, raw_folder_name) tuples.
    """
    videos = []
    for exercise_name in sorted(os.listdir(test_dir)):
        exercise_dir = os.path.join(test_dir, exercise_name)
        if not os.path.isdir(exercise_dir):
            continue
        # Take only the first video (sorted) from each class folder
        for fname in sorted(os.listdir(exercise_dir)):
            if fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                videos.append((os.path.join(exercise_dir, fname), exercise_name))
                break
    return videos


def predict_video_with_stabilizer(
    video_path: str,
    model,
    detector: PoseLandmarker,
    stabilizer_config,
) -> tuple:
    """Predict exercise by running every frame through the EMA stabilizer.

    Same pipeline as predict.py: extract pose → predict_proba → stabilizer.
    Returns (predicted_label, confidence, n_frames_processed).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return "unknown", 0.0, 0

    stabilizer = ClassificationStabilizer(
        ema_alpha=stabilizer_config.stabilizer_ema_alpha,
        hysteresis_margin=stabilizer_config.stabilizer_hysteresis_margin,
        min_hold_frames=stabilizer_config.stabilizer_min_hold_frames,
        min_confidence=stabilizer_config.min_classification_score,
    )

    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = detector.detect(mp_image)

        if not result.pose_landmarks:
            continue

        # Extract 132-dim raw landmarks (same as predict.py)
        raw_landmarks = []
        for lm in result.pose_landmarks[0]:
            raw_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility if lm.visibility else 0.0])

        X = np.array(raw_landmarks).reshape(1, -1)
        proba_array = model.predict_proba(X)[0]
        class_probs = {
            cls: float(prob)
            for cls, prob in zip(model.classes_, proba_array)
        }

        label, conf = stabilizer.update_with_probabilities(class_probs)
        frames_processed += 1

    cap.release()

    if label is None:
        return "unknown", 0.0, frames_processed

    return label, conf, frames_processed


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

    # Load config
    config = load_config(args.config)

    # Init MediaPipe
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.pose.model_path),
        output_segmentation_masks=False,
    )
    detector = PoseLandmarker.create_from_options(options)

    # Collect first video per class
    videos = collect_test_videos(args.test_dir)
    print(f"Found {len(videos)} classes (1 video each)\n")

    # Get model's known classes
    known_classes = set(model.classes_)

    # Evaluate
    strict_correct = 0
    family_correct = 0
    total = 0
    skipped = 0
    results = []

    print(f"{'Video':<50} {'True':<22} {'Predicted':<22} {'Conf':>5} {'Strict':>7} {'Family':>7}")
    print("=" * 120)

    for video_path, raw_label in videos:
        true_label = normalize_label(raw_label)
        fname = os.path.basename(video_path)

        # Skip classes not in training set
        if true_label not in known_classes:
            print(f"{fname:<50} {true_label:<22} {'SKIP (not in model)':<22}")
            skipped += 1
            continue

        pred_label, confidence, n_frames = predict_video_with_stabilizer(
            video_path, model, detector, config.exercise,
        )

        is_strict = pred_label == true_label
        is_family = is_strict or get_family(pred_label) == get_family(true_label)

        if is_strict:
            strict_correct += 1
        if is_family:
            family_correct += 1
        total += 1

        strict_str = "OK" if is_strict else "WRONG"
        family_str = "OK" if is_family else "WRONG"
        print(f"{fname:<50} {true_label:<22} {pred_label:<22} {confidence:>4.0%} {strict_str:>7} {family_str:>7}")

        results.append({
            "video": video_path,
            "true": true_label,
            "predicted": pred_label,
            "confidence": confidence,
            "strict_correct": is_strict,
            "family_correct": is_family,
            "frames_processed": n_frames,
        })

    detector.close()

    # Summary
    strict_acc = strict_correct / total if total > 0 else 0
    family_acc = family_correct / total if total > 0 else 0

    print(f"\n{'=' * 120}")
    print(f"RESULTS")
    print(f"{'=' * 120}")
    print(f"Classes evaluated: {total}")
    print(f"Skipped: {skipped}")
    print(f"Strict accuracy:  {strict_correct}/{total} ({strict_acc:.1%})")
    print(f"Family accuracy:  {family_correct}/{total} ({family_acc:.1%})")

    # Show families used
    families_used = set()
    for r in results:
        t_fam = get_family(r["true"])
        p_fam = get_family(r["predicted"])
        if t_fam != r["true"] or p_fam != r["predicted"]:
            families_used.add(t_fam)

    if families_used:
        print(f"\nExercise families applied:")
        for fam in sorted(families_used):
            members = [k for k, v in EXERCISE_FAMILIES.items() if v == fam]
            print(f"  {fam}: {', '.join(members)}")

    # Show wrong predictions
    wrong = [r for r in results if not r["family_correct"]]
    if wrong:
        print(f"\nWrong predictions (not even family match): {len(wrong)}")
        for r in wrong:
            fname = os.path.basename(r["video"])
            print(f"  {fname}")
            print(f"    True: {r['true']} ({get_family(r['true'])})")
            print(f"    Pred: {r['predicted']} ({get_family(r['predicted'])})")

    # Show family-saved predictions (wrong strict, correct family)
    saved = [r for r in results if not r["strict_correct"] and r["family_correct"]]
    if saved:
        print(f"\nFamily-saved predictions (wrong strict, OK family): {len(saved)}")
        for r in saved:
            fname = os.path.basename(r["video"])
            print(f"  {fname}: {r['true']} -> {r['predicted']} (family: {get_family(r['true'])})")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
