"""Extract pose angle features from training videos.

Processes videos through YOLOv8 person detection + MediaPipe pose estimation
+ angle computation on each frame, and outputs aggregated features.

Usage:
    # Test on a single video first:
    python scripts/extract_features.py --video data/videos/verified_data/verified_data/data_btc_10s/squat/some_video.mp4

    # Run on all training videos:
    python scripts/extract_features.py --all --output data/features.csv
"""

import argparse
import csv
import os
import sys
import traceback

# Add project root to Python path so 'src' imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tqdm import tqdm

from src.utils.config import load_config
from src.utils.video_io import read_video_frames, get_video_metadata
from src.pose.person_detector import PersonDetector
from src.pose.estimator import PoseEstimator
from src.pose.angles import compute_joint_angles, JointAngles

import numpy as np


# Directories containing training videos (both btc and crawl)
DATA_DIRS = [
    "data/videos/data_btc_10s",
    "data/videos/data_crawl_10s",
]

# The 9 angle names in the order they appear in JointAngles
ANGLE_NAMES = [
    "left_knee", "right_knee",
    "left_hip", "right_hip",
    "left_elbow", "right_elbow",
    "left_shoulder", "right_shoulder",
    "torso_inclination",
]


def collect_video_paths(data_dirs: list) -> list:
    """Walk data directories and collect (video_path, exercise_label) tuples."""
    videos = []
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            print(f"Warning: directory not found: {data_dir}")
            continue
        for exercise_name in sorted(os.listdir(data_dir)):
            exercise_dir = os.path.join(data_dir, exercise_name)
            if not os.path.isdir(exercise_dir):
                continue
            for fname in os.listdir(exercise_dir):
                if fname.endswith(".mp4"):
                    videos.append((os.path.join(exercise_dir, fname), exercise_name))
    return videos


def extract_angles_from_video(
    video_path: str,
    detector: PersonDetector,
    estimator: PoseEstimator,
    bbox_padding: float,
    verbose: bool = False,
) -> tuple:
    """Extract JointAngles for each frame of a video.

    Returns list of JointAngles, may be empty if no poses detected.
    """
    angles_list = []
    no_person = 0
    no_pose = 0

    for frame_idx, frame, timestamp_ms in read_video_frames(video_path):
        # Step 1: Detect person with YOLOv8
        person = detector.detect(frame)
        if person is None:
            no_person += 1
            if verbose:
                print(f"  Frame {frame_idx}: No person detected")
            continue

        # Step 2: Crop to detected person
        crop, cx, cy = detector.get_padded_crop(frame, person, padding=bbox_padding)

        # Step 3: Extract pose landmarks with MediaPipe
        landmarks = estimator.process(crop, offset_x=cx, offset_y=cy)
        if landmarks is None:
            no_pose += 1
            if verbose:
                print(f"  Frame {frame_idx}: Person found but no pose detected")
            continue

        # Step 4: Compute joint angles
        angles = compute_joint_angles(landmarks, timestamp_ms)
        angles_list.append(angles)

        if verbose:
            print(f"  Frame {frame_idx}: OK - L.knee={angles.left_knee:.0f}° "
                  f"R.knee={angles.right_knee:.0f}° torso={angles.torso_inclination:.0f}°")

    return angles_list, no_person, no_pose


def aggregate_features(angles_list: list) -> dict:
    """Compute mean and std of each angle across all frames of a video.

    Returns dict with keys like 'left_knee_mean', 'left_knee_std', etc.
    Also includes min and max for each angle (useful for classifying exercises
    by range of motion).
    """
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


def init_models(config):
    """Initialize YOLOv8 and MediaPipe models."""
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
    return detector, estimator


def test_single_video(video_path: str, config):
    """Test the full pipeline on a single video and print results."""
    from src.utils.video_io import get_video_metadata

    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    print("=" * 60)
    print("STEP 0: Reading video metadata")
    print("=" * 60)
    meta = get_video_metadata(video_path)
    print(f"  File:       {video_path}")
    print(f"  Resolution: {meta['width']}x{meta['height']}")
    print(f"  FPS:        {meta['fps']:.1f}")
    print(f"  Frames:     {meta['total_frames']}")
    print(f"  Duration:   {meta['duration_seconds']:.1f}s")

    print()
    print("=" * 60)
    print("STEP 1: Loading YOLOv8 person detector")
    print("=" * 60)
    print(f"  Model: {config.person_detection.model_path}")
    detector, estimator = init_models(config)
    print("  YOLOv8 loaded OK")
    print(f"  MediaPipe loaded OK ({config.pose.model_path})")

    print()
    print("=" * 60)
    print("STEP 2: Processing frames (YOLO -> MediaPipe -> Angles)")
    print("=" * 60)
    angles_list, no_person, no_pose = extract_angles_from_video(
        video_path, detector, estimator,
        bbox_padding=config.person_detection.bbox_padding,
        verbose=True,
    )
    estimator.close()

    print()
    print("=" * 60)
    print("STEP 3: Results summary")
    print("=" * 60)
    total = meta["total_frames"]
    detected = len(angles_list)
    print(f"  Total frames:     {total}")
    print(f"  Poses extracted:  {detected} ({100*detected/total:.1f}%)")
    print(f"  No person found:  {no_person}")
    print(f"  No pose detected: {no_pose}")

    if not angles_list:
        print("\n  No poses detected in this video.")
        return

    print()
    print("=" * 60)
    print("STEP 4: Aggregated features")
    print("=" * 60)
    features = aggregate_features(angles_list)

    print(f"  {'Angle':<22} {'Mean':>7} {'Std':>7} {'Min':>7} {'Max':>7}")
    print(f"  {'-'*52}")
    for name in ANGLE_NAMES:
        print(f"  {name:<22} {features[f'{name}_mean']:>6.1f}° {features[f'{name}_std']:>6.1f}° "
              f"{features[f'{name}_min']:>6.1f}° {features[f'{name}_max']:>6.1f}°")

    print(f"\n  Total feature columns: {len(features)}")
    print("  DONE - pipeline works!")


def run_all_videos(config, output_path: str):
    """Process all training videos and save features to CSV."""
    videos = collect_video_paths(DATA_DIRS)
    print(f"Found {len(videos)} videos across {len(DATA_DIRS)} directories")

    detector, estimator = init_models(config)

    # Build CSV header
    feature_columns = []
    for name in ANGLE_NAMES:
        feature_columns.extend([
            f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max",
        ])
    feature_columns.append("num_frames")
    header = ["video_path", "label", "source"] + feature_columns

    # Process all videos
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    skipped = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()

        for video_path, label in tqdm(videos, desc="Extracting features"):
            try:
                angles_list, _, _ = extract_angles_from_video(
                    video_path, detector, estimator,
                    bbox_padding=config.person_detection.bbox_padding,
                    verbose=False,
                )

                features = aggregate_features(angles_list)
                if features is None:
                    skipped += 1
                    continue

                source = "btc" if "data_btc_10s" in video_path else "crawl"
                row = {"video_path": video_path, "label": label, "source": source}
                row.update(features)
                writer.writerow(row)

            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                traceback.print_exc()
                skipped += 1

    estimator.close()

    processed = len(videos) - skipped
    print(f"\nDone!")
    print(f"  Processed: {processed}/{len(videos)} videos")
    print(f"  Skipped:   {skipped}")
    print(f"  Output:    {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract pose features from training videos")
    parser.add_argument("--video", help="Test on a single video file")
    parser.add_argument("--all", action="store_true", help="Process all training videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/features.csv")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Provide --video <path> to test one video, or --all to process everything")

    config = load_config(args.config)

    if args.video:
        test_single_video(args.video, config)
    elif args.all:
        run_all_videos(config, args.output)


if __name__ == "__main__":
    main()
