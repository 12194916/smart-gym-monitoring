"""Extract raw pose landmark features from training videos.

Matches the notebook approach: MediaPipe on full frame, all 33 landmarks
(x, y, z, visibility) per frame, sampled at 15 points across each video.

Output: CSV with 132 landmark features + label per row (one row per sampled frame).

Usage:
    python scripts/extract_pose_features.py --all
    python scripts/extract_pose_features.py --all --output data/landmarks.csv
    python scripts/extract_pose_features.py --video data/videos/data_btc_10s/squat/some.mp4
"""

import argparse
import csv
import os
import sys
import traceback

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
)
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config


# Training video directories (matching notebook: verified + raw data)
DATA_DIRS = [
    "data/videos/data_btc_10s",
    "data/videos/data_crawl_10s",
    "data/videos/raw_btc",
    "data/videos/raw_crawl",
]

# 15 sample points distributed across the video (matching notebook)
SAMPLE_FRACTIONS = [
    0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
    0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95,
]

# Column names: x0, y0, z0, v0, x1, y1, z1, v1, ..., x32, y32, z32, v32
FEATURE_COLUMNS = [f'{c}{i}' for i in range(33) for c in ['x', 'y', 'z', 'v']]


def normalize_label(label: str) -> str:
    """Normalize exercise label to match notebook convention."""
    label = label.lower().replace('-', ' ').replace('_', ' ').strip()
    corrections = {
        'lateral raise': 'lateral raises',
    }
    return corrections.get(label, label)


def collect_video_paths(data_dirs: list) -> list:
    """Walk data directories and collect (video_path, label) tuples."""
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
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    videos.append((os.path.join(exercise_dir, fname), exercise_name))
    return videos


def extract_landmarks_from_video(video_path: str, detector: PoseLandmarker) -> list:
    """Extract raw landmarks from sampled frames of a video.

    Samples 15 frames distributed across the video (matching notebook).
    Returns list of 132-element feature vectors (one per sampled frame).
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    # Compute sample frame indices
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


def init_detector(config) -> PoseLandmarker:
    """Initialize MediaPipe PoseLandmarker (matching notebook defaults)."""
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=config.pose.model_path),
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)


def test_single_video(video_path: str, config):
    """Test extraction on a single video."""
    if not os.path.exists(video_path):
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    print(f"Video: {video_path}")
    print(f"  Frames: {total_frames}, FPS: {fps:.1f}")

    print("\nLoading MediaPipe PoseLandmarker...")
    detector = init_detector(config)

    print("Extracting landmarks...")
    rows = extract_landmarks_from_video(video_path, detector)
    detector.close()

    print(f"\n  Sampled frames with pose: {len(rows)}")
    print(f"  Features per frame: {len(rows[0]) if rows else 0}")

    if rows:
        print(f"\n  First frame landmarks (first 5 of 33):")
        for i in range(5):
            x, y, z, v = rows[0][i*4:(i+1)*4]
            print(f"    Landmark {i}: x={x:.3f} y={y:.3f} z={z:.3f} vis={v:.3f}")
    print("\nDone!")


def run_all_videos(config, output_path: str):
    """Process all training videos and save raw landmark features to CSV."""
    videos = collect_video_paths(DATA_DIRS)
    print(f"Found {len(videos)} videos across {len(DATA_DIRS)} directories")

    detector = init_detector(config)

    header = ['label'] + FEATURE_COLUMNS
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    total_rows = 0
    skipped = 0

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for video_path, label in tqdm(videos, desc="Extracting"):
            try:
                rows = extract_landmarks_from_video(video_path, detector)
                if not rows:
                    skipped += 1
                    continue

                norm_label = normalize_label(label)
                for row in rows:
                    writer.writerow([norm_label] + row)
                    total_rows += 1

            except Exception as e:
                print(f"\nError processing {video_path}: {e}")
                traceback.print_exc()
                skipped += 1

    detector.close()

    print(f"\nDone!")
    print(f"  Videos processed: {len(videos) - skipped}/{len(videos)}")
    print(f"  Skipped: {skipped}")
    print(f"  Total rows: {total_rows}")
    print(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract raw pose landmark features")
    parser.add_argument("--video", help="Test on a single video")
    parser.add_argument("--all", action="store_true", help="Process all training videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="data/landmarks.csv")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.error("Provide --video <path> or --all")

    config = load_config(args.config)

    if args.video:
        test_single_video(args.video, config)
    elif args.all:
        run_all_videos(config, args.output)


if __name__ == "__main__":
    main()
