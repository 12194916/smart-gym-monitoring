"""Extract and crop faces from workout videos for face recognition enrollment.

Uses InsightFace's SCRFD face detector to detect faces across sampled frames,
crops them, and organizes into train/val directories per person.

Usage:
    python scripts/extract_faces.py \
        --videos "path/to/video1.mp4::person_name1" "path/to/video2.mp4::person_name2" \
        --output data/faces \
        --n-frames 30 \
        --val-ratio 0.2

    # Quick run with defaults (labels as person_1, person_2, ...):
    python scripts/extract_faces.py \
        --videos path/to/video1.mp4 path/to/video2.mp4 \
        --output data/faces
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from tqdm import tqdm


def parse_video_arg(video_arg: str, index: int) -> tuple[str, str]:
    """Parse 'path/to/video.mp4::label' or just 'path/to/video.mp4'."""
    if "::" in video_arg:
        path, label = video_arg.rsplit("::", 1)
    else:
        path = video_arg
        label = f"person_{index + 1}"
    return path.strip(), label.strip()


def sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    """Return evenly spaced frame indices across the video."""
    if n_frames >= total_frames:
        return list(range(total_frames))
    return np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()


def extract_faces_from_video(
    video_path: str,
    face_app: FaceAnalysis,
    n_frames: int = 30,
    margin: float = 0.3,
    min_face_size: int = 40,
) -> list[np.ndarray]:
    """Detect and crop faces from sampled frames of a video.

    Args:
        video_path: Path to the video file.
        face_app: Initialized InsightFace FaceAnalysis app.
        n_frames: Number of frames to sample across the video.
        margin: Fractional padding around the detected face bbox.
        min_face_size: Minimum face width/height in pixels to keep.

    Returns:
        List of cropped face images (BGR, numpy arrays).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"  ERROR: Video has 0 frames: {video_path}")
        cap.release()
        return []

    indices = sample_frame_indices(total_frames, n_frames)
    faces = []

    for idx in tqdm(indices, desc="  Sampling frames", leave=False):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        detections = face_app.get(frame)
        if not detections:
            continue

        # Single subject per video — pick the largest face
        best = max(detections, key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]))
        x1, y1, x2, y2 = best.bbox.astype(int)
        w, h = x2 - x1, y2 - y1

        if w < min_face_size or h < min_face_size:
            continue

        # Add margin
        mx, my = int(w * margin), int(h * margin)
        x1 = max(0, x1 - mx)
        y1 = max(0, y1 - my)
        x2 = min(frame.shape[1], x2 + mx)
        y2 = min(frame.shape[0], y2 + my)

        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            faces.append(crop)

    cap.release()
    return faces


def save_faces(
    faces: list[np.ndarray],
    output_dir: Path,
    label: str,
    val_ratio: float = 0.2,
    resize: tuple[int, int] | None = (160, 160),
) -> tuple[int, int]:
    """Save cropped faces into train/val directories.

    Returns:
        (n_train, n_val) counts.
    """
    train_dir = output_dir / label / "train"
    val_dir = output_dir / label / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(faces)
    n_val = max(1, int(n_total * val_ratio)) if n_total > 1 else 0
    n_train = n_total - n_val

    # Shuffle for random split
    indices = np.random.permutation(n_total)

    for i, idx in enumerate(indices):
        face = faces[idx]
        if resize:
            face = cv2.resize(face, resize, interpolation=cv2.INTER_AREA)

        if i < n_train:
            path = train_dir / f"face_{i:04d}.jpg"
        else:
            path = val_dir / f"face_{i - n_train:04d}.jpg"

        cv2.imwrite(str(path), face, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return n_train, n_val


def main():
    parser = argparse.ArgumentParser(
        description="Extract faces from workout videos for enrollment."
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help='Video paths, optionally with label: "path.mp4::person_name"',
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/faces",
        help="Output root directory (default: data/faces)",
    )
    parser.add_argument(
        "--n-frames",
        type=int,
        default=30,
        help="Number of frames to sample per video (default: 30)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of faces for validation (default: 0.2)",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
        help="Fractional padding around face bbox (default: 0.3)",
    )
    parser.add_argument(
        "--det-size",
        type=int,
        default=640,
        help="Face detection input size (default: 640)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)

    # Initialize InsightFace detector
    print("Loading InsightFace detector...")
    face_app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    face_app.prepare(ctx_id=-1, det_size=(args.det_size, args.det_size))
    print("Detector ready.\n")

    for i, video_arg in enumerate(args.videos):
        video_path, label = parse_video_arg(video_arg, i)

        if not Path(video_path).exists():
            print(f"SKIP: File not found: {video_path}")
            continue

        print(f"[{i + 1}/{len(args.videos)}] Processing: {Path(video_path).name} → label: {label}")

        faces = extract_faces_from_video(
            video_path=video_path,
            face_app=face_app,
            n_frames=args.n_frames,
            margin=args.margin,
        )

        if not faces:
            print(f"  WARNING: No faces detected in {video_path}\n")
            continue

        n_train, n_val = save_faces(
            faces=faces,
            output_dir=output_dir,
            label=label,
            val_ratio=args.val_ratio,
        )
        print(f"  Saved {n_train} train + {n_val} val faces → {output_dir / label}/\n")

    print("Done!")


if __name__ == "__main__":
    main()
