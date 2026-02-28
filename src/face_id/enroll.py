"""Build face embedding gallery from enrollment images.

Processes face images organized as:
    data/faces/<person_name>/train/*.jpg

For each person, computes face embeddings using InsightFace's recognition
model and averages them into a single 512-dim vector. The gallery is saved
as a pickle file for use by FaceRecognizer.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def build_gallery(config) -> dict:
    """Process enrollment images and save face embedding gallery to disk.

    Args:
        config: PipelineConfig (uses config.face for model settings).

    Returns:
        Gallery dict: {person_name: np.ndarray (512-dim embedding)}.
    """
    face_cfg = config.face

    # Initialize InsightFace
    app = FaceAnalysis(
        name=face_cfg.model_name,
        providers=face_cfg.providers,
    )
    app.prepare(ctx_id=-1, det_size=tuple(face_cfg.det_size))

    enrollment_dir = Path(face_cfg.enrollment_dir)
    if not enrollment_dir.exists():
        raise FileNotFoundError(f"Enrollment directory not found: {enrollment_dir}")

    gallery = {}

    for person_dir in sorted(enrollment_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        train_dir = person_dir / "train"

        # Support both data/faces/<name>/train/ and data/faces/<name>/
        image_dir = train_dir if train_dir.exists() else person_dir
        image_paths = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
        )

        if not image_paths:
            print(f"  SKIP {person_name}: no images found")
            continue

        embeddings = []
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            faces = app.get(img)
            if not faces:
                continue

            # Pick the largest face (should be only one in cropped images)
            best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embeddings.append(best.embedding)

        if not embeddings:
            print(f"  SKIP {person_name}: no faces detected in any image")
            continue

        # Average embeddings and L2-normalize
        avg_embedding = np.mean(embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        gallery[person_name] = avg_embedding
        print(f"  Enrolled {person_name}: {len(embeddings)} faces → 512-dim embedding")

    # Save gallery
    gallery_path = Path(face_cfg.gallery_path)
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    with open(gallery_path, "wb") as f:
        pickle.dump(gallery, f)

    print(f"\nGallery saved: {gallery_path} ({len(gallery)} people)")
    return gallery
