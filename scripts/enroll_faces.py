"""Build face embedding gallery from extracted face images.

Usage:
    python scripts/enroll_faces.py
    python scripts/enroll_faces.py --config config.yaml
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.face_id.enroll import build_gallery


def main():
    parser = argparse.ArgumentParser(description="Enroll faces into gallery")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    print(f"Enrollment directory: {config.face.enrollment_dir}")
    print(f"Gallery output: {config.face.gallery_path}\n")

    gallery = build_gallery(config)
    print(f"\nEnrolled {len(gallery)} people: {list(gallery.keys())}")


if __name__ == "__main__":
    main()
