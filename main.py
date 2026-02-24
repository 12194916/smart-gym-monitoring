"""Main entry point for gym monitoring pipeline.

Usage:
    python main.py <video_path>

Modules are added step by step as they are developed:
    Step 1: Exercise classification
    Step 2: Rep counting
    Step 3: Face identification
    Step 4: Full pipeline with JSON output + annotated video
"""

import argparse
import sys
import os

from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Gym Video Monitoring Pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    config = load_config(args.config)

    print(f"Processing video: {args.video}")
    print("Pipeline modules not yet integrated. Use scripts/ for training.")


if __name__ == "__main__":
    main()
