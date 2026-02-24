"""Main entry point for gym monitoring pipeline.

Usage:
    python main.py <video_path>

Modules are added step by step as they are developed:
    Step 1: Exercise classification (pose + angles + classifier)
    Step 2: Rep counting
    Step 3: Face identification
    Step 4: Full pipeline with JSON output + annotated video
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Gym Video Monitoring Pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    # --- Step 1: Exercise Classification ---
    # TODO: Implement after building pose estimator + angle computation + classifier
    print(f"Processing video: {args.video}")
    print("Exercise classification not yet implemented.")


if __name__ == "__main__":
    main()
