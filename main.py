"""Main entry point for gym monitoring pipeline.

Usage:
    python main.py <video_path>
    python main.py <video_path> --no-annotate
    python main.py <video_path> --output results/
"""

import argparse
import sys
import os

from src.utils.config import load_config
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(description="Gym Video Monitoring Pipeline")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip annotated video output (JSON only)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    config = load_config(args.config)

    pipeline = Pipeline(config, annotate=not args.no_annotate)
    result = pipeline.run(args.video, output_dir=args.output)

    print(f"\n--- Result ---")
    print(f"  User:     {result['identification']['user']}")
    print(f"  Exercise: {result['exercise']['label']} ({result['exercise']['confidence']:.0%})")
    reps = result['rep_counting']
    if reps['type'] == 'hold_duration_seconds':
        print(f"  Hold:     {reps['count']}s")
    else:
        print(f"  Reps:     {reps['count']}")


if __name__ == "__main__":
    main()
