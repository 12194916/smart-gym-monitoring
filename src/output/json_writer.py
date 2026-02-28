"""Structured JSON output writer.

Produces a JSON result file containing video metadata, identified person,
exercise classification, and rep counting details with per-rep timestamps.
"""

import json
from pathlib import Path
from typing import Optional


def build_result(
    video_path: str,
    fps: float,
    total_frames: int,
    identified_user: Optional[str],
    identification_confidence: float,
    exercise_label: str,
    exercise_confidence: float,
    rep_count: int,
    rep_events: list,
    is_hold: bool = False,
) -> dict:
    """Build the final structured JSON result dict.

    Args:
        video_path: Path to the input video.
        fps: Video frames per second.
        total_frames: Total number of frames processed.
        identified_user: Recognized person name (or None).
        identification_confidence: Face recognition similarity score.
        exercise_label: Stabilized exercise classification label.
        exercise_confidence: Exercise classification confidence.
        rep_count: Total rep count (or hold duration in seconds for plank).
        rep_events: List of RepEvent objects with per-rep details.
        is_hold: Whether the exercise is a static hold (e.g., plank).

    Returns:
        Structured result dict ready for JSON serialization.
    """
    duration_s = total_frames / fps if fps > 0 else 0

    result = {
        "video": {
            "path": str(video_path),
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_seconds": round(duration_s, 2),
        },
        "identification": {
            "user": identified_user or "unknown",
            "confidence": round(identification_confidence, 4),
        },
        "exercise": {
            "label": exercise_label,
            "confidence": round(exercise_confidence, 4),
        },
        "rep_counting": {
            "type": "hold_duration_seconds" if is_hold else "rep_count",
            "count": rep_count,
            "reps": [
                {
                    "rep_number": r.rep_number,
                    "frame": r.frame_number,
                    "timestamp_ms": round(r.timestamp_ms, 1),
                    "angle_at_count": round(r.angle_at_count, 1),
                }
                for r in rep_events
            ],
        },
    }

    return result


def save_result(result: dict, output_path: str, indent: int = 2):
    """Save result dict to a JSON file.

    Args:
        result: Structured result dict from build_result().
        output_path: Path for the output JSON file.
        indent: JSON indentation (default: 2).
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(result, f, indent=indent)

    print(f"Results saved → {output}")
