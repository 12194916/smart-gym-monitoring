"""Structured JSON output writer."""


def build_result(video_path: str, metadata: dict, identification: dict, exercise: dict, rep_counter) -> dict:
    """Build the final structured JSON result dict."""
    raise NotImplementedError


def save_result(result: dict, output_path: str):
    """Save result dict to a JSON file."""
    raise NotImplementedError
