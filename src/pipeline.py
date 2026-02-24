"""Pipeline orchestrator — single-pass frame loop tying all modules together."""

from src.utils.config import PipelineConfig


class Pipeline:
    """Main pipeline that processes a video through all analysis modules."""

    def __init__(self, config: PipelineConfig, annotate: bool = True):
        raise NotImplementedError

    def run(self, video_path: str, output_dir: str) -> dict:
        """Process a video file and return structured results."""
        raise NotImplementedError
