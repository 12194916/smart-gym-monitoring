"""YOLOv8-based person detection with smart selection logic.

Selects the most relevant person in frame using two criteria:
1. Largest bounding box (likely the main subject)
2. Closest to center of frame (likely the intended person)

These are combined with configurable weights to produce a final score.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from ultralytics import YOLO


@dataclass
class PersonBox:
    """Detected person bounding box."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    area: float
    center_x: float
    center_y: float


class PersonDetector:
    """Detects people in frames using YOLOv8 and selects the best candidate."""

    # COCO class index for 'person'
    PERSON_CLASS = 0

    def __init__(self, model_path: str, confidence_threshold: float = 0.5,
                 center_weight: float = 0.4, size_weight: float = 0.6,
                 device: str = "auto"):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.center_weight = center_weight
        self.size_weight = size_weight

        # Select best available device: MPS (Apple Silicon GPU) > CPU
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        print(f"  PersonDetector using device: {self.device}")

    def detect(self, frame: np.ndarray) -> Optional[PersonBox]:
        """Detect people in frame and return the best candidate, or None."""
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2
        frame_area = w * h

        results = self.model(frame, verbose=False, classes=[self.PERSON_CLASS],
                             conf=self.confidence_threshold, device=self.device)

        persons = self._extract_persons(results, frame_area, frame_center_x, frame_center_y)

        if not persons:
            return None

        if len(persons) == 1:
            return persons[0]

        return self._select_best(persons, w, h, frame_area)

    def _extract_persons(self, results, frame_area: float,
                         frame_center_x: float, frame_center_y: float) -> List[PersonBox]:
        """Extract PersonBox objects from YOLO results."""
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls = int(boxes.cls[i])
                if cls != self.PERSON_CLASS:
                    continue
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                area = float((x2 - x1) * (y2 - y1))
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                persons.append(PersonBox(
                    x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                    confidence=conf, area=area,
                    center_x=cx, center_y=cy,
                ))
        return persons

    def _select_best(self, persons: List[PersonBox], w: int, h: int,
                     frame_area: float) -> PersonBox:
        """Score each person and return the best one.

        Score = size_weight * (area / max_area) + center_weight * (1 - dist / max_dist)
        """
        max_area = max(p.area for p in persons)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)  # corner to center distance

        best = None
        best_score = -1.0

        for p in persons:
            size_score = p.area / max_area if max_area > 0 else 0
            dist = np.sqrt((p.center_x - w / 2) ** 2 + (p.center_y - h / 2) ** 2)
            center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
            score = self.size_weight * size_score + self.center_weight * center_score

            if score > best_score:
                best_score = score
                best = p

        return best

    def get_padded_crop(self, frame: np.ndarray, box: PersonBox,
                        padding: float = 0.15) -> tuple:
        """Get a padded crop of the person from the frame.

        Args:
            frame: Full frame.
            box: Detected person bounding box.
            padding: Fraction of bbox size to add as padding.

        Returns:
            (cropped_frame, crop_x1, crop_y1) - the crop and its top-left offset
            in full-frame coordinates.
        """
        h, w = frame.shape[:2]
        bw = box.x2 - box.x1
        bh = box.y2 - box.y1
        pad_x = int(bw * padding)
        pad_y = int(bh * padding)

        crop_x1 = max(0, box.x1 - pad_x)
        crop_y1 = max(0, box.y1 - pad_y)
        crop_x2 = min(w, box.x2 + pad_x)
        crop_y2 = min(h, box.y2 + pad_y)

        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        return crop, crop_x1, crop_y1
