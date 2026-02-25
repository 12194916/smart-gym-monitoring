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
                 iou_threshold: float = 0.3, max_missing_frames: int = 5,
                 device: str = "auto"):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.center_weight = center_weight
        self.size_weight = size_weight
        self.iou_threshold = iou_threshold
        self.max_missing_frames = max_missing_frames

        # Tracking state
        self._prev_box: Optional[PersonBox] = None
        self._frames_without_match: int = 0

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
        """Detect people in frame and return the best candidate, or None.

        Uses IoU-based tracking to prefer the same person across frames.
        Falls back to size+center scoring when no previous detection exists
        or tracking is lost.
        """
        h, w = frame.shape[:2]
        frame_center_x = w / 2
        frame_center_y = h / 2
        frame_area = w * h

        results = self.model(frame, verbose=False, classes=[self.PERSON_CLASS],
                             conf=self.confidence_threshold, device=self.device)

        persons = self._extract_persons(results, frame_area, frame_center_x, frame_center_y)

        if not persons:
            self._frames_without_match += 1
            if self._frames_without_match > self.max_missing_frames:
                self._prev_box = None
            return None

        if len(persons) == 1 and self._prev_box is None:
            selected = persons[0]
        else:
            selected = self._select_best_with_tracking(persons, w, h)

        self._prev_box = selected
        self._frames_without_match = 0
        return selected

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

    def _select_best_with_tracking(self, persons: List[PersonBox],
                                   w: int, h: int) -> PersonBox:
        """Score each person using center proximity, size, and IoU tracking.

        When tracking a person (prev_box exists), IoU continuity is added
        to the score so the same person is preferred across frames. Center
        and size are always factored in so a small/edge person doesn't get
        locked onto permanently.

        Score = size_weight * size + center_weight * center + iou_bonus
        where iou_bonus = 0.5 * iou (only when tracking)
        """
        max_area = max(p.area for p in persons)
        max_dist = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

        best = None
        best_score = -1.0

        for p in persons:
            size_score = p.area / max_area if max_area > 0 else 0
            dist = np.sqrt((p.center_x - w / 2) ** 2 + (p.center_y - h / 2) ** 2)
            center_score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0

            score = self.size_weight * size_score + self.center_weight * center_score

            # Add IoU tracking bonus when we have a previous detection
            if self._prev_box is not None:
                iou = self._compute_iou(self._prev_box, p)
                if iou >= self.iou_threshold:
                    score += 0.5 * iou

            if score > best_score:
                best_score = score
                best = p

        return best

    @staticmethod
    def _compute_iou(a: PersonBox, b: PersonBox) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        inter_x1 = max(a.x1, b.x1)
        inter_y1 = max(a.y1, b.y1)
        inter_x2 = min(a.x2, b.x2)
        inter_y2 = min(a.y2, b.y2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = a.area + b.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def reset_tracking(self):
        """Clear tracking state. Call when resetting the session."""
        self._prev_box = None
        self._frames_without_match = 0

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
