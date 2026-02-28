"""Pipeline orchestrator — single-pass frame loop tying all modules together.

Processes a video through:
    1. Face identification (first N frames → lock identity)
    2. Pose estimation (MediaPipe multi-pose with best-person selection)
    3. Exercise classification (raw landmark → ExtraTrees → EMA stabilizer)
    4. Rep counting (peak-valley on joint angles)
    5. Annotation (skeleton + label + rep count overlay)
    6. JSON output (structured results with per-rep timestamps)
"""

import os

import cv2
import joblib
import numpy as np

from src.utils.config import PipelineConfig
from src.pose.estimator import PoseEstimator
from src.exercise.stabilizer import ClassificationStabilizer
from src.rep_counter.counter import RepCounterManager
from src.face_id.recognize import FaceRecognizer
from src.output.json_writer import build_result, save_result


# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _draw_skeleton(frame, body_landmarks):
    """Draw pose skeleton on the frame."""
    if body_landmarks is None:
        return
    for name_a, name_b in SKELETON_CONNECTIONS:
        if name_a in body_landmarks and name_b in body_landmarks:
            a = body_landmarks[name_a]
            b = body_landmarks[name_b]
            if a.visibility > 0.5 and b.visibility > 0.5:
                cv2.line(frame, (int(a.x), int(a.y)), (int(b.x), int(b.y)), (0, 255, 0), 2)
    for name, lm in body_landmarks.items():
        if lm.visibility > 0.5:
            cv2.circle(frame, (int(lm.x), int(lm.y)), 4, (0, 0, 255), -1)


CORNER_LEN = 0.15
CORNER_THICKNESS = 2
FACE_BOX_COLOR = (255, 200, 0)


def _draw_face_box(frame, bbox, name, similarity):
    """Draw corner-bracket face box with person name above."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    cl_x = max(int(w * CORNER_LEN), 8)
    cl_y = max(int(h * CORNER_LEN), 8)

    # Corner brackets
    cv2.line(frame, (x1, y1), (x1 + cl_x, y1), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x1, y1), (x1, y1 + cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y1), (x2 - cl_x, y1), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y1), (x2, y1 + cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x1, y2), (x1 + cl_x, y2), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x1, y2), (x1, y2 - cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y2), (x2 - cl_x, y2), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y2), (x2, y2 - cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)

    # Name label above box
    label = f"{name} ({similarity:.0%})" if name else "unknown"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, 0.6, 2)
    pad = 4
    cv2.rectangle(frame, (x1, y1 - th - 2 * pad), (x1 + tw + 2 * pad, y1), FACE_BOX_COLOR, -1)
    cv2.putText(frame, label, (x1 + pad, y1 - pad), font, 0.6, (0, 0, 0), 2)


def _draw_hud(frame, exercise, exercise_conf, rep_count, angle, is_hold, identity, identity_conf):
    """Draw HUD overlay with identity, exercise label, and rep count."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Exercise label
    ex_text = f"{exercise} ({exercise_conf:.0%})"
    (tw, th), bl = cv2.getTextSize(ex_text, font, 1.0, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + bl), (0, 0, 0), -1)
    cv2.putText(frame, ex_text, (15, 15 + th), font, 1.0, (0, 255, 255), 2)

    # Rep count
    if is_hold:
        rep_text = f"Hold: {rep_count}s"
    else:
        rep_text = f"Reps: {rep_count}"
    if angle is not None:
        rep_text += f"  [{angle:.0f}]"
    (tw, th), bl = cv2.getTextSize(rep_text, font, 0.8, 2)
    y = 55
    cv2.rectangle(frame, (10, y), (20 + tw, y + 10 + th + bl), (0, 0, 0), -1)
    cv2.putText(frame, rep_text, (15, y + 5 + th), font, 0.8, (0, 255, 128), 2)

    # Locked identity banner (bottom-left)
    if identity:
        id_text = f"ID: {identity} ({identity_conf:.0%})"
        (tw, th), bl = cv2.getTextSize(id_text, font, 0.6, 2)
        y = frame.shape[0] - 40
        cv2.rectangle(frame, (10, y - 5), (20 + tw, y + th + bl + 5), (0, 0, 0), -1)
        cv2.putText(frame, id_text, (15, y + th), font, 0.6, FACE_BOX_COLOR, 2)


class Pipeline:
    """Main pipeline that processes a video through all analysis modules."""

    def __init__(self, config: PipelineConfig, annotate: bool = True):
        self.config = config
        self.annotate = annotate

        # Pose estimator
        self.pose = PoseEstimator(
            model_path=config.pose.model_path,
            min_detection_confidence=config.pose.min_detection_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
            num_poses=config.pose.max_poses,
            size_weight=config.pose.person_size_weight,
            center_weight=config.pose.person_center_weight,
        )

        # Exercise classifier
        self.model = joblib.load(config.exercise.classifier_model_path)

        # Label stabilizer
        self.stabilizer = ClassificationStabilizer(
            ema_alpha=config.exercise.stabilizer_ema_alpha,
            hysteresis_margin=config.exercise.stabilizer_hysteresis_margin,
            min_hold_frames=config.exercise.stabilizer_min_hold_frames,
            min_confidence=config.exercise.min_classification_score,
        )

        # Face recognizer
        self.recognizer = FaceRecognizer.from_config(config)

    def run(self, video_path: str, output_dir: str = "outputs") -> dict:
        """Process a video file and return structured results.

        Args:
            video_path: Path to the input video.
            output_dir: Directory for annotated video and JSON output.

        Returns:
            Structured result dict (also saved as JSON).
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        max_id_frames = self.config.face.max_identification_frames

        # Rep counter
        rep_manager = RepCounterManager(
            fps=fps,
            min_rep_duration_ms=self.config.rep_counter.min_rep_duration_ms,
            angle_smoothing_window=self.config.rep_counter.angle_smoothing_window,
        )

        # Video writer
        os.makedirs(output_dir, exist_ok=True)
        src_name = os.path.splitext(os.path.basename(video_path))[0]
        video_out_path = os.path.join(output_dir, f"{src_name}_annotated.mp4")
        json_out_path = os.path.join(output_dir, f"{src_name}_result.json")

        video_writer = None
        if self.annotate:
            codec = cv2.VideoWriter_fourcc(*self.config.output.annotated_video_codec)
            video_writer = cv2.VideoWriter(video_out_path, codec, fps, (w, h))

        # State
        stable_label = "..."
        stable_confidence = 0.0
        identified_user = None
        id_confidence = 0.0
        id_votes = {}  # {name: [similarity_scores]}
        frame_count = 0

        print(f"Processing: {video_path}")
        print(f"  {total_frames} frames, {fps:.0f} fps, {w}x{h}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            # --- Face detection + identification ---
            face_bbox = None
            face_name_display = None
            face_sim_display = 0.0

            faces = self.recognizer._app.get(frame)
            if faces:
                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                face_bbox = best_face.bbox

                if identified_user is None and frame_count <= max_id_frames:
                    query = best_face.embedding
                    query = query / np.linalg.norm(query)
                    similarities = self.recognizer._embeddings @ query
                    best_idx = int(np.argmax(similarities))
                    best_sim = float(similarities[best_idx])

                    if best_sim >= self.recognizer._threshold:
                        name = self.recognizer._names[best_idx]
                        id_votes.setdefault(name, []).append(best_sim)
                        face_name_display = name
                        face_sim_display = best_sim

                    if frame_count == max_id_frames and id_votes:
                        best_name = max(id_votes, key=lambda n: np.mean(id_votes[n]))
                        identified_user = best_name
                        id_confidence = float(np.mean(id_votes[best_name]))
                        print(f"  Identified: {identified_user} (confidence: {id_confidence:.2%})")

                elif identified_user is not None:
                    face_name_display = identified_user
                    face_sim_display = id_confidence

            # --- Pose estimation ---
            raw_landmarks, body_landmarks = self.pose.process_full_frame(frame)

            # --- Exercise classification ---
            if raw_landmarks is not None:
                X = np.array(raw_landmarks).reshape(1, -1)
                proba_array = self.model.predict_proba(X)[0]
                class_probs = {
                    cls: float(prob)
                    for cls, prob in zip(self.model.classes_, proba_array)
                }
                label, conf = self.stabilizer.update_with_probabilities(class_probs)
                if label is not None:
                    stable_label = label
                    stable_confidence = conf

            # --- Rep counting ---
            rep_count, tracked_angle = rep_manager.update(
                exercise=stable_label if stable_label != "..." else None,
                body_landmarks=body_landmarks if body_landmarks else {},
                frame_number=frame_count,
            )

            # --- Annotation ---
            if self.annotate:
                if body_landmarks is not None:
                    _draw_skeleton(frame, body_landmarks)
                if face_bbox is not None:
                    _draw_face_box(frame, face_bbox, face_name_display, face_sim_display)
                is_hold = stable_label == "plank"
                _draw_hud(
                    frame, stable_label, stable_confidence,
                    rep_count, tracked_angle, is_hold,
                    identified_user, id_confidence,
                )
                cv2.putText(
                    frame, f"Frame {frame_count}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1,
                )
                if video_writer is not None:
                    video_writer.write(frame)

        # Lock identity if video ended before max_id_frames
        if identified_user is None and id_votes:
            best_name = max(id_votes, key=lambda n: np.mean(id_votes[n]))
            identified_user = best_name
            id_confidence = float(np.mean(id_votes[best_name]))
            print(f"  Identified: {identified_user} (confidence: {id_confidence:.2%})")

        # Cleanup
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"  Annotated video → {video_out_path}")
        self.pose.close()

        # Build and save JSON result
        counter = rep_manager.current_counter
        rep_events = counter.reps if counter else []
        is_hold = stable_label == "plank"

        result = build_result(
            video_path=video_path,
            fps=fps,
            total_frames=frame_count,
            identified_user=identified_user,
            identification_confidence=id_confidence,
            exercise_label=stable_label,
            exercise_confidence=stable_confidence,
            rep_count=rep_count,
            rep_events=rep_events,
            is_hold=is_hold,
        )
        save_result(result, json_out_path, indent=self.config.output.json_indent)

        return result
