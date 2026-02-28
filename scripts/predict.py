"""Live / video exercise prediction with face ID and rep counting.

Runs the full pipeline in real-time with cv2.imshow:
    Face detection + identification → MediaPipe multi-pose → raw landmark
    classifier → EMA-stabilized label → joint angle rep counter → annotated overlay.

Usage:
    python scripts/predict.py --source 0                    # webcam
    python scripts/predict.py --source data/videos/test/squat/video1.mp4
    python scripts/predict.py --source data/videos/test/squat/video1.mp4 --save
"""

import argparse
import os
import sys

import cv2
import joblib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.pose.estimator import PoseEstimator, BODY_LANDMARK_INDICES
from src.exercise.stabilizer import ClassificationStabilizer
from src.rep_counter.counter import RepCounterManager
from src.face_id.recognize import FaceRecognizer


# Skeleton connections for drawing (pairs of landmark names)
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

# Corner bracket length (fraction of box side)
CORNER_LEN = 0.15
CORNER_THICKNESS = 2
FACE_BOX_COLOR = (255, 200, 0)  # cyan-gold
FACE_TEXT_COLOR = (255, 200, 0)


def draw_skeleton(frame, body_landmarks):
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


def draw_face_box(frame, bbox, name, similarity):
    """Draw stylish corner-bracket face box with person name above.

    Args:
        frame: BGR image.
        bbox: (x1, y1, x2, y2) face bounding box.
        name: Identified person name (or None).
        similarity: Cosine similarity score.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    cl_x = max(int(w * CORNER_LEN), 8)
    cl_y = max(int(h * CORNER_LEN), 8)

    # Top-left corner
    cv2.line(frame, (x1, y1), (x1 + cl_x, y1), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x1, y1), (x1, y1 + cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    # Top-right corner
    cv2.line(frame, (x2, y1), (x2 - cl_x, y1), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y1), (x2, y1 + cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    # Bottom-left corner
    cv2.line(frame, (x1, y2), (x1 + cl_x, y2), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x1, y2), (x1, y2 - cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)
    # Bottom-right corner
    cv2.line(frame, (x2, y2), (x2 - cl_x, y2), FACE_BOX_COLOR, CORNER_THICKNESS)
    cv2.line(frame, (x2, y2), (x2, y2 - cl_y), FACE_BOX_COLOR, CORNER_THICKNESS)

    # Name label above the box
    if name:
        label = f"{name} ({similarity:.0%})"
    else:
        label = "unknown"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    # Background pill behind text
    pad = 4
    cv2.rectangle(
        frame,
        (x1, y1 - th - 2 * pad),
        (x1 + tw + 2 * pad, y1),
        FACE_BOX_COLOR, -1,
    )
    cv2.putText(
        frame, label,
        (x1 + pad, y1 - pad),
        font, font_scale, (0, 0, 0), thickness,
    )


def draw_label(frame, label, confidence):
    """Draw exercise label and confidence on the frame."""
    text = f"{label} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(text, font, 1.0, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 15 + th), font, 1.0, (0, 255, 255), 2)


def draw_rep_count(frame, rep_count, angle, is_hold=False):
    """Draw rep counter (or hold duration) below the exercise label."""
    if is_hold:
        text = f"Hold: {rep_count}s"
    else:
        text = f"Reps: {rep_count}"
    if angle is not None:
        text += f"  [{angle:.0f}]"

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 65
    (tw, th), baseline = cv2.getTextSize(text, font, 0.8, 2)
    cv2.rectangle(frame, (10, y_offset), (20 + tw, y_offset + 10 + th + baseline), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, y_offset + 5 + th), font, 0.8, (0, 255, 128), 2)


def draw_identity_banner(frame, name, confidence):
    """Draw locked identity in bottom-left corner."""
    if name is None:
        return
    text = f"ID: {name} ({confidence:.0%})"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 0.6, 2)
    y = frame.shape[0] - 40
    cv2.rectangle(frame, (10, y - 5), (20 + tw, y + th + baseline + 5), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, y + th), font, 0.6, FACE_TEXT_COLOR, 2)


def main():
    parser = argparse.ArgumentParser(description="Live exercise prediction with face ID")
    parser.add_argument("--source", default="0",
                        help="Video file path or 0 for webcam")
    parser.add_argument("--model", default=None,
                        help="Path to classifier model (default: from config)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated video to outputs/")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Load classifier
    model_path = args.model or config.exercise.classifier_model_path
    if not os.path.exists(model_path):
        print(f"Error: Model not found: {model_path}")
        print("Run: python scripts/experiment.py")
        sys.exit(1)

    print(f"Loading classifier: {model_path}")
    model = joblib.load(model_path)
    print(f"  Classes: {list(model.classes_)}")

    # Init pose estimator
    print(f"Loading pose model: {config.pose.model_path}")
    pose = PoseEstimator(
        model_path=config.pose.model_path,
        min_detection_confidence=config.pose.min_detection_confidence,
        min_tracking_confidence=config.pose.min_tracking_confidence,
        num_poses=config.pose.max_poses,
        size_weight=config.pose.person_size_weight,
        center_weight=config.pose.person_center_weight,
    )

    # Init face recognizer
    print("Loading face recognizer...")
    try:
        recognizer = FaceRecognizer.from_config(config)
        face_id_enabled = True
        print(f"  Gallery loaded ({len(recognizer._names)} people)")
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  Face identification disabled. Run: python scripts/enroll_faces.py")
        face_id_enabled = False
        recognizer = None

    # Open video source
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Cannot open source: {args.source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_webcam = isinstance(source, int)
    print(f"Source: {args.source} ({'webcam' if is_webcam else f'{total_frames} frames, {fps:.0f} fps'})")

    # Video writer for --save
    video_writer = None
    save_path = None
    if args.save:
        os.makedirs("outputs", exist_ok=True)
        if isinstance(source, int):
            src_name = "webcam"
        else:
            src_name = os.path.splitext(os.path.basename(args.source))[0]
        save_path = os.path.join("outputs", f"{src_name}_annotated.mp4")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*config.output.annotated_video_codec)
        video_writer = cv2.VideoWriter(save_path, codec, fps, (w, h))
        print(f"Saving annotated video → {save_path}")

    # Stabilizer
    stabilizer = ClassificationStabilizer(
        ema_alpha=config.exercise.stabilizer_ema_alpha,
        hysteresis_margin=config.exercise.stabilizer_hysteresis_margin,
        min_hold_frames=config.exercise.stabilizer_min_hold_frames,
        min_confidence=config.exercise.min_classification_score,
    )
    stable_label = "..."
    stable_confidence = 0.0

    # Rep counter
    rep_manager = RepCounterManager(
        fps=fps,
        min_rep_duration_ms=config.rep_counter.min_rep_duration_ms,
        angle_smoothing_window=config.rep_counter.angle_smoothing_window,
    )

    # Face ID state
    max_id_frames = config.face.max_identification_frames
    identified_user = None
    id_confidence = 0.0
    id_votes = {}
    last_face_bbox = None  # for drawing the face box

    frame_count = 0
    print("\nPress 'q' to quit, 'r' to reset stabilizer + rep counter\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_webcam:
                continue
            break

        frame_count += 1

        # --- Face identification ---
        face_bbox = None
        face_name_display = None
        face_sim_display = 0.0

        if face_id_enabled and recognizer is not None:
            # Run face detection on every frame for the box drawing,
            # but only accumulate votes during identification window
            faces = recognizer._app.get(frame)
            if faces:
                best_face = max(
                    faces,
                    key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                )
                face_bbox = best_face.bbox

                if identified_user is None and frame_count <= max_id_frames:
                    # Compute similarity for voting
                    query = best_face.embedding
                    query = query / np.linalg.norm(query)
                    similarities = recognizer._embeddings @ query
                    best_idx = int(np.argmax(similarities))
                    best_sim = float(similarities[best_idx])

                    if best_sim >= recognizer._threshold:
                        name = recognizer._names[best_idx]
                        id_votes.setdefault(name, []).append(best_sim)
                        face_name_display = name
                        face_sim_display = best_sim
                    else:
                        face_name_display = None
                        face_sim_display = best_sim

                    # Lock identity after enough frames
                    if frame_count == max_id_frames and id_votes:
                        best_name = max(id_votes, key=lambda n: np.mean(id_votes[n]))
                        identified_user = best_name
                        id_confidence = float(np.mean(id_votes[best_name]))
                        print(f"  Identified: {identified_user} ({id_confidence:.2%})")

                elif identified_user is not None:
                    face_name_display = identified_user
                    face_sim_display = id_confidence

        # --- Pose estimation ---
        raw_landmarks, body_landmarks = pose.process_full_frame(frame)

        # --- Exercise classification ---
        if raw_landmarks is not None:
            X = np.array(raw_landmarks).reshape(1, -1)
            proba_array = model.predict_proba(X)[0]
            class_probs = {
                cls: float(prob)
                for cls, prob in zip(model.classes_, proba_array)
            }

            label, conf = stabilizer.update_with_probabilities(class_probs)
            if label is not None:
                stable_label = label
                stable_confidence = conf

        # --- Rep counting ---
        rep_count, tracked_angle = rep_manager.update(
            exercise=stable_label if stable_label != "..." else None,
            body_landmarks=body_landmarks if body_landmarks else {},
            frame_number=frame_count,
        )

        # --- Draw ---
        if body_landmarks is not None:
            draw_skeleton(frame, body_landmarks)

        # Face box with corner brackets + name
        if face_bbox is not None:
            draw_face_box(frame, face_bbox, face_name_display, face_sim_display)

        draw_label(frame, stable_label, stable_confidence)
        is_hold = stable_label == "plank"
        draw_rep_count(frame, rep_count, tracked_angle, is_hold=is_hold)

        # Locked identity banner
        draw_identity_banner(frame, identified_user, id_confidence)

        # Frame counter
        cv2.putText(frame, f"Frame {frame_count}", (10, frame.shape[0] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if video_writer is not None:
            video_writer.write(frame)

        cv2.imshow("Exercise Prediction", frame)

        key = cv2.waitKey(1 if is_webcam else int(1000 / fps)) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            stabilizer.reset()
            rep_manager.reset()
            identified_user = None
            id_confidence = 0.0
            id_votes.clear()
            stable_label = "..."
            stable_confidence = 0.0
            print("Reset all (stabilizer + rep counter + face ID)")

    # Lock identity if video ended before max_id_frames
    if identified_user is None and id_votes:
        best_name = max(id_votes, key=lambda n: np.mean(id_votes[n]))
        identified_user = best_name
        id_confidence = float(np.mean(id_votes[best_name]))

    if video_writer is not None:
        video_writer.release()
        print(f"Saved annotated video → {save_path}")
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    print(f"\nProcessed {frame_count} frames")
    if identified_user:
        print(f"Identified: {identified_user} ({id_confidence:.2%})")


if __name__ == "__main__":
    main()
