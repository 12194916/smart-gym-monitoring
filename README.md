# Smart Gym Monitoring

A computer vision pipeline for real-time exercise classification from video and webcam feeds using pose estimation and machine learning.

## How It Works

```
Video/Webcam → YOLOv8 Person Detection → MediaPipe Pose (33 landmarks)
    → 132 Raw Landmark Features → ML Classifier (ExtraTrees) → Stabilized Label
    → Annotated Output (skeleton + exercise label overlay)
```

1. **Person detection**: YOLOv8 nano finds the person in frame, scored by bounding box size + center proximity with IoU-based tracking across frames.
2. **Pose estimation**: MediaPipe PoseLandmarker Heavy extracts all 33 pose landmarks in normalized coordinates.
3. **Exercise classification**: The 132 raw features (33 landmarks x 4 values: x, y, z, visibility) are fed directly to a trained ExtraTrees classifier. A majority-vote stabilizer smooths predictions over a sliding window to prevent label flickering.
4. **Rep counting**: *(planned)* State machine tracking per-exercise joint angles for repetition counting.

## Model Performance

Trained on the [Kaggle Gym Workout/Exercises Video Dataset](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video) with 15-frame sampling per video and per-class balancing (max 2,000 samples/class).

| Model | 5-Fold CV Accuracy | Holdout Accuracy (80/20 split) |
|---|---|---|
| ExtraTrees_300 | **96.1%** | **96.9%** |
| RandomForest_300 | 95.5% | — |
| RandomForest_500 | 95.5% | — |

Best model: **ExtraTrees** (300 trees, max_depth=25, balanced class weights). Saved as `data/models/exercise_classifier.pkl`.

## Supported Exercises (22 classes)

barbell biceps curl, bench press, chest fly machine, deadlift, decline bench press, hammer curl, hip thrust, incline bench press, lat pulldown, lateral raise, leg extension, leg raises, plank, pull up, push-up, romanian deadlift, russian twist, shoulder press, squat, t bar row, tricep pushdown, tricep dips

## Known Issues

- **Frame-level label switching**: The classifier predicts independently per frame. During an exercise, certain body positions can momentarily resemble a different exercise class, causing the label to briefly switch. The stabilizer window helps but does not fully eliminate this. A planned fix is to accumulate predictions over a fixed number of frames and commit to a label only after sufficient agreement, rather than updating every frame.

## Project Structure

```
gym-monitoring/
├── main.py                           # Entry point (not yet integrated)
├── config.yaml                       # All configurable parameters
├── requirements.txt
│
├── scripts/
│   ├── extract_pose_features.py      # Extract 132 raw landmark features from training videos
│   ├── experiment.py                 # Train & compare classifiers (5-fold CV + holdout)
│   ├── evaluate.py                   # Evaluate model on unseen test videos
│   └── predict.py                    # Live prediction from video file or webcam
│
├── src/
│   ├── pose/
│   │   ├── person_detector.py        # YOLOv8 person detection with IoU tracking
│   │   ├── estimator.py              # MediaPipe PoseLandmarker (33 landmarks + 12 body)
│   │   └── angles.py                 # Joint angle computation (used by rep counter)
│   ├── exercise/
│   │   ├── definitions.py            # Exercise enum (15 exercises + UNKNOWN)
│   │   ├── classifier.py             # Raw landmark classifier wrapper
│   │   └── stabilizer.py             # Majority-vote label smoothing
│   ├── rep_counter/
│   │   ├── counter.py                # State machine rep counter (planned)
│   │   └── rep_rules.py              # Per-exercise angle thresholds (planned)
│   ├── output/
│   │   ├── video_annotator.py        # Skeleton + HUD overlay drawing
│   │   └── json_writer.py            # JSON output
│   └── utils/
│       ├── geometry.py               # Vector math, angle calculation
│       ├── video_io.py               # OpenCV video read/write helpers
│       └── config.py                 # YAML → typed dataclass config loader
│
├── notebooks/
│   ├── analysis.ipynb                # Feature exploration & visualization
│   └── pose-model-final-2.ipynb      # Kaggle notebook (reference model)
│
├── data/
│   ├── videos/                       # Training + test videos (not in git)
│   │   ├── data_btc_10s/            # Rep-aligned clips (~22 classes)
│   │   ├── data_crawl_10s/          # Sliding window clips
│   │   ├── raw_btc/                 # Raw clips (additional data)
│   │   ├── raw_crawl/               # Raw clips (additional data)
│   │   └── test/                    # Unseen test videos
│   ├── models/                       # Model files (not in git)
│   │   ├── yolov8n.pt              # YOLOv8 nano person detector
│   │   ├── pose_landmarker_heavy.task # MediaPipe pose model (heavy variant)
│   │   └── exercise_classifier.pkl  # Trained ExtraTrees classifier
│   └── landmarks.csv                # Extracted landmark features (not in git)
│
└── tests/
```

## Setup

```bash
conda create -n gym-monitoring python=3.10 -y
conda activate gym-monitoring
pip install -r requirements.txt
```

Download model files into `data/models/`:
- `yolov8n.pt` — auto-downloaded by ultralytics on first run
- `pose_landmarker_heavy.task` — from [MediaPipe model catalog](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#models)

## Training Data

Source: [Kaggle Gym Workout/Exercises Video Dataset](https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video)

- ~1,600 video clips across 22 exercise categories
- Pre-processed clips (10-13 seconds), MP4 format
- Four training directories: `data_btc_10s`, `data_crawl_10s`, `raw_btc`, `raw_crawl`
- 15 frames sampled per video at fixed proportions (10%, 15%, 20%, ..., 95% of total frames)

## Usage

### Train a model

```bash
# 1. Extract raw landmark features from all training videos
python scripts/extract_pose_features.py --all --output data/landmarks.csv

# 2. Train classifiers and save best model
python scripts/experiment.py --features data/landmarks.csv

# 3. Evaluate on unseen test videos
python scripts/evaluate.py --test-dir data/videos/test
```

### Predict

```bash
# From a video file
python scripts/predict.py --source "data/videos/test/deadlift/deadlift_1.mp4"

# Live webcam
python scripts/predict.py --source 0

# Without YOLO bounding box (pose + classification only)
python scripts/predict.py --source 0 --no-yolo
```

Controls:
- `q` — quit
- `r` — reset stabilizer and tracking

### Test extraction on a single video

```bash
python scripts/extract_pose_features.py --video "data/videos/data_btc_10s/squat/some_video.mp4"
```

## Configuration

All thresholds, model paths, and parameters are in `config.yaml`:

| Setting | Default | Description |
|---|---|---|
| `pose.model_path` | `pose_landmarker_heavy.task` | MediaPipe pose model variant |
| `exercise.stabilizer_window` | 5 | Majority voting window to smooth predictions |
| `exercise.classifier_model_path` | `exercise_classifier.pkl` | Trained sklearn classifier |
| `person_detection.confidence_threshold` | 0.5 | YOLO detection threshold |
| `person_detection.iou_threshold` | 0.3 | IoU threshold for person tracking |
| `person_detection.max_missing_frames` | 5 | Frames before tracking resets |

## Hardware

Tested on Apple M4 Pro (24GB). YOLOv8 uses MPS (Metal) GPU acceleration when available, falls back to CPU.
