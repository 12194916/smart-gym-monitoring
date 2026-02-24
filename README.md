# Smart Gym Monitoring

A computer vision pipeline that analyzes workout videos and live webcam feeds to classify exercises and count repetitions using pose estimation.

Built for internal demonstration and board review. Under active development — feature extraction is ongoing and the model will be retrained as more exercise classes are processed.

## How It Works

The core idea: **detect what exercise is being performed, then apply exercise-specific rep counting logic**. Different exercises have different movement patterns — a squat tracks knee angle, a bicep curl tracks elbow angle, a deadlift tracks hip angle. The ML classifier identifies the exercise first, then the rep counter uses the correct angle and thresholds for that specific exercise to count reps accurately.

## Pipeline

```
Video/Webcam → YOLOv8 Person Detection → MediaPipe Pose (12 landmarks)
    → 9 Joint Angles → ML Classifier (ExtraTrees) → Rep Counter (state machine)
    → Annotated Output (skeleton + HUD overlay)
```

**Pose pipeline**: YOLOv8 nano detects the person (scored by bbox size + center proximity), MediaPipe PoseLandmarker extracts 12 body landmarks, then 9 joint angles are computed (knees, hips, elbows, shoulders, torso inclination).

**Exercise classification**: A sliding window of the last 30 frames computes mean/std/min/max of all 9 angles (37 features total), fed to a trained ExtraTrees classifier. Currently ~85% cross-validation accuracy on 15 exercise classes (trained on partial data — will improve as remaining classes are extracted).

**Rep counting**: Two-state machine (EXTENDED ↔ CONTRACTED) tracks the primary angle per exercise with configurable thresholds and minimum duration debounce.

## Project Structure

```
gym-monitoring/
├── main.py                           # Entry point (not yet integrated)
├── config.yaml                       # All configurable parameters
├── requirements.txt
│
├── scripts/
│   ├── extract_pose_features.py      # Extract angle features from training videos
│   ├── experiment.py                 # Train & compare 6 classifiers (5-fold CV)
│   ├── evaluate.py                   # Evaluate model on unseen test videos
│   └── predict.py                    # Predict from video file or --live webcam
│
├── src/
│   ├── pose/
│   │   ├── person_detector.py        # YOLOv8 person detection (MPS GPU support)
│   │   ├── estimator.py              # MediaPipe PoseLandmarker wrapper
│   │   └── angles.py                 # 9 joint angle computation
│   ├── exercise/
│   │   ├── definitions.py            # Exercise enum (8 exercises + UNKNOWN)
│   │   └── classifier.py             # Sliding window ML classifier
│   ├── rep_counter/
│   │   ├── counter.py                # State machine rep counter
│   │   └── rep_rules.py              # Per-exercise angle thresholds
│   ├── output/
│   │   └── video_annotator.py        # Skeleton + HUD overlay drawing
│   └── utils/
│       ├── geometry.py               # Vector math, angle calculation
│       ├── video_io.py               # OpenCV video read/write helpers
│       └── config.py                 # YAML → typed dataclass config
│
├── notebooks/
│   └── analysis.ipynb                # Feature exploration & visualization
│
├── data/
│   ├── videos/                       # Training + test videos (not in git)
│   ├── models/                       # Model files (not in git)
│   │   ├── yolov8n.pt               # YOLOv8 nano person detector
│   │   ├── pose_landmarker_heavy.task # MediaPipe pose model
│   │   └── exercise_classifier.pkl   # Trained sklearn pipeline
│   ├── features.csv                  # Extracted features (not in git)
│   └── enrollment/                   # Face enrollment photos (future)
│
└── outputs/                          # Generated results
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

- ~1,600 video clips, 22 exercise categories
- Pre-processed clips (10-13 seconds), MP4 format
- Two subsets: `data_btc_10s` (rep-aligned) and `data_crawl_10s` (sliding window)

## Usage

### Train a model

```bash
# 1. Extract pose features from all training videos
python scripts/extract_pose_features.py --all --output data/features.csv

# 2. Train classifiers and save best model
python scripts/experiment.py --features data/features.csv

# 3. Evaluate on unseen test videos
python scripts/evaluate.py --test-dir data/videos/test
```

### Predict

```bash
# From a video file
python scripts/predict.py data/videos/test/deadlift/deadlift_1.mp4

# Live webcam mode (real-time classification + rep counting)
python scripts/predict.py --live
```

Live mode controls:
- `q` — quit
- `r` — reset rep count and classifier buffer

### Test a single video through the pose pipeline

```bash
python scripts/extract_pose_features.py --video data/videos/data_btc_10s/squat/some_video.mp4
```

## Supported Exercises

The goal is to first detect **which** exercise is happening, then apply the correct counting logic. Each exercise has its own tracked angle and thresholds:

| Exercise | Tracked Angle | Extended | Contracted | Min Rep Duration |
|----------|--------------|----------|------------|-----------------|
| Squat | avg knee | >160° | <100° | 800ms |
| Bicep Curl | avg elbow | >150° | <60° | 600ms |
| Push-up | avg elbow | >155° | <80° | 700ms |
| Deadlift | avg hip | >165° | <100° | 1000ms |
| Shoulder Press | avg elbow | >160° | <80° | 700ms |
| Lunge | min knee | >155° | <95° | 900ms |
| Pull-up | avg elbow | >160° | <70° | 800ms |
| Bench Press | avg elbow | >160° | <75° | 800ms |

The trained model currently recognizes 15 classes from the Kaggle dataset (partial extraction — more classes being added). Model labels are mapped to the above exercise categories for rep counting. Exercises without a mapping (e.g. plank, leg raises) are classified but don't have rep counting.

## Configuration

All thresholds, model paths, and parameters are in `config.yaml`. Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `exercise.window_size` | 30 | Sliding window frames for classification |
| `exercise.classification_interval_frames` | 15 | Classify every N frames in live mode |
| `exercise.min_classification_score` | 0.3 | Minimum confidence to accept prediction |
| `person_detection.confidence_threshold` | 0.5 | YOLO detection threshold |
| `rep_counter.min_rep_duration_ms` | 800 | Global minimum rep duration |

## Hardware

Tested on Apple M4 Pro (24GB). YOLOv8 uses MPS (Metal) GPU acceleration when available, falls back to CPU otherwise.
