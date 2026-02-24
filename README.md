# Smart Gym Monitoring

A computer vision pipeline that analyzes pre-recorded workout videos to identify people, classify exercises, and count repetitions using pose estimation.

Built for internal demonstration and board review.

## Goal

Process workout video files and output structured analysis results:
- Identify known individuals using facial recognition
- Classify the exercise being performed (from a fixed set of ~5-8 exercises)
- Count repetitions with per-rep timestamps
- Output results as JSON with confidence scores

## Project Structure

```
smart-gym-monitoring/
├── main.py                      # Entry point — run the pipeline
├── config.yaml                  # Configuration and thresholds
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── pipeline.py              # Orchestrator
│   ├── face_id/                 # Face enrollment and recognition
│   ├── pose/                    # Pose estimation and joint angles
│   ├── exercise/                # Exercise classification
│   ├── rep_counter/             # Repetition counting
│   ├── output/                  # JSON writer and video annotator
│   └── utils/                   # Geometry, video I/O, config loader
│
├── data/
│   ├── videos/                  # Input workout videos
│   ├── models/                  # Pre-trained model files
│   └── enrollment/              # Face enrollment photos
│
└── outputs/                     # Generated results
```

## Data

### Training Data — Kaggle Gym Workout/Exercises Video Dataset

Source: https://www.kaggle.com/datasets/philosopher0808/gym-workoutexercises-video

- ~1,600+ video clips of gym exercises
- 22 exercise categories organized by folder (folder name = label)
- Pre-processed clips segmented to 10-13 seconds
- MP4 format, sourced from YouTube
- Apache 2.0 license

Used for training the exercise classification model. Videos are processed through MediaPipe to extract pose landmarks and joint angles, which are then used to train a lightweight scikit-learn classifier.

### Model Files

- **MediaPipe Pose Landmarker** (`pose_landmarker_heavy.task`) — downloaded from the MediaPipe model catalog, placed in `data/models/`
- **Exercise Classifier** (`exercise_classifier.pkl`) — trained on the Kaggle dataset, saved in `data/models/`

### Client Data (later)

- Enrollment photos for face recognition (2-5 photos per person)
- Workout videos for final testing and threshold tuning

## Setup

```bash
conda create -n gym-monitoring python=3.10 -y
conda activate gym-monitoring
pip install -r requirements.txt
```

## Usage

```bash
python main.py <video_path>
```
