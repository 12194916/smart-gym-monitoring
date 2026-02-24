# Gym Video Monitoring MVP — Project Plan

## Overview

A Python-based computer vision pipeline that processes pre-recorded workout videos and outputs structured analysis results. The system identifies known individuals, classifies exercises, and counts repetitions using pose estimation.

This is an applied engineering project for internal demonstration and board review — not a production deployment.

---

## Objectives

1. **Person Identification** — Recognize known individuals in video clips using facial embeddings (closed-set recognition)
2. **Exercise Classification** — Classify the exercise being performed from a small, fixed set of exercises (approximately 5–8, to be defined with client)
3. **Repetition Counting** — Count reps for at least one exercise (e.g., squats) using pose-based heuristics
4. **Structured Output** — Results including identified user, exercise label, rep count, per-rep timestamps, and confidence scores

---

## Scope

### In Scope

- Pre-recorded video only (no live streaming)
- Single primary subject per clip
- Small known user set (enrollment images provided by client)
- Limited exercise set (approximately 5–8 exercises, defined by client)
- Rep counting required for at least one exercise
- Python implementation
- Cloud or local GPU friendly

### Out of Scope

- Live/real-time streaming
- Multi-person tracking
- Mobile or on-device optimization
- UI/frontend polish
- Production deployment or scaling

---

## Technical Approach

### Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| Pose Estimation | MediaPipe Pose (Google) | Fast, accurate, easy setup, CPU and GPU support |
| Face Recognition | InsightFace (ArcFace) | State-of-the-art accuracy, simple Python API |
| Video Processing | OpenCV | Industry standard for video I/O and frame manipulation |
| Language | Python 3.10+ | Rich CV ecosystem, per project requirements |
| Output | JSON + annotated MP4 | Machine-readable results + visual demo |

### Architecture

The system is a **modular single-pass pipeline** with clear separation between components:

```
Input Video
    │
    ▼
┌─────────────────────────────────────────────────┐
│              Pipeline Orchestrator                │
│                                                   │
│   ┌──────────────┐                               │
│   │ Face ID       │  Identify the person          │
│   │ (InsightFace) │  in the first few seconds     │
│   └──────────────┘                               │
│           │                                       │
│   ┌──────────────┐                               │
│   │ Pose          │  Extract body landmarks       │
│   │ Estimation    │  every frame                  │
│   │ (MediaPipe)   │                               │
│   └──────────────┘                               │
│           │                                       │
│   ┌──────────────┐                               │
│   │ Exercise      │  Classify which exercise      │
│   │ Classifier    │  is being performed           │
│   └──────────────┘                               │
│           │                                       │
│   ┌──────────────┐                               │
│   │ Rep Counter   │  Count repetitions            │
│   │               │  with timestamps              │
│   └──────────────┘                               │
│           │                                       │
│   ┌──────────────┐                               │
│   │ Output        │  JSON results +               │
│   │               │  annotated video              │
│   └──────────────┘                               │
└─────────────────────────────────────────────────┘
```

### How It Works

**Face Identification** — Process reference photos of known users into facial embeddings (enrollment). During video analysis, detect the face in the first few seconds, match against the gallery, and lock the identity for the clip.

**Pose Estimation** — MediaPipe extracts 33 body landmarks per frame. Key joint angles (knee, hip, elbow, shoulder, torso) are computed from these landmarks.

**Exercise Classification** — Rule-based heuristic approach using joint angle patterns. Each exercise has a defined signature based on which joints move and how. No separate ML model training required.

**Repetition Counting** — State machine that tracks joint angle cycles (extended → contracted → extended = 1 rep). Records per-rep timestamps and confidence.

**Output** — Structured JSON file with all results + annotated video with pose skeleton overlay, user label, exercise label, and live rep counter.

---

## Deliverables

| # | Deliverable | Description |
|---|------------|-------------|
| 1 | Python pipeline code | Full video analysis pipeline with clear modular separation between face identification, exercise classification, and rep counting logic |
| 2 | Sample outputs | Results on provided video files demonstrating all capabilities |
| 3 | README | How to run the pipeline, architecture description, setup instructions |

---

## Timeline — 4 Weeks

### Week 1 — Foundation & Pose Pipeline
- Project setup, dependencies, configuration
- Pose estimation module (MediaPipe integration)
- Joint angle computation from body landmarks
- Video annotator with skeleton overlay for visual verification
- **Milestone:** Extract and visualize pose landmarks from video

### Week 2 — Exercise Classification & Rep Counting
- Rule-based exercise classifier using joint angle patterns
- Rep counter state machine with per-exercise thresholds
- Integration of classification + counting with annotated video
- **Milestone:** Classify exercise and count reps with visual overlay

### Week 3 — Face Recognition & Full Pipeline Integration
- Face enrollment tool (reference images → embedding gallery)
- Face recognition module (match against gallery per video)
- Pipeline orchestrator connecting all modules end-to-end
- JSON output with full structured results
- CLI entry point (enroll + analyze commands)
- **Milestone:** Complete pipeline — identify person, classify exercise, count reps, produce output

### Week 4 — Testing, Tuning & Delivery
- Unit tests for core logic (angles, classifier, rep counter)
- Integration testing with provided sample videos
- Threshold tuning for accuracy on real data
- Edge case handling (no face detected, poor pose frames)
- README and documentation
- Sample outputs on all provided videos
- **Milestone:** Final tested deliverable ready for demo

---

## What Is Needed from Client

To begin and progress through development, the following materials and information are required:

| # | Item | Details | When Needed |
|---|------|---------|-------------|
| 1 | **Exercise list** | The specific 5–8 exercises to support (e.g., squats, push-ups, deadlifts, curls, etc.) | Before Week 1 |
| 2 | **Enrollment photos** | 2–5 clear, front-facing photos per known user (good lighting, visible face) | Before Week 3 |
| 3 | **User list** | Names and IDs for all known users in the system | Before Week 3 |
| 4 | **Sample workout videos** | Pre-recorded MP4 files showing single subjects performing exercises from the defined set | Before Week 2 (ideally Week 1) |
| 5 | **Video recording guidelines** | Confirm camera angle, resolution, and lighting conditions used for recordings | Before Week 1 |
| 6 | **GPU environment** (optional) | Cloud GPU instance or confirmation of local GPU availability. CPU-only works but is slower | Before Week 1 (preferred) |

---

## What Is Provided by Developer

| # | Item | Details |
|---|------|---------|
| 1 | Full Python pipeline code | Modular, clean, well-structured codebase |
| 2 | Face enrollment tool | CLI command to build face gallery from reference photos |
| 3 | Video analysis tool | CLI command to process videos and produce structured results |
| 4 | JSON output per video | User ID, exercise label, rep count, per-rep timestamps, confidence scores |
| 5 | Annotated demo video | Output video with pose skeleton, labels, and rep counter overlay |
| 6 | Configuration file | YAML-based config for all tunable parameters and thresholds |
| 7 | Unit & integration tests | Automated tests for core logic |
| 8 | README | Setup instructions, architecture overview, usage guide |
| 9 | Sample outputs | Processed results on all provided video files |

---

## Sample Output (JSON)

```json
{
  "input_video": "workout_clip_01.mp4",
  "processed_at": "2026-03-10T14:30:00Z",
  "video_metadata": {
    "duration_seconds": 45.2,
    "fps": 30,
    "resolution": [1920, 1080]
  },
  "identification": {
    "user_id": "user_001",
    "user_name": "Alice",
    "confidence": 0.87
  },
  "exercise": {
    "label": "squat",
    "confidence": 0.92
  },
  "rep_count": {
    "total": 8,
    "reps": [
      {
        "rep_number": 1,
        "start_ms": 2100,
        "end_ms": 3400,
        "duration_ms": 1300
      },
      {
        "rep_number": 2,
        "start_ms": 3400,
        "end_ms": 4600,
        "duration_ms": 1200
      }
    ]
  }
}
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Face not visible in video (back to camera, mask, etc.) | Attempt detection over first 60 frames; output "unidentified" if no match |
| Poor pose detection (occlusion, unusual camera angle) | Skip frames with no pose; smoothing buffers to handle gaps |
| Exercise misclassification between similar exercises | Multi-feature angle signatures with torso orientation and symmetry checks |
| Rep double-counting from noisy pose data | Minimum rep duration debounce and angle smoothing |
| Delayed or missing sample data from client | Development can proceed with publicly available workout videos for initial testing; client data used for final tuning |

---

## Deadline

**Start date:** February 23, 2026
**Target completion:** March 23, 2026 (4 weeks at most)

