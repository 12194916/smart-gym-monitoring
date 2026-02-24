"""Experiment with multiple classifiers on extracted pose features.

Loads features.csv, trains several models with cross-validation,
compares performance, and saves the best model.

Usage:
    python scripts/experiment.py
    python scripts/experiment.py --features data/features_partial.csv
"""

import argparse
import os
import sys
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Suppress noisy warnings from LogisticRegression on unscaled folds
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Models to experiment with
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# All models we want to compare
MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42,
    ),
    "SVM_RBF": SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42,
    ),
    "KNN_7": KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        n_jobs=-1,
    ),
    "LogisticRegression": LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    ),
}

# Feature columns (exclude metadata)
METADATA_COLS = ["video_path", "label", "source"]


def load_data(features_path: str):
    """Load features CSV and return X, y, feature names, label encoder."""
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples, {df['label'].nunique()} classes")
    print(f"Classes: {sorted(df['label'].unique().tolist())}\n")

    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    X = df[feature_cols].values
    y_raw = df["label"].values

    # Encode labels to integers
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    return X, y, feature_cols, le, df


def run_experiments(X, y, le):
    """Run all models with stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

    results = []

    print("=" * 70)
    print(f"{'Model':<22} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Time':>8}")
    print("=" * 70)

    for name, model in MODELS.items():
        # Wrap each model in a pipeline with StandardScaler
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        start = time.time()
        scores = cross_validate(
            pipe, X, y, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1,
        )
        elapsed = time.time() - start

        result = {
            "model": name,
            "accuracy": scores["test_accuracy"].mean(),
            "accuracy_std": scores["test_accuracy"].std(),
            "f1": scores["test_f1_weighted"].mean(),
            "f1_std": scores["test_f1_weighted"].std(),
            "precision": scores["test_precision_weighted"].mean(),
            "recall": scores["test_recall_weighted"].mean(),
            "time_seconds": elapsed,
        }
        results.append(result)

        print(f"{name:<22} {result['accuracy']:>9.1%} {result['f1']:>9.1%} "
              f"{result['precision']:>9.1%} {result['recall']:>9.1%} {elapsed:>7.1f}s")

    print("=" * 70)
    return results


def train_best_model(X, y, le, results, output_dir: str):
    """Train the best model on all data and save it."""
    # Sort by accuracy
    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    best = results_sorted[0]
    best_name = best["model"]

    print(f"\nBest model: {best_name} (accuracy: {best['accuracy']:.1%} +/- {best['accuracy_std']:.1%})")

    # Train on full training data
    print(f"Training {best_name} on all {len(X)} samples...")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MODELS[best_name]),
    ])
    pipe.fit(X, y)

    # Save model + label encoder + feature column names
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "exercise_classifier.pkl")

    # Build feature column names in the same order as the CSV
    feature_col_names = []
    angle_names = [
        "left_knee", "right_knee", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "torso_inclination",
    ]
    for name in angle_names:
        feature_col_names.extend([
            f"{name}_mean", f"{name}_std", f"{name}_min", f"{name}_max",
        ])
    feature_col_names.append("num_frames")

    with open(model_path, "wb") as f:
        pickle.dump({
            "pipeline": pipe,
            "label_encoder": le,
            "feature_columns": feature_col_names,
        }, f)
    print(f"Saved model to: {model_path}")

    # Save experiment results
    results_path = os.path.join(output_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(results_sorted, f, indent=2)
    print(f"Saved results to: {results_path}")

    return pipe, best_name


def print_detailed_report(X, y, le, best_name):
    """Print detailed classification report using cross-validation predictions."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MODELS[best_name]),
    ])

    # Get out-of-fold predictions (each sample predicted when it was in the test fold)
    y_pred = cross_val_predict(pipe, X, y, cv=cv, n_jobs=-1)

    print(f"\n{'=' * 70}")
    print(f"DETAILED REPORT — {best_name} (cross-validation predictions)")
    print(f"{'=' * 70}")
    print(classification_report(
        y, y_pred,
        target_names=le.classes_,
        digits=2,
        zero_division=0,
    ))

    # Confusion matrix — show most confused pairs
    cm = confusion_matrix(y, y_pred)
    print("Most confused pairs:")
    n_classes = len(le.classes_)
    confusions = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], le.classes_[i], le.classes_[j]))
    confusions.sort(reverse=True)
    if not confusions:
        print("  None — perfect classification!")
    for count, true_label, pred_label in confusions[:10]:
        print(f"  {true_label} -> {pred_label}: {count} times")


def main():
    parser = argparse.ArgumentParser(description="Experiment with exercise classifiers")
    parser.add_argument("--features", default="data/features.csv",
                        help="Path to features CSV")
    parser.add_argument("--output-dir", default="data/models",
                        help="Directory to save the best model")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"Error: {args.features} not found")
        print("Run extract_pose_features.py --all first")
        sys.exit(1)

    # Load data
    X, y, feature_cols, le, df = load_data(args.features)

    print(f"Features: {len(feature_cols)} columns")
    print(f"Samples per class:")
    for cls in sorted(df["label"].unique()):
        count = (df["label"] == cls).sum()
        print(f"  {cls:<25} {count}")
    print()

    # Run experiments
    results = run_experiments(X, y, le)

    # Train and save best model
    pipe, best_name = train_best_model(X, y, le, results, args.output_dir)

    # Detailed report (using cross-validation, not training data)
    print_detailed_report(X, y, le, best_name)

    print(f"\nDone! Next step: evaluate on test videos with scripts/evaluate.py")


if __name__ == "__main__":
    main()
