"""Experiment with classifiers on raw landmark features.

Loads landmarks.csv (132 features per frame), balances classes,
trains several models, compares performance, and saves the best model.

Follows the notebook approach: RandomForest with 300 trees as baseline,
max 2000 samples per class, 80/20 stratified split.

Usage:
    python scripts/experiment.py
    python scripts/experiment.py --features data/landmarks.csv
    python scripts/experiment.py --save-model data/models/exercise_classifier.pkl
"""

import argparse
import os
import sys
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Models to compare — notebook baseline is RandomForest_300
MODELS = {
    "RandomForest_300": RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "ExtraTrees_300": ExtraTreesClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "RandomForest_500": RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

# Max samples per class for balancing (matching notebook)
MAX_SAMPLES_PER_CLASS = 2000


def load_data(features_path: str, max_samples: int = MAX_SAMPLES_PER_CLASS):
    """Load landmarks CSV, balance classes, return X, y arrays."""
    df = pd.read_csv(features_path)
    feature_cols = [c for c in df.columns if c != "label"]

    print(f"Loaded {len(df)} samples, {df['label'].nunique()} classes")
    print(f"Features: {len(feature_cols)} columns")

    # Balance classes (cap at max_samples per class)
    balanced_dfs = []
    for label in df["label"].unique():
        label_df = df[df["label"] == label]
        if len(label_df) > max_samples:
            label_df = label_df.sample(n=max_samples, random_state=42)
        balanced_dfs.append(label_df)
    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    print(f"After balancing (max {max_samples}/class): {len(df_balanced)} samples")

    X = df_balanced[feature_cols].values
    y = df_balanced["label"].values

    return X, y, feature_cols, df_balanced


def run_experiments(X_train, y_train):
    """Run all models with 5-fold stratified cross-validation on training set."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "f1_weighted", "precision_weighted", "recall_weighted"]

    results = []

    print(f"\n{'=' * 75}")
    print(f"{'Model':<22} {'CV Accuracy':>12} {'F1':>10} {'Precision':>10} {'Recall':>10} {'Time':>8}")
    print(f"{'=' * 75}")

    for name, model in MODELS.items():
        start = time.time()
        scores = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring,
            return_train_score=False, n_jobs=-1,
        )
        elapsed = time.time() - start

        result = {
            "model": name,
            "accuracy": float(scores["test_accuracy"].mean()),
            "accuracy_std": float(scores["test_accuracy"].std()),
            "f1": float(scores["test_f1_weighted"].mean()),
            "f1_std": float(scores["test_f1_weighted"].std()),
            "precision": float(scores["test_precision_weighted"].mean()),
            "recall": float(scores["test_recall_weighted"].mean()),
            "time_seconds": elapsed,
        }
        results.append(result)

        print(f"{name:<22} {result['accuracy']:>10.1%} +/-{result['accuracy_std']:.1%}"
              f" {result['f1']:>9.1%} {result['precision']:>9.1%}"
              f" {result['recall']:>9.1%} {elapsed:>7.1f}s")

    print(f"{'=' * 75}")
    return results


def evaluate_on_test(model, X_test, y_test, model_name: str):
    """Evaluate trained model on held-out test set."""
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()

    print(f"\n{'=' * 75}")
    print(f"TEST SET EVALUATION — {model_name}")
    print(f"{'=' * 75}")
    print(f"Test accuracy: {accuracy:.1%}")
    print(classification_report(y_test, y_pred, digits=2, zero_division=0))

    # Most confused pairs
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("Most confused pairs:")
    confusions = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], labels[i], labels[j]))
    confusions.sort(reverse=True)
    if not confusions:
        print("  None — perfect classification!")
    for count, true_label, pred_label in confusions[:15]:
        print(f"  {true_label} -> {pred_label}: {count} times")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Experiment with exercise classifiers")
    parser.add_argument("--features", default="data/landmarks.csv",
                        help="Path to landmarks CSV")
    parser.add_argument("--save-model", default="data/models/exercise_classifier.pkl",
                        help="Path to save best model")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_CLASS,
                        help="Max samples per class for balancing")
    args = parser.parse_args()

    if not os.path.exists(args.features):
        print(f"Error: {args.features} not found")
        print("Run: python scripts/extract_pose_features.py --all")
        sys.exit(1)

    # Load and balance data
    X, y, feature_cols, df = load_data(args.features, args.max_samples)

    print(f"\nSamples per class:")
    for cls in sorted(df["label"].unique()):
        count = (df["label"] == cls).sum()
        print(f"  {cls:<25} {count}")

    # 80/20 stratified split (matching notebook)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Run cross-validation experiments on training set
    results = run_experiments(X_train, y_train)

    # Find best model by CV accuracy
    results_sorted = sorted(results, key=lambda r: r["accuracy"], reverse=True)
    best = results_sorted[0]
    best_name = best["model"]
    print(f"\nBest model: {best_name} (CV accuracy: {best['accuracy']:.1%})")

    # Train best model on full training set
    print(f"Training {best_name} on {len(X_train)} training samples...")
    best_model = MODELS[best_name]
    best_model.fit(X_train, y_train)

    # Evaluate on held-out test set
    test_accuracy = evaluate_on_test(best_model, X_test, y_test, best_name)

    # Retrain on ALL data for final model
    print(f"\nRetraining {best_name} on ALL {len(X)} samples for final model...")
    final_model = MODELS[best_name].__class__(**MODELS[best_name].get_params())
    final_model.fit(X, y)

    # Save model (joblib, matching notebook format — just the model object)
    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
    joblib.dump(final_model, args.save_model)
    print(f"Saved model to: {args.save_model}")

    # Save experiment results
    results_path = os.path.join(os.path.dirname(args.save_model), "experiment_results.json")
    experiment_info = {
        "best_model": best_name,
        "cv_accuracy": best["accuracy"],
        "test_accuracy": test_accuracy,
        "n_classes": len(set(y)),
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_total": len(X),
        "classes": sorted(set(y)),
        "results": results_sorted,
    }
    with open(results_path, "w") as f:
        json.dump(experiment_info, f, indent=2)
    print(f"Saved results to: {results_path}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
