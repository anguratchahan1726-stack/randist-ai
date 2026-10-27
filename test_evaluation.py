"""Unit tests for Scikit-Learn multi-label evaluation engine."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config.settings import DATA_DIR, SUPPORTED_FINDINGS
from evaluation import compute_multilabel_metrics, evaluate_test_dataset
from model_inference import ChestXRayClassifier


def test_compute_multilabel_metrics():
    # 4 samples, 8 classes
    num_classes = len(SUPPORTED_FINDINGS)
    y_true = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ])
    y_prob = np.array([
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.85, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.8, 0.75, 0.1, 0.1, 0.1, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.95],
    ])

    metrics = compute_multilabel_metrics(y_true, y_prob)

    assert "subset_accuracy" in metrics
    assert "hamming_loss" in metrics
    assert "macro_f1" in metrics
    assert "per_class" in metrics

    # Since y_prob matches y_true perfectly at threshold 0.5, subset accuracy is 1.0
    assert metrics["subset_accuracy"] == 1.0
    assert metrics["hamming_loss"] == 0.0


def test_evaluate_dataset_with_manifest():
    classifier = ChestXRayClassifier.get_instance()
    manifest_path = DATA_DIR / "ground_truth.json"

    if manifest_path.exists():
        eval_result = evaluate_test_dataset(classifier, manifest_path=manifest_path)
        assert eval_result["evaluated"] is True
        assert "subset_accuracy" in eval_result
        assert "macro_f1" in eval_result
        assert "per_class" in eval_result
        assert len(eval_result["processed_files"]) > 0


def test_evaluate_dataset_missing_manifest():
    classifier = ChestXRayClassifier.get_instance()
    missing_path = DATA_DIR / "non_existent_manifest.json"

    eval_result = evaluate_test_dataset(classifier, manifest_path=missing_path)
    assert eval_result["evaluated"] is False
    assert "not found" in eval_result["message"].lower() or "no ground-truth" in eval_result["message"].lower()
