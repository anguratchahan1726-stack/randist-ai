"""Unit tests for MobileNetV2 architecture and model inference."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from config.settings import SUPPORTED_FINDINGS
from model_inference import (
    ChestXRayClassifier,
    STATUS_FINE_TUNED,
    STATUS_IMAGENET_BACKBONE,
    build_mobilenetv2_classifier,
)


def test_build_mobilenetv2_classifier():
    num_classes = len(SUPPORTED_FINDINGS)
    model = build_mobilenetv2_classifier(
        input_shape=(224, 224, 3),
        num_classes=num_classes,
        use_pretrained_backbone=False,
    )

    assert model is not None
    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, num_classes)

    # Test forward pass with random tensor
    dummy_input = np.random.uniform(-1.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
    preds = model.predict(dummy_input, verbose=0)
    assert preds.shape == (1, num_classes)
    # Output of sigmoid must be strictly in [0, 1]
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_classifier_predict_wrapper():
    classifier = ChestXRayClassifier.get_instance()
    assert classifier.model is not None

    dummy_input = np.random.uniform(-1.0, 1.0, (1, 224, 224, 3)).astype(np.float32)
    result = classifier.predict(dummy_input)

    assert "raw_scores" in result
    assert "inference_time_ms" in result
    assert "model_status" in result
    assert result["is_clinically_validated"] is False

    raw_scores = result["raw_scores"]
    assert len(raw_scores) == len(SUPPORTED_FINDINGS)
    for finding in SUPPORTED_FINDINGS:
        assert finding in raw_scores
        score = raw_scores[finding]
        assert 0.0 <= score <= 1.0
