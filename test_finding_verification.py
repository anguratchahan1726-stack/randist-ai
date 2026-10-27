"""Unit tests for finding verification and uncertainty quantification."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from finding_verification import (
    FindingVerificationEngine,
    STATUS_BORDERLINE_UNCERTAIN,
    STATUS_VERIFIED_ABSENT,
    STATUS_VERIFIED_PRESENT,
)


def test_verification_clear_positive():
    engine = FindingVerificationEngine()
    # Mock clear positive for Cardiomegaly
    mock_scores = {
        "Atelectasis": 0.10,
        "Cardiomegaly": 0.88,
        "Consolidation": 0.05,
        "Edema": 0.12,
        "Infiltration": 0.08,
        "Pleural Effusion": 0.15,
        "Pneumothorax": 0.02,
        "No Finding": 0.10,
    }

    result = engine.verify_findings(mock_scores)
    assert "Cardiomegaly" in result["verified_present"]
    assert "Cardiomegaly" not in result["borderline_findings"]
    assert result["finding_details"]["Cardiomegaly"]["status"] == STATUS_VERIFIED_PRESENT


def test_verification_borderline_uncertainty():
    engine = FindingVerificationEngine(uncertainty_margin=0.10)
    # Threshold for Pleural Effusion is 0.45; score 0.48 is within 0.10 margin
    mock_scores = {
        "Atelectasis": 0.10,
        "Cardiomegaly": 0.10,
        "Consolidation": 0.10,
        "Edema": 0.10,
        "Infiltration": 0.10,
        "Pleural Effusion": 0.48,
        "Pneumothorax": 0.05,
        "No Finding": 0.20,
    }

    result = engine.verify_findings(mock_scores)
    assert "Pleural Effusion" in result["borderline_findings"]
    detail = result["finding_details"]["Pleural Effusion"]
    assert detail["is_borderline"] is True
    assert detail["uncertainty_index"] > 0.0


def test_clinical_conflict_detection():
    engine = FindingVerificationEngine()
    # Inconsistent case: High score for both 'No Finding' and 'Pneumothorax'
    mock_scores = {
        "Atelectasis": 0.05,
        "Cardiomegaly": 0.05,
        "Consolidation": 0.05,
        "Edema": 0.05,
        "Infiltration": 0.05,
        "Pleural Effusion": 0.05,
        "Pneumothorax": 0.85,
        "No Finding": 0.80,
    }

    result = engine.verify_findings(mock_scores)
    assert len(result["detected_conflicts"]) > 0
    assert any("Discordance" in c for c in result["detected_conflicts"])
