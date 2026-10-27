"""Unit tests for grounded radiology report generation."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from finding_verification import FindingVerificationEngine
from report_generation import RadiologyReportGenerator


def test_grounded_report_normal_case():
    verifier = FindingVerificationEngine()
    generator = RadiologyReportGenerator()

    # Normal case
    scores = {f: 0.05 for f in verifier.thresholds}
    scores["No Finding"] = 0.90
    verification_result = verifier.verify_findings(scores)

    report = generator.generate_report(verification_result)

    assert "findings" in report
    assert "impression" in report
    assert "generation_mode" in report
    assert "traceability_map" in report

    # Check normal impression
    assert "normal" in report["impression"].lower() or "no acute" in report["impression"].lower()
    # Check that unsupported diseases are NOT in impression
    assert "pneumothorax" not in report["impression"].lower()
    assert "cardiomegaly" not in report["impression"].lower()


def test_grounded_report_pathology_grounding():
    verifier = FindingVerificationEngine()
    generator = RadiologyReportGenerator()

    # Case with Cardiomegaly and Pleural Effusion
    scores = {f: 0.05 for f in verifier.thresholds}
    scores["Cardiomegaly"] = 0.82
    scores["Pleural Effusion"] = 0.77
    verification_result = verifier.verify_findings(scores)

    report = generator.generate_report(verification_result)

    impression = report["impression"]
    assert "CARDIOMEGALY" in impression
    assert "PLEURAL EFFUSION" in impression

    # Traceability map should track both findings
    assert "Cardiomegaly" in report["traceability_map"]
    assert "Pleural Effusion" in report["traceability_map"]
