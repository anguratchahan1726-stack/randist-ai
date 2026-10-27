"""Finding verification and uncertainty quantification engine for RadInsight AI.

Performs threshold comparison, confidence calibration, borderline uncertainty
quantification, and clinical inconsistency checking before report generation.
"""

from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    ANATOMICAL_LOCATIONS,
    DEFAULT_DECISION_THRESHOLDS,
    FINDING_DESCRIPTIONS,
    SUPPORTED_FINDINGS,
    UNCERTAINTY_MARGIN,
)

                  
STATUS_VERIFIED_PRESENT = "VERIFIED_PRESENT"
STATUS_BORDERLINE_UNCERTAIN = "BORDERLINE_UNCERTAIN"
STATUS_VERIFIED_ABSENT = "VERIFIED_ABSENT"

class FindingVerificationEngine:
    """Verifies predicted finding scores against calibrated decision thresholds

    and flags diagnostic uncertainty or conflicting assertions.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        uncertainty_margin: float = UNCERTAINTY_MARGIN,
    ):
        self.thresholds = dict(thresholds or DEFAULT_DECISION_THRESHOLDS)
        self.uncertainty_margin = uncertainty_margin

    def verify_findings(
        self,
        raw_scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Perform feature verification, uncertainty quantification, and conflict detection.

        Args:
            raw_scores: Dictionary mapping finding name to model prediction probability [0.0, 1.0].

        Returns:
            Structured verification report containing:
              - verified_present: List of findings meeting positive threshold
              - borderline_findings: List of findings falling in uncertainty band
              - verified_absent: List of findings below threshold
              - finding_details: Dict of per-finding diagnostic metadata
              - detected_conflicts: List of clinical inconsistency warnings
              - average_uncertainty: Aggregate uncertainty index
              - summary_status: High-level qualitative assessment
        """
        verified_present: List[str] = []
        borderline_findings: List[str] = []
        verified_absent: List[str] = []
        finding_details: Dict[str, Dict[str, Any]] = {}

        total_uncertainty_score = 0.0

        for finding in SUPPORTED_FINDINGS:
            score = raw_scores.get(finding, 0.0)
            threshold = self.thresholds.get(finding, 0.50)

                                           
            distance_to_boundary = abs(score - threshold)

                                                                                               
                                                             
            if distance_to_boundary <= self.uncertainty_margin:
                uncertainty_level = 1.0 - (distance_to_boundary / self.uncertainty_margin)
            else:
                uncertainty_level = 0.0

            total_uncertainty_score += uncertainty_level

                                       
            is_borderline = distance_to_boundary <= self.uncertainty_margin

            if is_borderline:
                status = STATUS_BORDERLINE_UNCERTAIN
                borderline_findings.append(finding)
            elif score >= threshold:
                status = STATUS_VERIFIED_PRESENT
                verified_present.append(finding)
            else:
                status = STATUS_VERIFIED_ABSENT
                verified_absent.append(finding)

            finding_details[finding] = {
                "raw_score": round(score, 4),
                "threshold": round(threshold, 4),
                "status": status,
                "is_borderline": is_borderline,
                "uncertainty_index": round(uncertainty_level, 4),
                "distance_to_threshold": round(score - threshold, 4),
                "anatomical_location": ANATOMICAL_LOCATIONS.get(finding, "chest cavity"),
                "clinical_definition": FINDING_DESCRIPTIONS.get(finding, ""),
            }

                                     
                                                         
        detected_conflicts: List[str] = []
        pathologies_present = [f for f in verified_present if f != "No Finding"]
        no_finding_detail = finding_details.get("No Finding", {})

        if "No Finding" in verified_present and len(pathologies_present) > 0:
            conflict_msg = (
                f"Discordance Detected: Model classified 'No Finding' as positive ({no_finding_detail.get('raw_score')}) "
                f"while concurrently verifying pathological findings: {', '.join(pathologies_present)}. "
                "Feature verification requires radiologist review to resolve mutual exclusivity."
            )
            detected_conflicts.append(conflict_msg)

                                             
        avg_uncertainty = total_uncertainty_score / max(len(SUPPORTED_FINDINGS), 1)
        if avg_uncertainty > 0.45:
            detected_conflicts.append(
                "High Global Ambiguity: Multiple findings fall within the decision boundary uncertainty band. "
                "Image may suffer from suboptimal exposure, motion artifact, or low contrast."
            )

                                                                                               
        reconciled_present = list(verified_present)
        if len(pathologies_present) > 0 and "No Finding" in reconciled_present:
            reconciled_present.remove("No Finding")

                                                                                                      
        if len(pathologies_present) == 0 and len(borderline_findings) == 0:
            if "No Finding" not in reconciled_present:
                reconciled_present.append("No Finding")

        return {
            "verified_present": reconciled_present,
            "borderline_findings": borderline_findings,
            "verified_absent": verified_absent,
            "finding_details": finding_details,
            "detected_conflicts": detected_conflicts,
            "average_uncertainty": round(avg_uncertainty, 4),
            "total_evaluated_findings": len(SUPPORTED_FINDINGS),
            "traceability_guarantee": (
                "Verified findings are linked strictly to calibrated classifier scores. "
                "Verification reduces thresholding hallucination but does NOT replace physician verification."
            ),
        }
