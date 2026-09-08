from typing import Dict, Any, List
from collections import Counter
import numpy as np
from PIL import Image

from app.config import config
from app.services.transformation_service import transformation_service

class MitigationService:
    def decide(
        self,
        raw_pred: Dict[str, Any],
        views: List[Dict[str, Any]],
        consistency: Dict[str, Any],
        risk: Dict[str, Any],
        image: Image.Image = None,
    ) -> Dict[str, Any]:
        """
        Determines mitigation action enforcing strict anti-adversarial recovery rules:
        1. LOW risk -> ACCEPTED (Raw prediction accepted unconditionally).
        2. If raw prediction is overconfident/adversarial (2+ fingerprint signals triggered)
           AND dominant view-class equals raw prediction:
           - Flagged raw class CANNOT self-validate by plurality.
           - Triggers real Expanded Robustness Recovery Pass at upper-end mild transform strength (0.85).
           - If expanded pass recovers a non-adversarial class -> CONSENSUS_RECOVERED.
           - If expanded pass fails to recover an alternative -> ABSTAIN (Honest Refusal).
        """
        risk_level = risk["risk_level"]
        risk_score = risk["risk_score"]
        raw_label = raw_pred["top1_label"]
        raw_conf = raw_pred.get("top1_confidence", 0.0)
        raw_entropy = raw_pred.get("entropy", 0.0)

        dominant_label = consistency["dominant_prediction"]
        agreement = consistency["prediction_agreement"]
        alerts = risk.get("active_alerts_count", 0)

        # Check if raw prediction triggered 2+ adversarial fingerprint signals
        is_flagged_adversarial = (
            (raw_conf >= config.SATURATED_CONFIDENCE_THRESHOLD) or
            (raw_entropy <= config.LOW_ENTROPY_THRESHOLD) or
            (alerts >= 2)
        )

        if risk_level == "LOW" and not is_flagged_adversarial:
            return {
                "final_decision": "ACCEPTED",
                "final_prediction": raw_label,
                "final_confidence": raw_conf,
                "mitigation_action": "Raw prediction verified stable across multi-view transformations. Accepted unconditionally.",
            }

        # Handle cases where dominant class equals suspicious raw class OR risk is MEDIUM/HIGH
        # We MUST NOT let flagged raw class win consensus by simple plurality
        needs_expanded_pass = (
            (dominant_label.lower() == raw_label.lower() and is_flagged_adversarial) or
            (risk_level in ["MEDIUM", "HIGH"] and dominant_label.lower() == raw_label.lower())
        )

        expanded_views_info = None
        if needs_expanded_pass and image is not None:
            # Trigger Stage 6 Real Expanded Robustness Pass using stronger transform strength
            expanded_views = transformation_service.run_multi_view_inference(
                image=image,
                strength=config.EXPANDED_PASS_STRENGTH,
                include_smoothing=True,
            )
            
            # Prepare JSON view trace for reporting
            expanded_views_info = [
                {k: val for k, val in v.items() if k != "_image_pil"}
                for v in expanded_views
            ]

            # Evaluate voting on expanded pass excluding flagged raw label
            non_raw_expanded = [v["prediction"] for v in expanded_views if v["prediction"].lower() != raw_label.lower()]
            
            if non_raw_expanded:
                exp_counts = Counter(non_raw_expanded)
                exp_alt_label, exp_alt_count = exp_counts.most_common(1)[0]
                exp_alt_ratio = exp_alt_count / len(expanded_views)

                if exp_alt_count >= 3 or exp_alt_ratio >= 0.25:
                    exp_confidences = [v["confidence"] for v in expanded_views if v["prediction"] == exp_alt_label]
                    weighted_conf = round(float(np.mean(exp_confidences)), 4) if exp_confidences else 0.0

                    return {
                        "final_decision": "CONSENSUS_RECOVERED",
                        "final_prediction": exp_alt_label,
                        "final_confidence": weighted_conf,
                        "mitigation_action": f"ARGUS detected adversarial instability ({risk_score}/100 risk) on raw '{raw_label}'. Ran expanded robustness pass (str={config.EXPANDED_PASS_STRENGTH}) and successfully recovered consensus class '{exp_alt_label}'.",
                        "expanded_pass_triggered": True,
                        "expanded_views": expanded_views_info,
                    }

            # If even the expanded pass cannot recover a clear alternative class other than flagged raw label:
            return {
                "final_decision": "ABSTAIN",
                "final_prediction": "ABSTAIN",
                "final_confidence": 0.0,
                "mitigation_action": f"ARGUS detected severe adversarial risk ({risk_score}/100 score). Refused raw prediction '{raw_label}'. Expanded recovery pass (str={config.EXPANDED_PASS_STRENGTH}) failed to establish a non-flagged alternative class. Abstaining from prediction.",
                "expanded_pass_triggered": True,
                "expanded_views": expanded_views_info,
            }

        # If dominant class differs from raw prediction and has strong agreement
        if dominant_label.lower() != raw_label.lower() and agreement >= 0.40:
            dominant_confidences = [v["confidence"] for v in views if v["prediction"] == dominant_label]
            weighted_conf = round(float(np.mean(dominant_confidences)), 4) if dominant_confidences else 0.0

            decision_type = "CONSENSUS_RECOVERED" if risk_level == "HIGH" else "ACCEPTED_CONSENSUS"
            return {
                "final_decision": decision_type,
                "final_prediction": dominant_label,
                "final_confidence": weighted_conf,
                "mitigation_action": f"Multi-view consensus recovered alternative class '{dominant_label}' (overriding raw prediction '{raw_label}').",
                "expanded_pass_triggered": False,
            }

        # Default fallback for unrecoverable instability
        return {
            "final_decision": "ABSTAIN",
            "final_prediction": "ABSTAIN",
            "final_confidence": 0.0,
            "mitigation_action": f"ARGUS detected high prediction variance across transformations ({risk_score}/100 risk). Refused prediction and abstained.",
            "expanded_pass_triggered": False,
        }

mitigation_service = MitigationService()
