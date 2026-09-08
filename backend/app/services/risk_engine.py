from typing import Dict, Any, List
from app.config import config

class RiskEngine:
    def evaluate(
        self, consistency: Dict[str, Any], raw_pred: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates prediction consistency metrics, entropy, confidence saturation,
        and margin signals to calculate a unified risk score (0-100), threat level (LOW, MEDIUM, HIGH),
        and dynamically generated explanatory reasons.
        Implements compounding evidence bonus when 3+ independent alert signals co-occur.
        """
        agreement = consistency["prediction_agreement"]
        switching_rate = consistency["prediction_switching_rate"]
        std_conf = consistency["std_confidence"]
        disagreement = consistency["disagreement_with_raw"]
        sensitivity = consistency["transformation_sensitivity"]
        flipping_ratio = consistency.get("view_flipping_ratio", 0.0)

        raw_conf = raw_pred.get("top1_confidence", 0.0)
        raw_margin = raw_pred.get("margin", 0.0)
        raw_entropy = raw_pred.get("entropy", 0.0)

        reasons: List[str] = []
        score: float = 0.0
        active_alerts_count: int = 0

        # Component 1: Agreement penalty (up to 30 pts)
        if agreement < 0.95:
            active_alerts_count += 1
            agreement_penalty = (1.0 - agreement) * 30.0
            score += agreement_penalty
            reasons.append(
                f"Low cross-view prediction agreement ({int(agreement * 100)}% agreement across {consistency['total_views']} transformed views)"
            )

        # Component 2: Disagreement between raw prediction & consensus class penalty
        if disagreement:
            active_alerts_count += 1
            if raw_conf > 0.60 or raw_entropy < 1.0:
                score += 20.0
                reasons.append(
                    f"Raw prediction '{raw_pred['top1_label']}' disagrees with multi-view consensus class '{consistency['dominant_prediction']}'"
                )
            else:
                score += 8.0
                reasons.append(
                    f"Mild fine-grained class variation between raw '{raw_pred['top1_label']}' and view consensus '{consistency['dominant_prediction']}'"
                )

        # Component 3: Class switching penalty (up to 15 pts)
        if switching_rate > 2:
            active_alerts_count += 1
            switch_penalty = min(15.0, (switching_rate - 2) * 3.0)
            score += switch_penalty
            reasons.append(
                f"Prediction switched classes {switching_rate} times across perturbation sequence"
            )

        # Component 4: Confidence instability / std dev penalty (up to 15 pts)
        if std_conf > 0.10:
            active_alerts_count += 1
            conf_penalty = min(15.0, ((std_conf - 0.10) / 0.25) * 15.0)
            score += conf_penalty
            reasons.append(
                f"High confidence variance across transformed views (std-dev: {std_conf:.2f})"
            )

        # Component 5: Transformation sensitivity (up to 10 pts)
        if sensitivity > 0.18:
            active_alerts_count += 1
            sens_penalty = min(10.0, ((sensitivity - 0.18) / 0.3) * 10.0)
            score += sens_penalty
            reasons.append(
                f"High prediction sensitivity to mild input perturbations ({sensitivity:.2f} mean conf shift)"
            )

        # Component 6: Saturated Raw Confidence & High Margin Fingerprint Penalty (up to 40 pts)
        is_overconfident = (raw_conf >= config.SATURATED_CONFIDENCE_THRESHOLD) or (raw_margin >= config.HIGH_MARGIN_THRESHOLD)
        has_disagreement = (flipping_ratio > 0.0) or (agreement < 1.0) or (std_conf > 0.05) or disagreement

        if is_overconfident and has_disagreement:
            active_alerts_count += 1
            overconf_penalty = config.SATURATED_OVERCONFIDENCE_PENALTY_WEIGHT * max(0.7, flipping_ratio + (1.0 - agreement))
            overconf_penalty = min(40.0, overconf_penalty)
            score += overconf_penalty
            reasons.append(
                f"Adversarial fingerprint detected: Saturated raw confidence ({raw_conf*100:.1f}%, margin {raw_margin:.2f}) combined with cross-view view flipping ({int(flipping_ratio*100)}% flipped views)"
            )

        # Component 7: Softmax Entropy Anomaly Signal (up to 15 pts)
        if (raw_entropy <= config.LOW_ENTROPY_THRESHOLD) and has_disagreement:
            active_alerts_count += 1
            entropy_penalty = config.LOW_ENTROPY_PENALTY_WEIGHT
            score += entropy_penalty
            reasons.append(
                f"Degenerate softmax distribution detected (near-zero entropy {raw_entropy:.4f} with perturbation instability)"
            )

        # COMPOUNDING EVIDENCE BONUS: If 3+ distinct signal categories fire simultaneously
        if active_alerts_count >= config.COMPOUNDING_SIGNALS_THRESHOLD:
            compounding_bonus = config.COMPOUNDING_SIGNALS_BONUS
            score += compounding_bonus
            reasons.append(
                f"Compounding evidence detected: {active_alerts_count} independent adversarial risk signals co-occurring simultaneously (+{compounding_bonus:.1f} compounding bonus)"
            )

        # Final score bounding
        score = round(min(100.0, max(0.0, score)), 1)

        if score <= config.RISK_THRESHOLD_LOW:
            level = "LOW"
            if not reasons:
                reasons.append("Stable model behavior verified across all transformation views")
        elif score <= config.RISK_THRESHOLD_MEDIUM:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return {
            "risk_score": score,
            "risk_level": level,
            "reasons": reasons,
            "active_alerts_count": active_alerts_count,
        }

risk_engine = RiskEngine()
