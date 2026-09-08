import numpy as np
from collections import Counter
from typing import List, Dict, Any

class ConsistencyService:
    def analyze(
        self, raw_pred: Dict[str, Any], views: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculates real mathematical consistency metrics across multi-view predictions.
        Includes entropy, top1-top2 margin, view flipping ratios, and consensus voting maps.
        """
        if not views:
            return {
                "prediction_agreement": 1.0,
                "prediction_switching_rate": 0,
                "dominant_prediction": raw_pred["top1_label"],
                "dominant_class_index": raw_pred["top1_class_index"],
                "mean_confidence": raw_pred["top1_confidence"],
                "std_confidence": 0.0,
                "disagreement_with_raw": False,
                "transformation_sensitivity": 0.0,
                "raw_entropy": raw_pred.get("entropy", 0.0),
                "raw_margin": raw_pred.get("margin", 0.0),
                "view_flipping_ratio": 0.0,
                "total_views": 0,
            }

        class_indices = [v["class_index"] for v in views]
        labels = [v["prediction"] for v in views]
        confidences = [v["confidence"] for v in views]

        # Dominant prediction (mode)
        class_counts = Counter(class_indices)
        dominant_class_index, dominant_count = class_counts.most_common(1)[0]
        
        # Find label for dominant class
        dominant_label = labels[class_indices.index(dominant_class_index)]

        # Agreement ratio (dominant class count / total views)
        agreement_ratio = round(float(dominant_count) / float(len(views)), 4)

        # View flipping ratio (number of views predicting a class OTHER than raw top-1)
        raw_class_index = raw_pred["top1_class_index"]
        flipped_views_count = sum(1 for c in class_indices if c != raw_class_index)
        view_flipping_ratio = round(float(flipped_views_count) / float(len(views)), 4)

        # Switching rate across view sequence
        sequence = [raw_class_index] + class_indices
        switching_count = sum(1 for i in range(len(sequence) - 1) if sequence[i] != sequence[i+1])

        # Mean and standard deviation of confidences
        mean_conf = round(float(np.mean(confidences)), 4)
        std_conf = round(float(np.std(confidences)), 4)

        # Disagreement between raw prediction and consensus dominant class
        disagreement_with_raw = (raw_class_index != dominant_class_index)

        # Transformation sensitivity (average confidence drop/shift from raw)
        raw_conf = raw_pred["top1_confidence"]
        conf_diffs = [abs(raw_conf - c) for c in confidences]
        sensitivity = round(float(np.mean(conf_diffs)), 4)

        return {
            "prediction_agreement": agreement_ratio,
            "prediction_switching_rate": switching_count,
            "dominant_prediction": dominant_label,
            "dominant_class_index": dominant_class_index,
            "mean_confidence": mean_conf,
            "std_confidence": std_conf,
            "disagreement_with_raw": disagreement_with_raw,
            "transformation_sensitivity": sensitivity,
            "raw_entropy": raw_pred.get("entropy", 0.0),
            "raw_margin": raw_pred.get("margin", 0.0),
            "view_flipping_ratio": view_flipping_ratio,
            "total_views": len(views),
        }

consistency_service = ConsistencyService()
