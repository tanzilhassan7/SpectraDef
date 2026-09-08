from app.services.consistency_service import consistency_service

def test_consistency_service_math():
    raw_pred = {"top1_label": "tiger cat", "top1_class_index": 282, "top1_confidence": 0.9}
    views = [
        {"class_index": 282, "prediction": "tiger cat", "confidence": 0.88},
        {"class_index": 282, "prediction": "tiger cat", "confidence": 0.85},
        {"class_index": 9, "prediction": "ostrich", "confidence": 0.60},
        {"class_index": 282, "prediction": "tiger cat", "confidence": 0.80},
    ]

    res = consistency_service.analyze(raw_pred, views)

    assert res["prediction_agreement"] == 0.75  # 3 out of 4 agree
    assert res["dominant_prediction"] == "tiger cat"
    assert res["dominant_class_index"] == 282
    assert res["disagreement_with_raw"] is False
    assert res["prediction_switching_rate"] == 2  # 282 -> 282 -> 9 -> 282
