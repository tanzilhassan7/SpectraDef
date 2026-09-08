from app.services.risk_engine import risk_engine

def test_risk_engine_low_risk():
    raw_pred = {"top1_label": "tiger cat", "top1_class_index": 282, "top1_confidence": 0.95}
    consistency = {
        "prediction_agreement": 1.0,
        "prediction_switching_rate": 0,
        "dominant_prediction": "tiger cat",
        "dominant_class_index": 282,
        "mean_confidence": 0.94,
        "std_confidence": 0.01,
        "disagreement_with_raw": False,
        "transformation_sensitivity": 0.01,
    }

    res = risk_engine.evaluate(consistency, raw_pred)
    assert res["risk_level"] == "LOW"
    assert res["risk_score"] <= 30.0
    assert len(res["reasons"]) > 0

def test_risk_engine_high_risk():
    raw_pred = {"top1_label": "ostrich", "top1_class_index": 9, "top1_confidence": 0.91}
    consistency = {
        "prediction_agreement": 0.28,
        "prediction_switching_rate": 4,
        "dominant_prediction": "tiger cat",
        "dominant_class_index": 282,
        "mean_confidence": 0.65,
        "std_confidence": 0.25,
        "disagreement_with_raw": True,
        "transformation_sensitivity": 0.35,
    }

    res = risk_engine.evaluate(consistency, raw_pred)
    assert res["risk_level"] == "HIGH"
    assert res["risk_score"] > 65.0
    assert any("disagrees" in r for r in res["reasons"])
