from app.services.mitigation_service import mitigation_service

def test_mitigation_low_risk():
    raw_pred = {"top1_label": "tiger cat", "top1_confidence": 0.95}
    risk = {"risk_level": "LOW", "risk_score": 10.0}
    consistency = {"prediction_agreement": 1.0, "dominant_prediction": "tiger cat"}

    res = mitigation_service.decide(raw_pred, [], consistency, risk)
    assert res["final_decision"] == "ACCEPTED"
    assert res["final_prediction"] == "tiger cat"

def test_mitigation_high_risk_consensus_recovery():
    raw_pred = {"top1_label": "ostrich", "top1_confidence": 0.91}
    risk = {"risk_level": "HIGH", "risk_score": 85.0}
    views = [
        {"prediction": "tiger cat", "confidence": 0.85},
        {"prediction": "tiger cat", "confidence": 0.82},
        {"prediction": "tiger cat", "confidence": 0.88},
        {"prediction": "ostrich", "confidence": 0.40},
    ]
    consistency = {"prediction_agreement": 0.75, "dominant_prediction": "tiger cat"}

    res = mitigation_service.decide(raw_pred, views, consistency, risk)
    assert res["final_decision"] == "CONSENSUS_RECOVERED"
    assert res["final_prediction"] == "tiger cat"

def test_mitigation_high_risk_abstain():
    raw_pred = {"top1_label": "ostrich", "top1_confidence": 0.91}
    risk = {"risk_level": "HIGH", "risk_score": 95.0}
    views = [
        {"prediction": "lion", "confidence": 0.30},
        {"prediction": "banana", "confidence": 0.35},
        {"prediction": "tiger cat", "confidence": 0.32},
        {"prediction": "ostrich", "confidence": 0.20},
    ]
    consistency = {"prediction_agreement": 0.25, "dominant_prediction": "banana"}

    res = mitigation_service.decide(raw_pred, views, consistency, risk)
    assert res["final_decision"] == "ABSTAIN"
    assert res["final_prediction"] == "ABSTAIN"
