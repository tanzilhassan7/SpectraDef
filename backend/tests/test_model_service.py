import pytest
from PIL import Image
from app.services.model_service import model_service

def test_model_service_prediction():
    img = Image.new("RGB", (299, 299), color=(100, 150, 200))
    res = model_service.predict(img)

    assert "top1_label" in res
    assert "top1_class_index" in res
    assert "top1_confidence" in res
    assert "top5" in res
    assert len(res["top5"]) == 5
    assert "latency_ms" in res
    assert res["top1_confidence"] >= 0.0 and res["top1_confidence"] <= 1.0
