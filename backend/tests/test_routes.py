import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_route_contract_fixtures():
    response = client.get("/api/fixtures")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert data[0]["id"] == "tiger_cat_benchmark"

def test_route_contract_predict():
    response = client.post("/api/predict", data={"fixture_id": "tiger_cat_benchmark"})
    assert response.status_code == 200
    data = response.json()
    assert "top1_label" in data
    assert "top1_confidence" in data

def test_route_contract_argus_analyze_post():
    # Route contract test hitting /api/argus/analyze with POST method
    response = client.post("/api/argus/analyze", data={"fixture_id": "tiger_cat_benchmark", "shield": "on"})
    assert response.status_code == 200
    data = response.json()
    assert data["shield"] == "on"
    assert "raw_prediction" in data
    assert "pipeline_stages" in data

def test_route_contract_adversarial_generate():
    response = client.post(
        "/api/adversarial/generate",
        data={
            "fixture_id": "tiger_cat_benchmark",
            "method": "iterative_target",
            "target_class": 9,
            "strength_preset": "moderate",
            "iterations_preset": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "generation_status" in data
