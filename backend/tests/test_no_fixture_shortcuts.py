import inspect
from app.services import argus_shield_service, model_service, consistency_service, risk_engine, mitigation_service

def test_assert_no_fixture_id_branching_or_shortcuts():
    """
    CRITICAL GROUND-TRUTH RULE:
    Verify that none of the core services contain hardcoded fixture shortcuts or label overrides.
    """
    services = [
        argus_shield_service.ArgusShieldService,
        model_service.ModelService,
        consistency_service.ConsistencyService,
        risk_engine.RiskEngine,
        mitigation_service.MitigationService,
    ]

    for s in services:
        source_code = inspect.getsource(s)
        assert "tiger_cat_ostrich" not in source_code
        assert "fixture_id ==" not in source_code
        assert "if fixture_id" not in source_code
