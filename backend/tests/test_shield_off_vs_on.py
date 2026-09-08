from PIL import Image
from app.services.argus_shield_service import argus_shield_service

def test_shield_off_vs_on_behavior():
    img = Image.new("RGB", (299, 299), color=(150, 150, 150))

    # Shield OFF
    res_off = argus_shield_service.analyze(img, shield="off")
    assert res_off["shield"] == "off"
    assert res_off["mitigation"]["final_decision"] == "ACCEPTED"
    assert res_off["transformed_views"] is None
    assert res_off["consistency"] is None
    assert res_off["risk"] is None
    assert res_off["pipeline_stages"]["3_multiview_generation"] == "SKIPPED (Shield OFF)"

    # Shield ON
    res_on = argus_shield_service.analyze(img, shield="on")
    assert res_on["shield"] == "on"
    assert res_on["transformed_views"] is not None
    assert len(res_on["transformed_views"]) == 7
    assert res_on["consistency"] is not None
    assert res_on["risk"] is not None
    assert res_on["mitigation"] is not None
    assert "7_final_decision" in res_on["pipeline_stages"]
