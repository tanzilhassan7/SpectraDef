import io
from PIL import Image
from typing import Dict, Any, Optional

from app.services.model_service import model_service
from app.services.transformation_service import transformation_service
from app.services.consistency_service import consistency_service
from app.services.risk_engine import risk_engine
from app.services.mitigation_service import mitigation_service

class ArgusShieldService:
    def analyze(
        self, image: Image.Image, shield: str = "on"
    ) -> Dict[str, Any]:
        """
        Orchestrates the analysis pipeline.
        shield: 'off' | 'on'
        """
        pipeline_stages = {}

        # Stage 1: Input Validation
        try:
            if image.mode != "RGB":
                image = image.convert("RGB")
            width, height = image.size
            if width < 10 or height < 10:
                raise ValueError("Image dimensions too small")
            pipeline_stages["1_validation"] = "PASSED"
        except Exception as e:
            pipeline_stages["1_validation"] = f"FAILED: {str(e)}"
            raise ValueError(f"Input image validation failed: {str(e)}")

        # Stage 2: Base Inference (Raw prediction)
        raw_pred = model_service.predict(image)
        pipeline_stages["2_base_inference"] = "COMPLETE"

        if shield.lower() == "off":
            # Shield OFF path: Accept raw prediction blindly without analysis
            pipeline_stages["3_multiview_generation"] = "SKIPPED (Shield OFF)"
            pipeline_stages["4_consistency_analysis"] = "SKIPPED (Shield OFF)"
            pipeline_stages["5_risk_engine"] = "SKIPPED (Shield OFF)"
            pipeline_stages["6_mitigation"] = "ACCEPTED_UNCONDITIONALLY"

            mitigation_res = {
                "final_decision": "ACCEPTED",
                "final_prediction": raw_pred["top1_label"],
                "final_confidence": raw_pred["top1_confidence"],
                "mitigation_action": "Shield OFF mode: Raw prediction accepted blindly without stability verification.",
            }

            return {
                "shield": "off",
                "raw_prediction": raw_pred,
                "transformed_views": None,
                "consistency": None,
                "risk": None,
                "mitigation": mitigation_res,
                "pipeline_stages": pipeline_stages,
            }

        # Shield ON path — 7-stage pipeline
        # Stage 3: Multi-view transformation & inference
        views = transformation_service.run_multi_view_inference(image)
        pipeline_stages["3_multiview_generation"] = f"COMPLETE ({len(views)} views)"

        # Prepare JSON-safe view items (remove internal _image_pil reference)
        json_views = []
        for v in views:
            view_item = {k: val for k, val in v.items() if k != "_image_pil"}
            json_views.append(view_item)

        # Stage 4: Consistency Analysis
        consistency_res = consistency_service.analyze(raw_pred, views)
        pipeline_stages["4_consistency_analysis"] = "COMPLETE"

        # Stage 5: Risk Engine
        risk_res = risk_engine.evaluate(consistency_res, raw_pred)
        pipeline_stages["5_risk_engine"] = f"COMPLETE ({risk_res['risk_level']} risk: {risk_res['risk_score']}/100)"

        # Stage 6: Threat Level & Mitigation
        mitigation_res = mitigation_service.decide(raw_pred, views, consistency_res, risk_res, image=image)
        pipeline_stages["6_mitigation"] = f"EXECUTED ({mitigation_res['final_decision']})"

        # Stage 7: Final Decision
        pipeline_stages["7_final_decision"] = "DELIVERED"

        return {
            "shield": "on",
            "raw_prediction": raw_pred,
            "transformed_views": json_views,
            "consistency": consistency_res,
            "risk": risk_res,
            "mitigation": mitigation_res,
            "pipeline_stages": pipeline_stages,
        }

argus_shield_service = ArgusShieldService()
