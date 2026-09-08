from fastapi import APIRouter, HTTPException
from PIL import Image
import io

from app.storage import storage
from app.services.model_service import model_service
from app.services.adversarial_generation_service import adversarial_generation_service
from app.services.argus_shield_service import argus_shield_service

router = APIRouter(prefix="/api/demo", tags=["demo"])

@router.post("/run")
async def run_scripted_demo():
    try:
        # Step 1: Fetch benchmark fixture (Tiger Cat)
        fixture = storage.get_fixture("tiger_cat_benchmark")
        if not fixture:
            raise HTTPException(status_code=500, detail="Benchmark fixture not available")
        clean_img = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")
        clean_pred = model_service.predict(clean_img)

        # Step 2: Generate adversarial artifact (target: Ostrich)
        gen_res = adversarial_generation_service.generate(
            image=clean_img,
            method="iterative_target",
            target_class=9,  # Ostrich
            strength_preset="moderate",
            iterations_preset=10,
        )

        artifact_id = gen_res["id"]
        artifact = storage.get_artifact(artifact_id)
        adv_img = Image.open(io.BytesIO(artifact["image_bytes"])).convert("RGB")

        # Step 3: Shield OFF analysis
        shield_off_res = argus_shield_service.analyze(adv_img, shield="off")

        # Step 4: Shield ON analysis
        shield_on_res = argus_shield_service.analyze(adv_img, shield="on")

        return {
            "demo_scenario": "Tiger Cat -> Ostrich Adversarial Robustness Benchmark",
            "clean_prediction": clean_pred,
            "adversarial_generation": gen_res,
            "shield_off_result": shield_off_res,
            "shield_on_result": shield_on_res,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scripted demo failed: {str(e)}")
