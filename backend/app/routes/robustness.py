from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io
from typing import Optional, List

from app.services.transformation_service import transformation_service
from app.services.model_service import model_service
from app.services.consistency_service import consistency_service
from app.services.risk_engine import risk_engine
from app.storage import storage
from app.schemas import RobustnessSweepResponse, SweepDataPoint

router = APIRouter(prefix="/api", tags=["robustness"])

@router.post("/robustness-sweep", response_model=RobustnessSweepResponse)
async def robustness_sweep(
    transform_name: str = Form("jpeg"),
    artifact_id: Optional[str] = Form(None),
    fixture_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    try:
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        elif artifact_id:
            artifact = storage.get_artifact(artifact_id)
            if not artifact:
                raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")
            image = Image.open(io.BytesIO(artifact["image_bytes"])).convert("RGB")
        elif fixture_id:
            fixture = storage.get_fixture(fixture_id)
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")
        else:
            fixture = storage.get_fixture("tiger_cat_benchmark")
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")

        raw_pred = model_service.predict(image)
        series: List[SweepDataPoint] = []
        strengths = [0.1, 0.3, 0.5, 0.7, 0.9]

        for s in strengths:
            transformed_views = transformation_service.run_multi_view_inference(
                image, transform_names=[transform_name, "blur", "noise"], strength=s
            )
            consistency = consistency_service.analyze(raw_pred, transformed_views)
            risk = risk_engine.evaluate(consistency, raw_pred)

            point_view = transformed_views[0]  # view for target transform_name
            series.append(
                SweepDataPoint(
                    strength=s,
                    prediction=point_view["prediction"],
                    confidence=point_view["confidence"],
                    agreement=consistency["prediction_agreement"],
                    risk_contribution=risk["risk_score"],
                )
            )

        return RobustnessSweepResponse(transform_name=transform_name, series=series)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Robustness sweep failed: {str(e)}")
