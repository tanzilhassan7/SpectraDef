from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Response
from PIL import Image
import io
import json
from typing import Optional

from app.services.adversarial_generation_service import adversarial_generation_service
from app.services.model_service import model_service
from app.storage import storage
from app.schemas import (
    AdversarialGenerateResponse,
    AdversarialVerifyRequest,
    PredictionResponse,
)

router = APIRouter(prefix="/api/adversarial", tags=["adversarial"])

@router.post("/generate", response_model=AdversarialGenerateResponse)
@router.post("/generate/", response_model=AdversarialGenerateResponse)
async def generate_adversarial(
    fixture_id: Optional[str] = Form(None),
    method: str = Form("iterative_target"),
    target_class: int = Form(9),
    strength_preset: str = Form("moderate"),
    iterations_preset: int = Form(10),
    file: Optional[UploadFile] = File(None),
):
    try:
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        elif fixture_id:
            fixture = storage.get_fixture(fixture_id)
            if not fixture:
                raise HTTPException(status_code=404, detail=f"Fixture '{fixture_id}' not found")
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")
        else:
            fixture = storage.get_fixture("tiger_cat_benchmark")
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")

        res = adversarial_generation_service.generate(
            image=image,
            method=method,
            target_class=target_class,
            strength_preset=strength_preset,
            iterations_preset=iterations_preset,
        )
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Adversarial generation failed: {str(e)}")

@router.post("/verify", response_model=PredictionResponse)
@router.post("/verify/", response_model=PredictionResponse)
async def verify_adversarial(payload: AdversarialVerifyRequest):
    artifact = storage.get_artifact(payload.artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact '{payload.artifact_id}' not found")

    image = Image.open(io.BytesIO(artifact["image_bytes"])).convert("RGB")
    res = model_service.predict(image)
    return res

@router.get("/artifact/{artifact_id}")
@router.get("/artifact/{artifact_id}/")
async def get_artifact_image(artifact_id: str):
    artifact = storage.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")
    return Response(content=artifact["image_bytes"], media_type="image/png")

@router.get("/{artifact_id}/download")
@router.get("/{artifact_id}/download/")
async def download_adversarial_artifact(artifact_id: str):
    """Serves the generated adversarial image artifact as a downloadable lossless PNG file."""
    artifact = storage.get_artifact(artifact_id)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Artifact '{artifact_id}' not found")
    
    headers = {
        "Content-Disposition": f'attachment; filename="adversarial_{artifact_id}.png"'
    }
    return Response(content=artifact["image_bytes"], media_type="image/png", headers=headers)
