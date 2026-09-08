from fastapi import APIRouter, File, UploadFile, Form, Query, HTTPException, Request
from PIL import Image
import io
from typing import Optional

from app.services.argus_shield_service import argus_shield_service
from app.storage import storage
from app.schemas import ArgusAnalyzeResponse

router = APIRouter(prefix="/api/argus", tags=["argus"])

@router.post("/analyze", response_model=ArgusAnalyzeResponse)
@router.post("/analyze/", response_model=ArgusAnalyzeResponse)
@router.get("/analyze", response_model=ArgusAnalyzeResponse)
@router.get("/analyze/", response_model=ArgusAnalyzeResponse)
async def analyze_image(
    artifact_id: Optional[str] = Form(None),
    fixture_id: Optional[str] = Form(None),
    shield: str = Form("on"),
    file: Optional[UploadFile] = File(None),
    # Support query params for GET requests
    artifact_id_query: Optional[str] = Query(None, alias="artifact_id"),
    fixture_id_query: Optional[str] = Query(None, alias="fixture_id"),
    shield_query: Optional[str] = Query(None, alias="shield"),
):
    try:
        art_id = artifact_id or artifact_id_query
        fix_id = fixture_id or fixture_id_query
        shield_val = shield if shield != "on" or not shield_query else shield_query

        image = None
        if file is not None:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        elif art_id:
            artifact = storage.get_artifact(art_id)
            if not artifact:
                raise HTTPException(status_code=404, detail=f"Artifact '{art_id}' not found")
            image = Image.open(io.BytesIO(artifact["image_bytes"])).convert("RGB")
        elif fix_id:
            fixture = storage.get_fixture(fix_id)
            if not fixture:
                raise HTTPException(status_code=404, detail=f"Fixture '{fix_id}' not found")
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")
        else:
            fixture = storage.get_fixture("tiger_cat_benchmark")
            image = Image.open(io.BytesIO(fixture["image_bytes"])).convert("RGB")

        res = argus_shield_service.analyze(image=image, shield=shield_val)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ARGUS analysis failed: {str(e)}")
