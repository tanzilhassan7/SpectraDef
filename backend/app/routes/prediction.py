from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from PIL import Image
import io
from typing import Optional

from app.services.model_service import model_service
from app.storage import storage
from app.schemas import PredictionResponse

router = APIRouter(prefix="/api", tags=["prediction"])

@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
    fixture_id: Optional[str] = Form(None),
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

        res = model_service.predict(image)
        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
