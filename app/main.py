import os
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from src.pipeline import YOLOPipeline
from src.schemas import PreprocessConfig, PipelineOutput

app = FastAPI(
    title="SpectraDef - YOLO Image Input & Baseline Detection API",
    description="Standardized object detection pipeline with OpenCV preprocessing and YOLOv8 inference.",
    version="1.0.0"
)

# Enable CORS for cross-origin requests from other team components
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[YOLOPipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    print("🚀 Initializing YOLO Pipeline Engine...")
    pipeline = YOLOPipeline(model_name="yolov8n.pt", conf_threshold=0.25)
    print("✅ Pipeline ready!")


# Mount static assets for frontend dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SpectraDef API Ready</h1><p>Frontend loading...</p>")


@app.get("/api/health")
async def health_check():
    return {
        "status": "online",
        "service": "SpectraDef YOLO Pipeline Engine",
        "model_loaded": pipeline is not None and pipeline.detector is not None,
        "classes_count": len(pipeline.detector.get_class_names()) if pipeline else 0
    }


@app.get("/api/classes")
async def get_classes():
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline engine not initialized.")
    return {"classes": pipeline.detector.get_class_names()}


@app.post("/api/detect/image", response_model=PipelineOutput)
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = Form(0.25),
    resize_width: Optional[int] = Form(None),
    gaussian_blur: int = Form(0),
    clahe_contrast: bool = Form(False),
    brightness: float = Form(1.0),
    contrast: float = Form(1.0),
    sharpen: bool = Form(False),
    color_space: str = Form("RGB")
):
    """
    Process an uploaded image file through OpenCV preprocessor + YOLOv8 detector.
    Returns clean standardized detection result.
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline engine not initialized.")

    try:
        contents = await file.read()
        config = PreprocessConfig(
            resize_width=resize_width,
            gaussian_blur=gaussian_blur,
            clahe_contrast=clahe_contrast,
            brightness=brightness,
            contrast=contrast,
            sharpen=sharpen,
            color_space=color_space
        )

        result = pipeline.process(
            image_input=contents,
            config=config,
            conf_threshold=conf_threshold,
            return_annotated_base64=True
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/api/detect/frame", response_model=PipelineOutput)
async def detect_frame(
    frame_data: dict
):
    """
    Process base64 encoded video frame (from webcam stream).
    """
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline engine not initialized.")

    try:
        base64_image = frame_data.get("image")
        conf_threshold = frame_data.get("conf_threshold", 0.25)
        blur = frame_data.get("gaussian_blur", 0)
        clahe = frame_data.get("clahe_contrast", False)
        sharpen = frame_data.get("sharpen", False)

        if not base64_image:
            raise HTTPException(status_code=400, detail="Missing base64 image field.")

        # Remove data:image header if present
        if "," in base64_image:
            base64_image = base64_image.split(",")[1]

        image_bytes = base64.b64decode(base64_image)
        config = PreprocessConfig(
            gaussian_blur=blur,
            clahe_contrast=clahe,
            sharpen=sharpen
        )

        result = pipeline.process(
            image_input=image_bytes,
            config=config,
            conf_threshold=conf_threshold,
            return_annotated_base64=True
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Frame processing failed: {str(e)}")
