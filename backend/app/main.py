import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.services.model_service import model_service
from app.routes.prediction import router as prediction_router
from app.routes.adversarial import router as adversarial_router
from app.routes.argus import router as argus_router
from app.routes.robustness import router as robustness_router
from app.routes.fixtures import router as fixtures_router
from app.routes.demo import router as demo_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ARGUS Shield backend starting up...")
    _ = model_service.model
    logger.info(f"ModelService pre-warmed on device: {model_service.device}")
    yield
    logger.info("ARGUS Shield backend shutting down...")

app = FastAPI(
    title="ARGUS SHIELD API",
    description="Computer Vision Robustness & Anomalous Prediction Detection Layer",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Router mounting
app.include_router(prediction_router)
app.include_router(adversarial_router)
app.include_router(argus_router)
app.include_router(robustness_router)
app.include_router(fixtures_router)
app.include_router(demo_router)

@app.get("/api")
@app.get("/api/")
async def root():
    return {
        "status": "online",
        "system": "ARGUS SHIELD",
        "model": "InceptionV3",
        "device": str(model_service.device),
    }

# Serve built frontend static files if present (for production deployment e.g. Render)
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../frontend/dist"))
if os.path.exists(frontend_dist):
    logger.info(f"Mounting static frontend files from: {frontend_dist}")
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception handled: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"},
    )
