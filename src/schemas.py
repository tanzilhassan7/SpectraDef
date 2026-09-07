from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class PreprocessConfig(BaseModel):
    """Configuration for OpenCV Image Preprocessing."""
    resize_width: Optional[int] = Field(default=None, description="Target width for resizing")
    resize_height: Optional[int] = Field(default=None, description="Target height for resizing")
    gaussian_blur: int = Field(default=0, description="Gaussian blur kernel size (must be odd integer like 3, 5, 7, 0 to disable)")
    clahe_contrast: bool = Field(default=False, description="Apply Contrast Limited Adaptive Histogram Equalization")
    brightness: float = Field(default=1.0, description="Brightness multiplier (1.0 = unchanged)")
    contrast: float = Field(default=1.0, description="Contrast multiplier (1.0 = unchanged)")
    sharpen: bool = Field(default=False, description="Apply image sharpening filter")
    color_space: str = Field(default="RGB", description="Target color space: RGB, BGR, GRAY, HSV")


class BoundingBox(BaseModel):
    """Standardized Bounding Box in both absolute and normalized formats."""
    # Absolute pixel coordinates
    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")
    
    # Normalized coordinates (0.0 to 1.0)
    x_center_norm: float = Field(..., description="Normalized center X coordinate")
    y_center_norm: float = Field(..., description="Normalized center Y coordinate")
    width_norm: float = Field(..., description="Normalized width")
    height_norm: float = Field(..., description="Normalized height")


class Detection(BaseModel):
    """Single object detection result standardized across the system."""
    id: int = Field(..., description="Detection index sequence number")
    class_id: int = Field(..., description="COCO or model class ID")
    class_name: str = Field(..., description="Human-readable class name (e.g. 'person', 'car')")
    confidence: float = Field(..., description="Detection confidence score between 0.0 and 1.0")
    bbox: BoundingBox = Field(..., description="Bounding box details")


class ImageMetadata(BaseModel):
    """Image dimensions and properties."""
    width: int
    height: int
    channels: int


class PipelineOutput(BaseModel):
    """Standardized result schema returned to the rest of the system."""
    success: bool = True
    total_detections: int = 0
    detections: List[Detection] = []
    metadata: ImageMetadata
    preprocess_time_ms: float = 0.0
    inference_time_ms: float = 0.0
    total_time_ms: float = 0.0
    annotated_image_base64: Optional[str] = None

    def to_summary(self) -> Dict[str, Any]:
        """Returns a clean summary dict for downstream system modules."""
        return {
            "success": self.success,
            "count": self.total_detections,
            "classes": [d.class_name for d in self.detections],
            "detections": [
                {
                    "class": d.class_name,
                    "confidence": round(d.confidence, 4),
                    "bbox_pixels": [round(d.bbox.x1, 1), round(d.bbox.y1, 1), round(d.bbox.x2, 1), round(d.bbox.y2, 1)],
                    "bbox_normalized": [round(d.bbox.x_center_norm, 4), round(d.bbox.y_center_norm, 4), round(d.bbox.width_norm, 4), round(d.bbox.height_norm, 4)]
                }
                for d in self.detections
            ],
            "latency_ms": round(self.total_time_ms, 2)
        }
