import time
import base64
import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO

from .schemas import Detection, BoundingBox


class YOLODetector:
    """
    Ultralytics YOLOv8 Inference wrapper and detection parser.
    Extracts class names, bounding boxes, and confidence scores into standardized structures.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize the YOLO detector.
        :param model_name: Pretrained weights filename or custom model path (e.g. 'yolov8n.pt')
        :param conf_threshold: Default confidence threshold
        :param iou_threshold: Default NMS IoU threshold
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print(f"[YOLODetector] Loading model weights: {model_name}...")
        self.model = YOLO(model_name)
        print(f"[YOLODetector] Model loaded successfully! Registered classes: {len(self.model.names)}")

    def get_class_names(self) -> dict:
        """Returns map of class IDs to class names."""
        return self.model.names

    def predict(
        self,
        image: np.ndarray,
        conf: Optional[float] = None,
        iou: Optional[float] = None
    ) -> Tuple[List[Detection], float]:
        """
        Runs YOLOv8 inference on input image.
        Returns list of standardized Detection objects and inference latency in milliseconds.
        """
        conf_val = conf if conf is not None else self.conf_threshold
        iou_val = iou if iou is not None else self.iou_threshold

        start_time = time.time()
        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf_val,
            iou=iou_val,
            verbose=False
        )
        inference_time_ms = (time.time() - start_time) * 1000.0

        detections: List[Detection] = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            orig_h, orig_w = image.shape[:2]

            for idx, box in enumerate(boxes):
                # Extract coordinates
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                w_px = max(0.0, x2 - x1)
                h_px = max(0.0, y2 - y1)

                # Normalized coordinates
                x_center_norm = ((x1 + x2) / 2.0) / orig_w
                y_center_norm = ((y1 + y2) / 2.0) / orig_h
                w_norm = w_px / orig_w
                h_norm = h_px / orig_h

                # Class and confidence
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                bbox_obj = BoundingBox(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    width=w_px,
                    height=h_px,
                    x_center_norm=x_center_norm,
                    y_center_norm=y_center_norm,
                    width_norm=w_norm,
                    height_norm=h_norm
                )

                detection = Detection(
                    id=idx + 1,
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=confidence,
                    bbox=bbox_obj
                )
                detections.append(detection)

        return detections, inference_time_ms

    def annotate_image(
        self,
        image: np.ndarray,
        detections: List[Detection],
        draw_labels: bool = True,
        draw_confidence: bool = True
    ) -> np.ndarray:
        """
        Draws custom high-visibility bounding boxes and labels on an image.
        """
        annotated = image.copy()
        
        # Color palette generator based on class ID
        def get_color(cls_id: int):
            colors = [
                (0, 220, 255),    # Neon Cyan
                (50, 255, 120),   # Neon Green
                (255, 100, 50),   # Bright Orange
                (230, 80, 255),   # Magenta
                (255, 220, 0),    # Bright Yellow
                (100, 160, 255),  # Soft Blue
            ]
            return colors[cls_id % len(colors)]

        for d in detections:
            color = get_color(d.class_id)
            x1, y1, x2, y2 = int(d.bbox.x1), int(d.bbox.y1), int(d.bbox.x2), int(d.bbox.y2)

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Prepare text label
            if draw_labels:
                label = d.class_name
                if draw_confidence:
                    label += f" {d.confidence * 100:.1f}%"

                # Text background box
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw label background header
                bg_y1 = max(0, y1 - text_h - 8)
                bg_y2 = max(y1, text_h + 8)
                cv2.rectangle(annotated, (x1, bg_y1), (x1 + text_w + 10, y1), color, -1)
                
                # Text string in contrasting dark color
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        return annotated

    @staticmethod
    def to_base64(image: np.ndarray, format: str = ".jpg") -> str:
        """Converts an OpenCV image matrix to base64 string."""
        success, buffer = cv2.imencode(format, image)
        if not success:
            raise ValueError("Failed to encode image to base64.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
