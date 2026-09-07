import time
import cv2
import numpy as np
from typing import Optional, Union, Dict, Any

from .schemas import PreprocessConfig, PipelineOutput, ImageMetadata
from .preprocessor import ImagePreprocessor
from .yolo_detector import YOLODetector


class YOLOPipeline:
    """
    Master Image Input + YOLO Detection Pipeline.
    Combines input validation, OpenCV preprocessing, YOLOv8 inference,
    and returns clean standardized system results.
    """

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.25):
        """
        Initialize pipeline with YOLO model.
        """
        self.detector = YOLODetector(model_name=model_name, conf_threshold=conf_threshold)

    def process(
        self,
        image_input: Union[np.ndarray, str, bytes],
        config: Optional[PreprocessConfig] = None,
        conf_threshold: Optional[float] = None,
        return_annotated_base64: bool = True
    ) -> PipelineOutput:
        """
        Runs complete pipeline on input image.
        :param image_input: OpenCV BGR matrix, image file path (str), or raw image bytes
        :param config: PreprocessConfig object or None for defaults
        :param conf_threshold: Optional override for confidence threshold
        :param return_annotated_base64: Whether to generate base64 visualization string
        :return: Standardized PipelineOutput object
        """
        total_start = time.time()
        if config is None:
            config = PreprocessConfig()

        # 1. Image Loading / Decoding
        if isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not read image file at path: {image_input}")
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Could not decode image from input bytes.")
        elif isinstance(image_input, np.ndarray):
            image = image_input
        else:
            raise TypeError("Unsupported image input type. Must be np.ndarray, file path string, or bytes.")

        # 2. OpenCV Preprocessing
        preproc_start = time.time()
        processed_img, meta = ImagePreprocessor.preprocess(image, config)
        preprocess_time_ms = (time.time() - preproc_start) * 1000.0

        # 3. YOLO Inference
        detections, inference_time_ms = self.detector.predict(
            image=processed_img,
            conf=conf_threshold
        )

        # 4. Annotation Visualization
        base64_str = None
        if return_annotated_base64:
            annotated_img = self.detector.annotate_image(processed_img, detections)
            base64_str = YOLODetector.to_base64(annotated_img)

        total_time_ms = (time.time() - total_start) * 1000.0

        # Metadata
        h, w = processed_img.shape[:2]
        c = processed_img.shape[2] if len(processed_img.shape) == 3 else 1

        output = PipelineOutput(
            success=True,
            total_detections=len(detections),
            detections=detections,
            metadata=ImageMetadata(width=w, height=h, channels=c),
            preprocess_time_ms=round(preprocess_time_ms, 2),
            inference_time_ms=round(inference_time_ms, 2),
            total_time_ms=round(total_time_ms, 2),
            annotated_image_base64=base64_str
        )

        return output
