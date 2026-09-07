import cv2
import numpy as np
from typing import Tuple
from .schemas import PreprocessConfig


class ImagePreprocessor:
    """
    OpenCV Preprocessing Engine for YOLO object detection pipeline.
    Provides image enhancement, resizing, noise reduction, and color conversion routines.
    """

    @staticmethod
    def preprocess(image: np.ndarray, config: PreprocessConfig) -> Tuple[np.ndarray, dict]:
        """
        Applies configured OpenCV preprocessing pipeline on input BGR image matrix.
        Returns preprocessed image matrix and metadata dict.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image matrix is empty or invalid.")

        img = image.copy()
        orig_h, orig_w = img.shape[:2]

        # 1. Resize if requested
        if config.resize_width and config.resize_height:
            img = cv2.resize(img, (config.resize_width, config.resize_height), interpolation=cv2.INTER_LINEAR)
        elif config.resize_width:
            aspect_ratio = orig_h / orig_w
            new_h = int(config.resize_width * aspect_ratio)
            img = cv2.resize(img, (config.resize_width, new_h), interpolation=cv2.INTER_LINEAR)
        elif config.resize_height:
            aspect_ratio = orig_w / orig_h
            new_w = int(config.resize_height * aspect_ratio)
            img = cv2.resize(img, (new_w, config.resize_height), interpolation=cv2.INTER_LINEAR)

        # 2. Gaussian Blur for noise reduction
        if config.gaussian_blur > 0:
            ksize = config.gaussian_blur
            if ksize % 2 == 0:
                ksize += 1  # Kernel size must be odd
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        # 3. Brightness and Contrast Adjustment
        if config.brightness != 1.0 or config.contrast != 1.0:
            img = cv2.convertScaleAbs(img, alpha=config.contrast, beta=(config.brightness - 1.0) * 50)

        # 4. CLAHE Contrast Limited Adaptive Histogram Equalization
        if config.clahe_contrast:
            if len(img.shape) == 3 and img.shape[2] == 3:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            elif len(img.shape) == 2:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)

        # 5. Image Sharpening Filter
        if config.sharpen:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            img = cv2.filter2D(img, -1, kernel)

        # 6. Color Space Conversions
        color_mode = config.color_space.upper()
        if color_mode == "GRAY":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel for YOLO compatibility
        elif color_mode == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif color_mode == "RGB":
            # YOLO ultralytics handles BGR images directly, but we provide RGB flag for downstream export
            pass

        h, w = img.shape[:2]
        c = img.shape[2] if len(img.shape) == 3 else 1

        meta = {
            "original_width": orig_w,
            "original_height": orig_h,
            "processed_width": w,
            "processed_height": h,
            "channels": c,
        }

        return img, meta
