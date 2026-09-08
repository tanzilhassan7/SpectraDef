import io
import base64
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
from typing import Dict, Any, List, Tuple

from app.config import config
from app.services.model_service import model_service

class TransformationService:
    def apply_transform(
        self, image: Image.Image, name: str, strength: float = 0.6
    ) -> Image.Image:
        """
        Applies a semantically-preserving input transformation.
        strength is a float between 0.0 and 1.0.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        if name == "blur":
            radius = 1.0 + strength * 2.5  # radius 1.0 to 3.5
            return image.filter(ImageFilter.GaussianBlur(radius=radius))

        elif name == "jpeg":
            quality = int(75 - strength * 45)  # quality 75 down to 30
            quality = max(20, min(95, quality))
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert("RGB")

        elif name == "resize":
            # Downsample round-trip
            scale = 1.0 - (strength * 0.45)  # scale 1.0 down to 0.55
            orig_w, orig_h = image.size
            new_w = max(32, int(orig_w * scale))
            new_h = max(32, int(orig_h * scale))
            small = image.resize((new_w, new_h), Image.BILINEAR)
            return small.resize((orig_w, orig_h), Image.BILINEAR)

        elif name == "brightness":
            factor = 1.0 + (0.25 * (1.0 if strength > 0.5 else -1.0) * strength)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)

        elif name == "contrast":
            factor = 1.0 + (0.3 * strength)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)

        elif name == "noise":
            arr = np.array(image, dtype=np.float32)
            std = strength * 25.0  # std 0 to 25
            noise = np.random.normal(0, std, arr.shape)
            noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_arr)

        elif name == "sharpen":
            enhancer = ImageEnhance.Sharpness(image)
            factor = 1.0 + (2.0 * strength)
            return enhancer.enhance(factor)

        elif name.startswith("random_smoothing") or name.startswith("gaussian_noise"):
            arr = np.array(image, dtype=np.float32)
            std = strength * 255.0 if strength < 1.0 else strength  # std in 0-255 scale
            noise = np.random.normal(0, std, arr.shape)
            noisy_arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            return Image.fromarray(noisy_arr)

        else:
            return image

    def run_multi_view_inference(
        self,
        image: Image.Image,
        transform_names: List[str] = None,
        strength: float = 0.6,
        include_smoothing: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Runs batched multi-view transformation pipeline.
        Includes deterministic views + multi-sample randomized smoothing views.
        """
        if transform_names is None:
            transform_names = list(config.DEFAULT_TRANSFORMS)

        transformed_imgs = []
        view_metadata = []

        # 1. Deterministic Multi-View Transforms
        for name in transform_names:
            t_img = self.apply_transform(image, name, strength=strength)
            transformed_imgs.append(t_img)
            view_metadata.append({
                "name": name,
                "category": "deterministic",
                "strength": strength,
            })

        # 2. Randomized Smoothing Category (Repeated Gaussian Noise across noise levels)
        if include_smoothing:
            noise_levels = config.RANDOMIZED_SMOOTHING_NOISE_LEVELS
            repeats = config.RANDOMIZED_SMOOTHING_REPEATS
            for n_idx, noise_std in enumerate(noise_levels):
                for r in range(repeats):
                    view_name = f"rand_smoothing_n{n_idx+1}_rep{r+1}"
                    t_img = self.apply_transform(image, "random_smoothing", strength=noise_std)
                    transformed_imgs.append(t_img)
                    view_metadata.append({
                        "name": view_name,
                        "category": "randomized_smoothing",
                        "strength": round(noise_std, 3),
                    })

        # Batched inference over all transformed images in 1 GPU/CPU pass
        predictions = model_service.predict_batch(transformed_imgs)

        views = []
        for meta, pred, t_img in zip(view_metadata, predictions, transformed_imgs):
            # Base64 thumbnail for UI
            thumb = t_img.resize((150, 150), Image.BILINEAR)
            buf = io.BytesIO()
            thumb.save(buf, format="JPEG", quality=80)
            b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

            views.append({
                "name": meta["name"],
                "category": meta["category"],
                "strength": meta["strength"],
                "prediction": pred["top1_label"],
                "class_index": pred["top1_class_index"],
                "confidence": pred["top1_confidence"],
                "margin": pred.get("margin", 0.0),
                "entropy": pred.get("entropy", 0.0),
                "latency_ms": pred["latency_ms"],
                "image_base64": f"data:image/jpeg;base64,{b64_img}",
                "_image_pil": t_img,
            })

        return views

transformation_service = TransformationService()
