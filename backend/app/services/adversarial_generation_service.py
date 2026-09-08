import io
import base64
import datetime
from typing import Dict, Any, Tuple
from PIL import Image, ImageOps
import torch

from app.config import config
from app.generators import GENERATORS
from app.services.model_service import model_service
from app.imagenet_labels import get_label
from app.storage import storage

class AdversarialGenerationService:
    def generate(
        self,
        image: Image.Image,
        method: str = "iterative_target",
        target_class: int = 9,
        strength_preset: str = "moderate",
        iterations_preset: int = 10,
    ) -> Dict[str, Any]:
        # Server-side validation of parameters
        if method not in GENERATORS:
            method = "iterative_target"

        if target_class not in config.APPROVED_TARGET_CLASSES:
            target_class = config.DEFAULT_TARGET_CLASS

        epsilon = config.EPSILON_PRESETS.get(strength_preset, 0.05)
        epsilon = max(config.MIN_EPSILON, min(config.MAX_EPSILON, float(epsilon)))
        
        iterations = max(config.MIN_ITERATIONS, min(config.MAX_ITERATIONS, int(iterations_preset)))

        # Clean source prediction
        clean_pred = model_service.predict(image)
        source_label = clean_pred["top1_label"]
        source_confidence = clean_pred["top1_confidence"]

        # Run generator
        generator_cls = GENERATORS[method]
        generator = generator_cls()
        adv_image, diff_tensor = generator.generate(
            image=image,
            epsilon=epsilon,
            iterations=iterations,
            target_class=target_class,
        )

        # Run real inference on generated image
        adv_pred = model_service.predict(adv_image)
        adv_label = adv_pred["top1_label"]
        adv_confidence = adv_pred["top1_confidence"]
        adv_class_index = adv_pred["top1_class_index"]

        target_label = get_label(target_class)

        # Verify generation outcome
        if method in ["iterative_target", "one_step_targeted"]:
            status = "success" if adv_class_index == target_class else "failed"
            status_reason = None if status == "success" else f"Generation did not reach target condition '{target_label}' (got '{adv_label}')"
        else:
            status = "success" if adv_label != source_label else "failed"
            status_reason = None if status == "success" else "Generation did not alter initial prediction"

        # Save adversarial image to PNG bytes (lossless encoding preserves pixel-level adversarial noise)
        buffer = io.BytesIO()
        adv_image.save(buffer, format="PNG")
        adv_bytes = buffer.getvalue()

        # Generate amplified difference visualization (scaled x10 for clear visual rendering)
        diff_scaled = torch.clamp(diff_tensor * 10.0, 0.0, 1.0)
        diff_pil = generator.tensor_to_pil(diff_scaled)
        diff_buffer = io.BytesIO()
        diff_pil.save(diff_buffer, format="PNG")
        diff_b64 = base64.b64encode(diff_buffer.getvalue()).decode("utf-8")

        # Store artifact
        metadata = {
            "model": "InceptionV3",
            "dataset": "ImageNet",
            "source_prediction": source_label,
            "source_confidence": source_confidence,
            "target_class": target_label,
            "target_class_index": target_class,
            "method": method,
            "parameters": {
                "epsilon": epsilon,
                "iterations": iterations,
                "strength_preset": strength_preset,
            },
            "generation_status": status,
            "status_reason": status_reason,
            "adversarial_prediction": adv_label,
            "adversarial_confidence": adv_confidence,
            "created_at": datetime.datetime.utcnow().isoformat() + "Z",
            "purpose": "ARGUS controlled robustness evaluation",
        }

        artifact_id = storage.store_artifact(adv_bytes, metadata)

        return {
            "id": artifact_id,
            "model": metadata["model"],
            "dataset": metadata["dataset"],
            "source_prediction": source_label,
            "source_confidence": source_confidence,
            "target_class": target_label,
            "method": method,
            "parameters": metadata["parameters"],
            "generation_status": status,
            "status_reason": status_reason,
            "adversarial_prediction": adv_label,
            "adversarial_confidence": adv_confidence,
            "created_at": metadata["created_at"],
            "purpose": metadata["purpose"],
            "diff_image_base64": f"data:image/png;base64,{diff_b64}",
        }

adversarial_generation_service = AdversarialGenerationService()
