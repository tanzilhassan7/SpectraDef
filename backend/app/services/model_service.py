import time
import logging
import math
from typing import Dict, Any, List, Tuple, Union
import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
import torchvision.transforms.functional as TF
from PIL import Image

from app.imagenet_labels import get_label

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info("Initializing ModelService and loading InceptionV3 weights...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weights = Inception_V3_Weights.IMAGENET1K_V1
        self.model = inception_v3(weights=self.weights)
        self.model.to(self.device)
        self.model.eval()  # Explicit eval mode
        self.transforms = self.weights.transforms()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        logger.info(f"InceptionV3 loaded successfully on device: {self.device}")

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # If image is already 299x299, avoid bilinear resize resampling to preserve fine pixel details
        if image.size == (299, 299):
            tensor = TF.to_tensor(image).unsqueeze(0).to(self.device)
            mean = self.mean.to(self.device)
            std = self.std.to(self.device)
            return (tensor - mean) / std
        else:
            tensor = self.transforms(image)
            return tensor.unsqueeze(0).to(self.device)

    def normalize_tensor_0_1(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies ImageNet mean & std normalization to [0, 1] input tensor."""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        if tensor.min() >= -0.1 and tensor.max() <= 1.1:
            mean = self.mean.to(self.device)
            std = self.std.to(self.device)
            return (tensor - mean) / std
        return tensor

    def predict(self, image_input: Union[Image.Image, torch.Tensor]) -> Dict[str, Any]:
        """
        Accepts either PIL Image or Preprocessed Tensor.
        Computes real softmax over real logits, entropy, and confidence margin.
        """
        start_time = time.time()
        self.model.eval()  # Ensure model stays in eval mode
        
        if isinstance(image_input, Image.Image):
            tensor_input = self.preprocess(image_input)
        elif isinstance(image_input, torch.Tensor):
            tensor_input = self.normalize_tensor_0_1(image_input)
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        with torch.no_grad():
            logits = self.model(tensor_input)
            probs = torch.softmax(logits, dim=1)[0]  # Softmax over class dimension dim=1

        # Calculate Shannon entropy over 1000-class softmax distribution
        # H(p) = -sum(p * log(p))
        eps = 1e-12
        entropy = -float(torch.sum(probs * torch.log(probs + eps)).item())

        top5_probs, top5_indices = torch.topk(probs, 5)
        top5_probs = top5_probs.cpu().tolist()
        top5_indices = top5_indices.cpu().tolist()

        top1_index = top5_indices[0]
        top1_prob = top5_probs[0]
        top2_prob = top5_probs[1] if len(top5_probs) > 1 else 0.0
        margin = top1_prob - top2_prob
        top1_label = get_label(top1_index)

        top5_list = []
        for idx, prob in zip(top5_indices, top5_probs):
            top5_list.append({
                "label": get_label(idx),
                "class_index": idx,
                "confidence": round(float(prob), 4),
            })

        latency_ms = round((time.time() - start_time) * 1000.0, 2)

        return {
            "top1_label": top1_label,
            "top1_class_index": top1_index,
            "top1_confidence": round(float(top1_prob), 4),
            "margin": round(float(margin), 4),
            "entropy": round(float(entropy), 4),
            "top5": top5_list,
            "latency_ms": latency_ms,
        }

    def predict_batch(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """
        Runs fast batched inference across multiple PIL images in a single GPU/CPU call.
        """
        if not images:
            return []

        start_time = time.time()
        self.model.eval()

        tensors = []
        for img in images:
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.size == (299, 299):
                t = TF.to_tensor(img)
            else:
                # Use standard transform without unsqueeze
                t = self.transforms(img)
                if t.shape[0] == 3 and (t.shape[1] != 299 or t.shape[2] != 299):
                    pass
            tensors.append(t)

        batch_tensor = torch.stack(tensors).to(self.device)
        if batch_tensor.min() >= -0.1 and batch_tensor.max() <= 1.1:
            mean = self.mean.to(self.device)
            std = self.std.to(self.device)
            batch_tensor = (batch_tensor - mean) / std

        with torch.no_grad():
            logits = self.model(batch_tensor)
            probs_batch = torch.softmax(logits, dim=1)

        batch_latency_ms = round((time.time() - start_time) * 1000.0, 2)
        single_latency = round(batch_latency_ms / len(images), 2)

        results = []
        eps = 1e-12
        for probs in probs_batch:
            entropy = -float(torch.sum(probs * torch.log(probs + eps)).item())
            top5_probs, top5_indices = torch.topk(probs, 5)
            top5_probs = top5_probs.cpu().tolist()
            top5_indices = top5_indices.cpu().tolist()

            top1_index = top5_indices[0]
            top1_prob = top5_probs[0]
            top2_prob = top5_probs[1] if len(top5_probs) > 1 else 0.0
            margin = top1_prob - top2_prob
            top1_label = get_label(top1_index)

            top5_list = []
            for idx, prob in zip(top5_indices, top5_probs):
                top5_list.append({
                    "label": get_label(idx),
                    "class_index": idx,
                    "confidence": round(float(prob), 4),
                })

            results.append({
                "top1_label": top1_label,
                "top1_class_index": top1_index,
                "top1_confidence": round(float(top1_prob), 4),
                "margin": round(float(margin), 4),
                "entropy": round(float(entropy), 4),
                "top5": top5_list,
                "latency_ms": single_latency,
            })

        return results

model_service = ModelService()
