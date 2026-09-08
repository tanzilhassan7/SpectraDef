from abc import ABC, abstractmethod
import io
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple, Dict, Any

from app.services.model_service import model_service

class AdversarialGenerator(ABC):
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self):
        self.model_service = model_service
        self.model = model_service.model
        self.device = model_service.device

    def normalize(self, tensor_0_1: torch.Tensor) -> torch.Tensor:
        """Converts [0, 1] tensor (B, 3, H, W) to ImageNet normalized tensor."""
        if tensor_0_1.dim() == 3:
            tensor_0_1 = tensor_0_1.unsqueeze(0)
        mean = self.IMAGENET_MEAN.to(tensor_0_1.device)
        std = self.IMAGENET_STD.to(tensor_0_1.device)
        return (tensor_0_1 - mean) / std

    def denormalize(self, tensor_norm: torch.Tensor) -> torch.Tensor:
        """Converts ImageNet normalized tensor to [0, 1] tensor, clamped."""
        if tensor_norm.dim() == 3:
            tensor_norm = tensor_norm.unsqueeze(0)
        mean = self.IMAGENET_MEAN.to(tensor_norm.device)
        std = self.IMAGENET_STD.to(tensor_norm.device)
        tensor_0_1 = tensor_norm * std + mean
        return torch.clamp(tensor_0_1, 0.0, 1.0)

    def tensor_to_pil(self, tensor_0_1: torch.Tensor) -> Image.Image:
        """Converts [0, 1] float tensor (1, 3, H, W) or (3, H, W) to PIL Image."""
        if tensor_0_1.dim() == 4:
            tensor_0_1 = tensor_0_1.squeeze(0)
        tensor_0_1 = torch.clamp(tensor_0_1.detach().cpu(), 0.0, 1.0)
        return TF.to_pil_image(tensor_0_1)

    def pil_to_tensor_0_1(self, image: Image.Image) -> torch.Tensor:
        """Converts PIL Image to [0, 1] float tensor (1, 3, 299, 299)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        if image.size != (299, 299):
            image = image.resize((299, 299), Image.BILINEAR)
        tensor = TF.to_tensor(image)  # Shape (3, 299, 299)
        return tensor.unsqueeze(0).to(self.device)

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        epsilon: float,
        iterations: int,
        target_class: int,
    ) -> Tuple[Image.Image, torch.Tensor]:
        """
        Generates adversarial image.
        Returns tuple of (adversarial_pil_image, raw_difference_tensor_0_1).
        """
        pass
