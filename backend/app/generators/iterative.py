import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple

from app.generators.base import AdversarialGenerator

class IterativeGenerator(AdversarialGenerator):
    """Basic Iterative Method (BIM) untargeted generator."""
    def generate(
        self,
        image: Image.Image,
        epsilon: float,
        iterations: int = 10,
        target_class: int = 9,
    ) -> Tuple[Image.Image, torch.Tensor]:
        orig_tensor_0_1 = self.pil_to_tensor_0_1(image)
        adv_tensor_0_1 = orig_tensor_0_1.clone().detach()
        alpha = max(0.001, epsilon / max(1, iterations))

        for _ in range(iterations):
            adv_tensor_0_1.requires_grad_(True)
            norm_x = self.normalize(adv_tensor_0_1)
            logits = self.model(norm_x)

            initial_pred = logits.argmax(dim=1).item()
            target_tensor = torch.tensor([initial_pred], device=self.device)
            loss = nn.functional.cross_entropy(logits, target_tensor)

            self.model.zero_grad()
            loss.backward()

            grad_sign = adv_tensor_0_1.grad.data.sign()
            adv_tensor_0_1 = adv_tensor_0_1.detach() + alpha * grad_sign
            
            # Clip within epsilon ball of original image and valid image range [0, 1]
            eta = torch.clamp(adv_tensor_0_1 - orig_tensor_0_1, min=-epsilon, max=epsilon)
            adv_tensor_0_1 = torch.clamp(orig_tensor_0_1 + eta, 0.0, 1.0)

        diff_tensor = torch.abs(adv_tensor_0_1 - orig_tensor_0_1)
        adv_pil = self.tensor_to_pil(adv_tensor_0_1)
        return adv_pil, diff_tensor
