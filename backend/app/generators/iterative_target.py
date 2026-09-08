import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple

from app.generators.base import AdversarialGenerator

class IterativeTargetGenerator(AdversarialGenerator):
    """
    Default demo path: Quantization-Aware Iterative Targeted Generator.
    Minimizes loss toward a target class (e.g. Ostrich, class 9) over N iterations.
    Includes discrete 8-bit uint8 quantization rounding inside optimization loop.
    """
    def generate(
        self,
        image: Image.Image,
        epsilon: float,
        iterations: int = 10,
        target_class: int = 9,
    ) -> Tuple[Image.Image, torch.Tensor]:
        orig_tensor_0_1 = self.pil_to_tensor_0_1(image)
        adv_tensor_0_1 = orig_tensor_0_1.clone().detach()

        alpha = max(0.015, epsilon / float(max(1, iterations)))
        target_tensor = torch.tensor([target_class], device=self.device)

        for _ in range(iterations):
            adv_tensor_0_1.requires_grad_(True)
            norm_x = self.normalize(adv_tensor_0_1)
            logits = self.model(norm_x)

            loss = nn.functional.cross_entropy(logits, target_tensor)

            self.model.zero_grad()
            loss.backward()

            grad_sign = adv_tensor_0_1.grad.data.sign()
            # Gradient step minimizing cross-entropy loss to target class
            adv_tensor_0_1 = adv_tensor_0_1.detach() - alpha * grad_sign

            # Clip perturbation within epsilon box of original image
            eta = torch.clamp(adv_tensor_0_1 - orig_tensor_0_1, min=-epsilon, max=epsilon)
            adv_tensor_0_1 = torch.clamp(orig_tensor_0_1 + eta, 0.0, 1.0)

            # Quantization-Aware Optimization: Round to 8-bit uint8 pixel grid (0..255)
            adv_tensor_0_1 = torch.round(adv_tensor_0_1 * 255.0) / 255.0

        diff_tensor = torch.abs(adv_tensor_0_1 - orig_tensor_0_1)
        adv_pil = self.tensor_to_pil(adv_tensor_0_1)
        return adv_pil, diff_tensor
