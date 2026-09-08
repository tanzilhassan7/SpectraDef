import torch
import torch.nn as nn
from PIL import Image
from typing import Tuple

from app.generators.base import AdversarialGenerator

class FGSMGenerator(AdversarialGenerator):
    def generate(
        self,
        image: Image.Image,
        epsilon: float,
        iterations: int = 1,
        target_class: int = 9,
    ) -> Tuple[Image.Image, torch.Tensor]:
        orig_tensor_0_1 = self.pil_to_tensor_0_1(image)
        x_0_1 = orig_tensor_0_1.clone().detach().requires_grad_(True)
        
        norm_x = self.normalize(x_0_1)
        logits = self.model(norm_x)
        
        # Untargeted loss against initial top prediction
        initial_pred = logits.argmax(dim=1).item()
        target_tensor = torch.tensor([initial_pred], device=self.device)
        loss = nn.functional.cross_entropy(logits, target_tensor)
        
        self.model.zero_grad()
        loss.backward()
        
        # Add gradient sign step
        grad_sign = x_0_1.grad.data.sign()
        adv_tensor_0_1 = torch.clamp(x_0_1 + epsilon * grad_sign, 0.0, 1.0)
        
        diff_tensor = torch.abs(adv_tensor_0_1 - orig_tensor_0_1)
        adv_pil = self.tensor_to_pil(adv_tensor_0_1)
        return adv_pil, diff_tensor
