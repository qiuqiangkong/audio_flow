import torch
from torch import Tensor
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dim = self.model.config.vision_config.projection_dim

    def forward(self, image: Tensor) -> Tensor:
        """
        video: (b, h, w, c) uint8 or float
        return: (b, d)
        """

        inputs = self.processor(
            images=list(image),
            return_tensors="pt"
        ).to(image.device)

        with torch.no_grad():
            out = self.model.get_image_features(**inputs)  # (b, d)

        return out