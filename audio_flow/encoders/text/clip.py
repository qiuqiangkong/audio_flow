import torch
from torch import Tensor
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

        self.model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32"
        ).to(device)

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        
        self.dim = self.model.config.text_config.projection_dim

    def forward(self, text: list) -> Tensor:
        """
        text: list
        return: (b, d)
        """

        inputs = self.processor(
            text=text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)  # (b, d)

        return out