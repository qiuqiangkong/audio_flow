'''
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class CLIPVideoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.dim = self.model.config.vision_config.projection_dim

    def forward(self, video: list[str]) -> Tensor:
        r"""Convert text into CLAP embedding.

        b: batch_size
        d: dim
        t: frames
        h: height
        w: weight
        c: channels

        Args:
            video: (b, t, h, w, c)

        Returns:
            latent: (b, d)
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            self.encoder.eval()
            latent = self.encoder.get_text_features(**inputs)  # (b, d)

        return latent
'''

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

        self.dim = self.model.config.vision_config.projection_dim

    def forward(self, image: Tensor) -> Tensor:
        """
        video: (B, H, W, C) uint8 or float
        return: (B, D)
        """

        inputs = self.processor(
            images=list(image),
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.get_image_features(**inputs)  # (B*T, D)

        return out