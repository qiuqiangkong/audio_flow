import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class CLAPEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("laion/clap-htsat-unfused")
        self.encoder = AutoModel.from_pretrained("laion/clap-htsat-unfused")
        self.dim = self.encoder.config.audio_config.projection_dim
        self.saveable = False
        
    def forward(self, text: list[str]) -> Tensor:
        r"""Convert text into CLAP embedding.

        b: batch_size
        d: dim

        Args:
            text: list[str]

        Returns:
            latent: (b, d)
        """
        device = next(self.parameters()).device
        inputs = self.tokenizer(text, padding=True, return_tensors="pt").to(device)

        with torch.no_grad():
            self.encoder.eval()
            latent = self.encoder.get_text_features(**inputs)  # (b, d)
            # latent = self.encoder.get_text_features(**inputs).pooler_output  # (b, d)
        
        return latent
