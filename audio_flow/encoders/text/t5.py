import torch
import torch.nn as nn
from torch import Tensor
from transformers import T5EncoderModel, T5Tokenizer


class T5(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.encoder = T5EncoderModel.from_pretrained("t5-base")
        self.dim = self.encoder.config.d_model
        self.saveable = False

    def forward(self, text: list[str]) -> Tensor:
        r"""Convert text into T5 embedding.

        b: batch_size
        l: seq_len
        d: dim

        Args:
            text: list[str]

        Returns:
            latent: (b, l, d)
            mask: (b, l)
        """
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            self.encoder.eval()
            latent = self.encoder(**inputs).last_hidden_state  # (b, l, d)

        mask = inputs["attention_mask"].bool()  # (b, l)

        return latent, mask