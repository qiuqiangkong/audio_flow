import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer, T5EncoderModel


class FlanT5(nn.Module):
    def __init__(self):
        super().__init__()

        model_name = "google/flan-t5-large"
        self.tokenizer = tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = T5EncoderModel.from_pretrained(model_name)
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
            feat: (b, l, d)
            mask: (b, l)
        """
        device = next(self.parameters()).device

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            self.encoder.eval()
            feat = self.encoder(**inputs).last_hidden_state  # (b, l, d)

        mask = inputs["attention_mask"].bool()  # (b, l)

        return feat, mask