import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.encoders.text.t5 import T5
from audio_flow.encoders.text.char import CharEncoder
from audio_flow.adapters.convnext import ConvNeXt

from audio_flow.utils import mean_pool, check_masks_type


class TTSAdapter(nn.Module): 
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # Character encoder
        self.char_encoder = CharEncoder()
        self.char_embedder = nn.Embedding(self.char_encoder.vocab_size, dim)
        self.char_conv = ConvNeXt(dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)
        
        # Prompt
        prompt, prompt_mask = self.char_encoder(data["prompt"])  # (b, l_text, d), (b, l_text)
        prompt = self.char_embedder(prompt)  # (b, l_text, d)
        prompt = self.char_conv(prompt)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)
        cross_attn_mask = prompt_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_v)
        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)

        c = task  # (b, 1, d) | (b, l, d)
        seq = prompt  # (b, l, d)

        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 
