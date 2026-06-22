import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.encoders.audio.clap import CLAPEncoder
from audio_flow.encoders.text.flan_t5 import FlanT5

from audio_flow.utils import mean_pool, check_masks_type


class TTAAdapter2(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = FlanT5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)
        
        # CLAP
        self.clap = CLAPEncoder()
        self.clap_fc = nn.Linear(self.clap.dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)
        
        # Clap
        clap = self.clap(data["prompt"])  # (b, d)
        clap = self.clap_fc(clap)[:, None, :]  # (b, 1, d)

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_v, d)
        prompt = self.t5_fc(prompt)  # (b, l1, d)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)
        cross_attn_mask = prompt_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_v)
        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)

        c = task + clap  # (b, 1, d) | (b, l, d)
        seq = prompt  # (b, l, d)

        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 
