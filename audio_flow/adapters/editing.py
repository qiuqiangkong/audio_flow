import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.encoders.text.t5 import T5

from audio_flow.utils import mean_pool, check_masks_type


class EditingAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        # Latent
        self.latent_fc = nn.Linear(in_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_v, d)
        prompt = self.t5_fc(prompt)  # (b, l1, d)

        # Latent
        input_latent = self.latent_fc(data["input_latent"])  # (b, l, d)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Cross attention mask
        prompt_cross_attn_mask = prompt_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_v)

        input_cross_attn_mask = torch.eye(
            n=input_latent.shape[1], 
            dtype=torch.bool, 
            device=mask.device
        )[None, None, :, :].expand(task.shape[0], -1, -1, -1)  # (b, 1, l_q, l_v)

        cross_attn_mask = torch.cat([prompt_cross_attn_mask, input_cross_attn_mask], dim=3)

        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)
        
        seq = torch.cat([prompt, input_latent], dim=1)  # (b, l, d)
        c = task  # (b, 1, d) | (b, l, d)

        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 
