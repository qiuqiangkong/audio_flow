import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.encoders.text.t5 import T5

from audio_flow.utils import mean_pool, check_masks_type


class Midi2AudioAdapter(nn.Module): 
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

        # Latent
        input_latent = self.latent_fc(data["input_latent"].float())  # (b, l, d)

        # Build masks
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Build cross attention mask
        L_tar = target_mask.shape[1]
        L_in = input_latent.shape[1]
        assert L_in % L_tar == 0
        r = int(L_in // L_tar)
        cross_attn_mask = torch.zeros((L_tar, L_in), dtype=torch.bool, device=mask.device)

        for i in range(L_tar):
            cross_attn_mask[i, i*r : (i+1)*r] = True

        cross_attn_mask = cross_attn_mask[None, None, :, :]  # (1, 1, l_q, l_v)

        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)
        
        seq = input_latent  # (b, l, d)
        c = task  # (b, 1, d) | (b, l, d)
        
        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 
