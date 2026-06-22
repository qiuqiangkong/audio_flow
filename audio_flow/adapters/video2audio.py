import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.adapters.mss import MSSAdapter
from audio_flow.encoders.text.clip import CLIPEncoder
from audio_flow.encoders.text.t5 import T5
from audio_flow.utils import mean_pool, check_masks_type


# Video2AudioAdapter = MSSAdapter


'''
class Video2AudioAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        self.clip = CLIPEncoder()
        self.clip_fc = nn.Linear(self.clip.dim, dim)

        # Latent
        self.latent_fc = nn.Linear(in_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)

        # Latent
        prompt = self.clip(data["prompt"])  # (b, d)
        prompt = self.clip_fc(prompt)
        input_latent = self.latent_fc(data["input_latent"])  # (b, l, d)

        # Target mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Prompt mask
        prompt_cross_attn_mask = target_mask[:, None, :, None]  # (b, 1, l_q, l_v)

        input_cross_attn_mask = torch.eye(
            n=input_latent.shape[1], 
            dtype=torch.bool, 
            device=mask.device
        )[None, None, :, :].expand(task.shape[0], -1, -1, -1)  # (b, 1, l_q, l_v)

        cross_attn_mask = torch.cat([prompt_cross_attn_mask, input_cross_attn_mask], dim=3)

        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)
        
        seq = torch.cat([prompt[:, None, :], input_latent], dim=1)  # (b, l, d)
        c = task  # (b, 1, d) | (b, l, d)
        
        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 
'''


class Video2AudioAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        self.clip = CLIPEncoder()
        self.clip_fc = nn.Linear(self.clip.dim, dim)

        # Latent
        self.latent_fc = nn.Linear(in_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)

        # Latent
        prompt = self.clip(data["prompt"])  # (b, d)
        prompt = self.clip_fc(prompt)
        input_latent = self.latent_fc(data["input_latent"])  # (b, l, d)

        # Target mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Prompt mask
        prompt_cross_attn_mask = target_mask[:, None, :, None]  # (b, 1, l_q, l_v)

        input_cross_attn_mask = torch.eye(
            n=input_latent.shape[1], 
            dtype=torch.bool, 
            device=mask.device
        )[None, None, :, :].expand(task.shape[0], -1, -1, -1)  # (b, 1, l_q, l_v)

        # cross_attn_mask = torch.cat([prompt_cross_attn_mask, input_cross_attn_mask], dim=3)
        cross_attn_mask = prompt_cross_attn_mask

        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)
        
        # seq = torch.cat([prompt[:, None, :], input_latent], dim=1)  # (b, l, d)
        seq = prompt[:, None, :]
        c = task  # (b, 1, d) | (b, l, d)
        
        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 


class Video2AudioAdapterCfg(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = T5()
        self.t5_fc = nn.Linear(self.t5.dim, dim)

        self.clip = CLIPEncoder()
        self.clip_fc = nn.Linear(self.clip.dim, dim)

        # Latent
        self.latent_fc = nn.Linear(in_dim, dim)

        self.null_c = nn.Parameter(torch.randn(self.clip.dim))  # (d,)

    def forward(self, data: dict) -> Tensor:

        # Task
        task, mask = self.t5(data["task"])  # (b, l, d)
        task = self.t5_fc(task)  # (b, l, d)
        task = mean_pool(task, mask, keepdims=True)  # (b, 1, d)

        # Latent
        prompt = self.clip(data["prompt"])  # (b, d)
        prompt = self.clip_fc(prompt)
        input_latent = self.latent_fc(data["input_latent"])  # (b, l, d)

        # Target mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Prompt mask
        prompt_cross_attn_mask = target_mask[:, None, :, None]  # (b, 1, l_q, l_v)

        input_cross_attn_mask = torch.eye(
            n=input_latent.shape[1], 
            dtype=torch.bool, 
            device=mask.device
        )[None, None, :, :].expand(task.shape[0], -1, -1, -1)  # (b, 1, l_q, l_v)

        # cross_attn_mask = torch.cat([prompt_cross_attn_mask, input_cross_attn_mask], dim=3)
        cross_attn_mask = prompt_cross_attn_mask

        assert check_masks_type([self_attn_mask, cross_attn_mask], torch.bool)
        
        # seq = torch.cat([prompt[:, None, :], input_latent], dim=1)  # (b, l, d)
        seq = prompt[:, None, :]
        c = task  # (b, 1, d) | (b, l, d)
        
        controls = {
            "c": c,
            "seq": seq,
            "self_attn_mask": self_attn_mask,
            "cross_attn_mask": cross_attn_mask
        }

        return controls 


class Video2AudioMaeAdapter(nn.Module): 
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
        input_latent = self.latent_fc(data["input_latent"])  # (b, l, d)

        # Build mask
        target_mask = data["target_mask"]  # (b, l_q)
        self_attn_mask = target_mask[:, None, None, :] * target_mask[:, None, :, None]  # (b, 1, l_q, l_q)

        # Build cross attention mask
        L_tar = target_mask.shape[1]
        L_in = input_latent.shape[1]
        assert L_tar % L_in == 0
        r = int(L_tar // L_in)
        cross_attn_mask = torch.zeros((L_tar, L_in), dtype=torch.bool, device=mask.device)

        for i in range(L_in):
            cross_attn_mask[i*r : (i+1)*r, i] = True

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