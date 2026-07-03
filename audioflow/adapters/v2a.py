import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audioflow.encoders.text.flan_t5 import FlanT5
from audioflow.encoders.text.clap_text import CLAPTextEncoder

from audioflow.utils.torch import stretched_eye, check_masks_type
from audioflow.utils.xml import batch_parse_xml


class V2AAdapter(nn.Module): 
    def __init__(self, in_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = FlanT5()
        self.t5_proj = nn.Linear(self.t5.dim, dim)

        # CLAP Text
        self.clap_text = CLAPTextEncoder()
        self.clap_text_proj = nn.Linear(self.clap_text.dim, dim)

        # Video
        self.video_proj = nn.Linear(in_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_text, d)
        prompt = self.t5_proj(prompt)  # (b, l_text, d)
        B = prompt.shape[0]

        text = batch_parse_xml(data["prompt"])

        # Clap
        text = batch_parse_xml(data["prompt"])
        clap = self.clap_text(text)  # (b, d)
        clap = self.clap_text_proj(clap)[:, None, :]  # (b, 1, d)

        # Video
        video = data["input_feature"]  # (b, l_video, d)
        video = self.video_proj(video)  # (b, l_video, d)

        # Sequence
        seq = torch.cat([video, prompt], dim=1)

        # Self attention mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]  # (b, l_q, l_q)

        # Cross attention mask
        cm1 = stretched_eye(
            n=data["target_latent"].shape[1],
            m=data["input_feature"].shape[1],
            device=video.device
        )[None, :, :].expand(B, -1, -1)  # (b, l_q, l_v)
        
        cm2 = tgt_mask[:, :, None] * prompt_mask[:, None, :]  # (b, l_q, l_v)
        cross_mask = torch.cat([cm1, cm2], dim=-1)

        # Check
        assert check_masks_type([self_mask, cross_mask], torch.bool)

        controls = {
            "c": clap,  # (b, 1, d)
            "seq": seq,  # (b, l, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1)  # (b, 1, l_q, l_v)
        }

        return controls 


'''
class V2AAdapter(nn.Module): 
    def __init__(self, video_dim: int, dim: int, **kwargs):
        super().__init__()

        # T5
        self.t5 = FlanT5()
        self.t5_proj = nn.Linear(self.t5.dim, dim)

        # CLAP Text
        self.clap_text = CLAPTextEncoder()
        self.clap_text_proj = nn.Linear(self.clap_text.dim, dim)

        # Video
        self.video_proj = nn.Linear(video_dim, dim)

    def forward(self, data: dict) -> Tensor:

        # Prompt
        prompt, prompt_mask = self.t5(data["prompt"])  # (b, l_text, d)
        prompt = self.t5_proj(prompt)  # (b, l_text, d)
        B = prompt.shape[0]

        text = batch_parse_xml(data["prompt"])

        # Clap
        text = batch_parse_xml(data["prompt"])
        clap = self.clap_text(text)  # (b, d)
        clap = self.clap_text_proj(clap)[:, None, :]  # (b, 1, d)

        # Video
        video = data["input_feature"]  # (b, l_video, d)
        video = self.video_proj(video)  # (b, l_video, d)

        # Sequence
        seq = torch.cat([video, prompt], dim=1)

        # Self attention mask
        tgt_mask = data["target_mask"]  # (b, l_q)
        self_mask = tgt_mask[:, :, None] * tgt_mask[:, None, :]  # (b, l_q, l_q)

        # Cross attention mask
        cm1 = stretched_eye(
            n=data["target_latent"].shape[1],
            m=data["input_feature"].shape[1],
            device=video.device
        )[None, :, :].expand(B, -1, -1)  # (b, l_q, l_v)
        
        cm2 = tgt_mask[:, :, None] * prompt_mask[:, None, :]  # (b, l_q, l_v)
        cross_mask = torch.cat([cm1, cm2], dim=-1)

        # Check
        assert check_masks_type([self_mask, cross_mask], torch.bool)

        controls = {
            "c": clap,  # (b, 1, d)
            "seq": seq,  # (b, l, d)
            "self_attn_mask": self_mask.unsqueeze(1),  # (b, 1, l_q, l_q)
            "cross_attn_mask": cross_mask.unsqueeze(1)  # (b, 1, l_q, l_v)
        }

        return controls 
'''