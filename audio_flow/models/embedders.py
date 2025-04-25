import math

import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class TimestepEmbedder(nn.Module):
    r"""Time step embedder.
    
    References:
    [1] https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/unet/nn.py
    [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """
    def __init__(self, out_channels: int, freq_size: int = 256):
        super().__init__()

        self.freq_size = freq_size

        self.mlp = nn.Sequential(
            nn.Linear(freq_size, out_channels, bias=True),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def timestep_embedding(self, t: Tensor, max_period=10000) -> Tensor:
        r"""

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            embedding: (b, d)
        """
        
        half = self.freq_size // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half) / half).to(t.device)  # (b,)
        args = t[:, None] * freqs[None, :]  # (b, dim/2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (b, dim)
        
        return embedding

    def forward(self, t: Tensor) -> Tensor:
        r"""Calculate time embedding.

        Args:
            t: (b,), between 0. and 1.

        Outputs:
            out: (b, d)
        """

        t = self.timestep_embedding(t)
        t = self.mlp(t)
        
        return t


class LabelEmbedder(nn.Module):

    def __init__(self, classes_num: int, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Embedding(classes_num, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: LongTensor) -> Tensor:
        r"""Calculate label embedding.

        Args:
            x: (b,), LongTensor

        Outputs:
            out: (b, d)
        """
        
        return self.mlp(x)


class MlpEmbedder(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        r"""Calculate MLP embedding.

        Args:
            x: (b, d, ...)

        Outputs:
            out: (b, d, ...)
        """
        
        x = x.transpose(1, -1)  # (b, ..., d)
        x = self.mlp(x)  # (b, ..., d)
        x = x.transpose(1, -1)  # (b, d, ...)
        return x