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
    def __init__(
        self, 
        dim: int, 
        freq_size: int = 256,
        scale: float = 1.  # Use 100. for flow matching
    ):
        super().__init__()

        self.freq_size = freq_size
        self.scale = scale

        self.mlp = nn.Sequential(
            nn.Linear(freq_size, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
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
        args = self.scale * t[:, None] * freqs[None, :]  # (b, dim/2)
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