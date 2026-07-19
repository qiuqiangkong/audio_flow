from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audioflow.layers.attention import Block
from audioflow.layers.embedders import TimestepEmbedder
from audioflow.layers.rope import build_rope


class Transformer(nn.Module):
    def __init__(
        self,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        **kwargs
    ):
        super().__init__()

        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=1000.)

        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        self.head_dim = dim // num_heads
        
    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
        **kwargs,
    ) -> Tensor:
        r"""DiT.

        b: batch_size
        d: dim
        t: time_step

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, t, d)

        Outputs:
            out: (b, t, d)
        """

        device = x.device

        c = controls["c"]  # (b, 1, d) | (b, t, d)
        seq = controls["seq"]  # (b, l, d)
        self_attn_mask = controls["self_attn_mask"]  # (b, t, d)
        cross_attn_mask = controls["cross_attn_mask"]  # (b, l, d)
        
        in_pos = controls.get("input_pos", torch.arange(seq.shape[1]).to(device))
        tgt_pos = controls.get("target_pos", torch.arange(x.shape[1]).to(device))
        rope_q = build_rope_from_pos(self.head_dim, tgt_pos)
        rope_k = build_rope_from_pos(self.head_dim, in_pos)

        # Time embedding
        if t.dim() == 0:
            t = t.repeat(x.shape[0])  # (b,)

        c = c + self.t_embedder(t)[:, None, :]  # (b, 1, d) | (b, t, d)

        # Transformer
        for block in self.blocks:
            x = block(x, c, seq, rope_q, rope_k, self_attn_mask, cross_attn_mask)
        
        return x


def build_rope_from_pos(
    head_dim: int,
    pos: Tensor | list[Tensor]
) -> Tensor:

    if isinstance(pos, Tensor):
        return build_rope(head_dim, pos).to(pos.device)

    elif isinstance(pos, list):
        return torch.cat([build_rope_from_pos(head_dim // len(pos), p) for p in pos], dim=0)

    else:
        raise TypeError(pos)