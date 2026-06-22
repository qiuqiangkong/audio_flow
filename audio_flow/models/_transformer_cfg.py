from __future__ import annotations

import torch.nn as nn
from einops import rearrange
from torch import Tensor
import torch

from audio_flow.models.attention import Block
from audio_flow.models.embedders import TimestepEmbedder
from audio_flow.models.rope import RoPE


class Transformer(nn.Module):
    def __init__(
        self,
        in_dim=16,
        dim=384,
        mlp_ratio=4.0,
        num_layers=12,
        num_heads=12,
        rope_len=8192,
        cfg_drop=0.5,
        **kwargs
    ):
        
        super().__init__()

        self.cfg_drop = cfg_drop
        self.fc_in = nn.Linear(in_dim, dim)
        self.fc_out = nn.Linear(dim, in_dim)
        
        # Time embedder
        self.t_embedder = TimestepEmbedder(dim=dim, freq_size=256, scale=100.)

        self.blocks = nn.ModuleList(Block(dim, num_heads) for _ in range(num_layers))

        head_dim = dim // num_heads
        self.rope = RoPE(head_dim, max_len=rope_len)

        self.null_c = nn.Parameter(torch.randn(dim))
        self.null_seq = nn.Parameter(torch.randn((1, dim)))

    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
        # cfg_scale=None,
        **kargs,
    ) -> Tensor:
        
        if self.cfg_drop == 0.:
            return self.forward_no_cfg(t, x, controls)

        else:
            if self.training:
                return self.forward_with_drop(t, x, controls)
            else:
                return self.forward_with_cfg(t, x, controls, kargs["cfg_scale"])

    def forward_no_cfg(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
    ) -> Tensor:

        r"""DiT.

        b: batch_size
        d: dim
        t: time_step

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        # Time embedding
        if t.dim() == 0:
            t = t.repeat(x.shape[0])  # (b,)

        c = controls["c"]  # (b, d)
        seq = controls["seq"]  # (b, l, d)
        seq_mask = controls["seq_mask"]  # (b, 1, l_audio, l_text)
        x_mask = controls["x_mask"]

        c = c + self.t_embedder(t)
        c = c[:, None, :]  # (b, 1, d)

        # Transformer
        x = self.fc_in(x)

        for block in self.blocks:
            x = block(x, c, seq, self.rope, x_mask, seq_mask)
        
        x = self.fc_out(x)

        return x

    def forward_with_drop(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
    ) -> Tensor:

        device = x.device
        B = x.shape[0]
        ids = torch.rand(B, device=device) < self.cfg_drop
        controls["c"][ids] = self.null_c  # (b, d)
        controls["seq"][ids] = self.null_seq  # (b, l, d)
        controls["seq_mask"][ids] = True  # (b, 1, l_audio, l_text)
        return self.forward_no_cfg(t, x, controls)

    def forward_with_cfg(
        self, 
        t: Tensor, 
        x: Tensor, 
        controls: dict,
        cfg_scale
    ) -> Tensor:
        r"""DiT.

        b: batch_size
        d: dim
        t: time_step

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, l, d)

        Outputs:
            out: (b, l, d)
        """

        device = x.device
        B = x.shape[0]

        x = torch.cat([x, x], dim=0)
        for key in ["c", "seq", "seq_mask"]:
            controls[key] = torch.cat([controls[key], controls[key]], dim=0)
        
        controls["c"][B:] = self.null_c
        controls["seq"][B:] = self.null_seq
        controls["seq_mask"][B:] = True

        from IPython import embed; embed(using=False); os._exit(0)
        out = self.forward_no_cfg(t, x, controls)  # (Bx2, ...)

        cond_eps, uncond_eps = torch.split(out, B, dim=0)
        out = uncond_eps + cfg_scale * (cond_eps - uncond_eps)

        return out

    

# def repeat(x, n):
#     return x.unsqueeze(0).expand(n, *x.shape)
