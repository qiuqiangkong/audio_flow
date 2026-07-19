import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from .rope import apply_rope
from .norm import RMSNorm


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    r"""Modulate input with scale and shift.

    Args:
        x: (b, t, d)
        shift: (b, t, d)
        scale: (b, t, d)

    Outputs:
        out: (b, t, d)
    """
    return x * (1 + scale) + shift


class Block(nn.Module):
    r"""Self attention block.

    Ref: 
        [1] https://github.com/facebookresearch/DiT/blob/main/models.py
        [2] https://huggingface.co/hpcai-tech/OpenSora-STDiT-v1-HQ-16x256x256/blob/main/layers.py
    """
    def __init__(self, dim, num_heads) -> None:
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.self_attn = SelfAttention(dim, num_heads)
        self.cross_attn = CrossAttention(dim, num_heads)
        
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), 
            nn.GELU(approximate='tanh'),
            nn.Linear(dim * 4, dim)
        )

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim)
        )

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        seq: Tensor,
        rope_q: Tensor,
        rope_k: Tensor,
        self_attn_mask: Tensor,
        cross_attn_mask: Tensor,
    ) -> torch.Tensor:
        r"""Self attention block.

        Args:
            x: (b, l, d)
            rope: (t, head_dim/2, 2)
            mask: None | (1, 1, l, l)
            emb: (b, l, d)

        Outputs:
            out: (b, l, d)
        """
        
        e = self.modulation(c).chunk(6, dim=2)

        # Self-attention
        h = modulate(self.norm1(x), e[0], e[1])
        x = x + e[2] * self.self_attn(h, rope_q, self_attn_mask)

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x), seq, rope_q, rope_k, cross_attn_mask)

        # FFN
        h = modulate(self.norm3(x), e[3], e[4])
        x = x + e[5] * self.ffn(h)

        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads) -> None:
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.qkv_linear = nn.Linear(dim, 3 * dim)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: Tensor,
        rope: Tensor,
        mask: Tensor | None
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: n_head
        h: head_dim

        Args:
            x: (b, l, d)
            rope: (l, head_dim/2, 2)
            mask: (1, 1)

        Outputs:
            x: (b, l, d)
        """

        # Calculate query, key, values
        q, k, v = self.qkv_linear(x).chunk(chunks=3, dim=2)  # shapes: (b, l, d)
        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)

        # Apply RoPE
        q = apply_rope(q, rope)  # (b, l, n, h)
        k = apply_rope(k, rope)  # (b, l, n, h)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=mask, 
            dropout_p=0.0
        )  # (b, n, l, h)
        
        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.norm_q = RMSNorm(dim)
        self.norm_k = RMSNorm(dim)

        self.proj = nn.Linear(dim, dim)

    def forward(
        self, 
        x: Tensor, 
        seq: Tensor,
        rope_q: Tensor,
        rope_k: Tensor,
        mask: Tensor
    ) -> Tensor:
        r"""Causal self attention.

        b: batch_size
        l: seq_len
        d: latent_dim
        n: heads_num
        h: head_dim
        k: rope_dim

        Args:
            x: (b, l, d)
            rope: (l, h/2, 2)
            pos: (l, k)
            mask: (1, 1, )

        Outputs:
            x: (b, l, d)
        """
        B, L, D = x.shape

        # Calculate query, key, values
        q = self.q_linear(x)  # shapes: (b, lx, d)
        k, v = self.kv_linear(seq).chunk(chunks=2, dim=2)  # shapes: (b, lc, d)

        q = rearrange(self.norm_q(q), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        k = rearrange(self.norm_k(k), 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        v = rearrange(v, 'b l (n h) -> b l n h', h=self.head_dim)  # (b, l, n, h)
        
        # Apply RoPE
        q = apply_rope(q, rope_q)  # (b, l, n, h)
        k = apply_rope(k, rope_k)  # (b, l, n, h)

        # from IPython import embed; embed(using=False); os._exit(0)

        # Efficient attention using Flash Attention CUDA kernels
        x = F.scaled_dot_product_attention(
            query=rearrange(q, 'b l n h -> b n l h'), 
            key=rearrange(k, 'b l n h -> b n l h'), 
            value=rearrange(v, 'b l n h -> b n l h'), 
            attn_mask=mask, 
            dropout_p=0.0
        )  # (b, n, l, h)

        x = rearrange(x, 'b n l h -> b l (n h)')
        x = self.proj(x)  # (b, l, d)
        
        return x