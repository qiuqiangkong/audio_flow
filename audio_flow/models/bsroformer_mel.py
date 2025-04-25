from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from audio_flow.models.attention import Block
from audio_flow.models.embedders import LabelEmbedder, MlpEmbedder, TimestepEmbedder
from audio_flow.models.pad import pad1d, pad2d, unpad2d
from audio_flow.models.patch import Patch1D, Patch2D
from audio_flow.models.rope import build_rope


@dataclass
class Config:
    
    name: str

    # Condition params
    y_dim: int
    c_dim: int
    ct_dim: int
    ctf_dim: int
    cx_dim: int
    in_channels: int = 1

    # Transformer params
    patch_size: tuple[int, int] = (4, 4)
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 384


class BSRoformerMel(nn.Module):
    def __init__(self, config): 
        
        super().__init__()

        self.config = config

        # Input embedders
        self.patch_x = Patch2D(
            patch_size=config.patch_size,
            in_channels=config.in_channels * np.prod(config.patch_size),
            out_channels=config.n_embd
        )

        # Time embedder
        self.t_embedder = TimestepEmbedder(config.n_embd)
        self.patch_size = config.patch_size
        self.head_dim = config.n_embd // config.n_head
        
        # One-hot label embedder
        if config.y_dim:
            self.y_embedder = LabelEmbedder(config.y_dim, config.n_embd)

        # Global embedder
        if config.c_dim:
            self.c0_embedder = MlpEmbedder(config.c_dim, config.n_embd)

        # Temporal embedder
        if config.ct_dim:
            self.patch_ct = Patch1D(
                patch_size=config.patch_size[0], 
                in_channels=config.ct_dim * config.patch_size[0], 
                out_channels=config.n_embd
            )
            self.ct_embedder = MlpEmbedder(config.n_embd, config.n_embd)

        # Temporal-frequency embedder
        if config.ctf_dim:
            self.patch_ctf = Patch2D(
                patch_size=config.patch_size,
                in_channels=config.ctf_dim * np.prod(config.patch_size),
                out_channels=config.n_embd
            )
            self.ctf_embedder = MlpEmbedder(config.n_embd, config.n_embd)

        # Cross-attention embedder
        if config.cx_dim:
            self.cx_embedder = MlpEmbedder(config.cx_dim, config.n_embd)

        # Transformer blocks
        self.t_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.f_blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))

        # Build RoPE cache
        t_rope = build_rope(seq_len=2048, head_dim=self.head_dim)
        f_rope = build_rope(seq_len=2048, head_dim=self.head_dim)
        self.register_buffer(name="t_rope", tensor=t_rope)  # shape: (t, head_dim/2, 2)
        self.register_buffer(name="f_rope", tensor=f_rope)  # shape: (t, head_dim/2, 2)
        
    def forward(
        self, 
        t: Tensor, 
        x: Tensor, 
        cond_dict: dict
    ) -> Tensor:
        """Model

        Args:
            t: (b,), random time steps between 0. and 1.
            x: (b, c, t, f)
            cond_dict: dict

        Outputs:
            output: (b, c, t, f)
        """

        # --- 1. Patchify input ---
        orig_shape = x.shape
        x, pad_t, pad_f = pad2d(x, self.patch_size)  # x: (b, d, t, f)
        x = self.patch_x(x)  # shape: (b, d, t, f)
        B, D, T, F_ = x.shape

        # --- 2. Prepare condition embeddings ---
        # Initialize conditional embedding
        emb = torch.zeros(B, self.config.n_embd, T, F_).to(x.device)

        # Repeat B times for inference
        if t.dim() == 0:
            t = t.repeat(B)

        # 2.1 Time embedder
        y = self.t_embedder(t)  # (b, d)
        emb += y[:, :, None, None]

        # 2.2 One-hot label embedding
        if self.config.y_dim:
            y = self.y_embedder(cond_dict["y"])  # (b, d)
            emb += y[:, :, None, None]
            
        # 2.3 Global embedding
        if self.config.c_dim:
            c = self.c_embedder(cond_dict["c"])  # (b, d)
            emb += c[:, :, None, None]

        # 2.4 Temporal embedding
        if self.config.ct_dim:
            assert cond_dict["ct"].shape[2] == orig_shape[2]
            ct, _ = pad1d(cond_dict["ct"], self.patch_size[0])  # x: (b, d, t, f)
            ct = self.ct_embedder(self.patch_ct(cond_dict["ct"]))  # (b, d, t)
            emb += ct[:, :, :, None]

        # 2.5 Temporal-frequency embedding
        if self.config.ctf_dim:
            assert cond_dict["ctf"].shape[2 :] == orig_shape[2 :]
            ctf, _, _ = pad2d(cond_dict["ctf"], self.patch_size)  # x: (b, d, t, f)
            ctf = self.ctf_embedder(self.patch_ctf(ctf))  # (b, d, t, f)
            emb += ctf[:, :, :, :]

        # 2.6 Cross-attention embedding
        if self.config.cx_dim:
            cx = self.cx_embedder(cond_dict["cx"])  # (b, d, t)

        # --- 2. Transformer along time and frequency axes ---
        for t_block, f_block in zip(self.t_blocks, self.f_blocks):

            emb = rearrange(emb, 'b d t f -> (b f) t d')
            x = rearrange(x, 'b d t f -> (b f) t d')
            x = t_block(x, self.t_rope, mask=None, emb=emb)  # shape: (b*f, t, d)

            emb = rearrange(emb, '(b f) t d -> (b t) f d', b=B)
            x = rearrange(x, '(b f) t d -> (b t) f d', b=B)
            x = f_block(x, self.f_rope, mask=None, emb=emb)  # shape: (b*t, f, d)

            emb = rearrange(emb, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)
            x = rearrange(x, '(b t) f d -> b d t f', b=B)  # shape: (b, d, t, f)

        # --- 3. Unpatchify ---
        x = self.patch_x.inverse(x)  # shape: (b, c, t, f)
        x = unpad2d(x, pad_t, pad_f)  # shape: (b, c, t, f)

        return x