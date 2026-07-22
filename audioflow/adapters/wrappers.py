import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from einops import rearrange

from audioflow.encoders.text.char import CharEncoder
from audioflow.encoders.text.clap_text import ClapTextEncoder
from audioflow.encoders.text.flan_t5 import FlanT5
from .layers.convnext import ConvNeXt


class T5Wrapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = FlanT5()
        self.proj = nn.Linear(self.encoder.dim, dim)

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        x, mask = self.encoder(text)  # (b, l_text, d)
        x = self.proj(x)  # (b, l_text, d) 
        return x, mask


class ClapTextWrapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = ClapTextEncoder()
        self.proj = nn.Linear(self.encoder.dim, dim)

    def forward(self, text: list[str]) -> Tensor:
        x = self.encoder(text)  # (b, d)
        x = self.proj(x)[:, None, :]  # (b, 1, d)
        return x


class CharWrapper(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.encoder = CharEncoder()
        self.proj = nn.Embedding(self.encoder.vocab_size, dim)
        self.conv = ConvNeXt(dim)
        
    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        x, mask = self.encoder(text)  # (b, l_text, d)
        x = self.proj(x)  # (b, l_text, d)
        x = self.conv(x)  # (b, l_text, d)
        return x, mask


class TokenWrapper(nn.Module):
    def __init__(self, n_quantizers: int, codebook_size: int, dim: int):
        super().__init__()
        self.codebooks = nn.ModuleList(
            nn.Embedding(codebook_size, dim) for _ in range(n_quantizers)
        )
        self.proj = nn.Linear(n_quantizers * dim, dim)
        
    def forward(self, x: LongTensor) -> Tensor:
        x = torch.cat([
            self.codebooks[i](x[..., i]) for i in range(len(self.codebooks))], 
            dim=-1
        )  # (b, t, q*d)
        x = self.proj(x)  # (b, t, d)
        return x


class ConvNeXtWrapper(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, dim)
        self.conv = ConvNeXt(dim)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # (b, l, d)
        x = self.conv(x)  # (b, l, d)
        return x