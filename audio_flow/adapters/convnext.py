import torch.nn as nn
from einops import rearrange


class ConvNeXt(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.blocks = nn.ModuleList([ConvNeXtBlock(dim) for _ in range(3)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        r"""

        Args:
            x: (b, l, d)

        Returns:
            out: (b, l, d)
        """

        residual = x
        x = rearrange(x, 'b l d -> b d l')
        x = self.dwconv(x)
        x = rearrange(x, 'b d l -> b l d')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + residual