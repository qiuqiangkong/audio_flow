import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor


class Patch1D(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, out_channels: int):
        super().__init__()

        self.patch_size = patch_size

        self.fc_in = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.fc_out = nn.Linear(in_features=out_channels, out_features=in_channels)

    def __call__(self, x: Tensor) -> Tensor:
        r"""Patchify 1D data.

        b: batch_size
        d: latent_dim
        t1: patches_num_t
        t2: patch_size_t, e.g., 4

        Args:
            x: (b, d, t1*t2)

        Outputs:
            x: (b, d, t1)
        """

        t2 = self.patch_size
        x = rearrange(x, 'b d (t1 t2) -> b t1 (t2 d)', t2=t2)
        x = self.fc_in(x)  # (b, t1, d)
        x = rearrange(x, 'b t1 d -> b d t1')

        return x

    def inverse(self, x: Tensor) -> Tensor:
        r"""Unpatchify 1D data.

        Args:
            x: (b, d, t1)

        Outputs:
            x: (b, d, t1*t2)
        """
        
        t2, f2 = self.patch_size
        
        x = rearrange(x, 'b d t1 -> b t1 d')  # (b, t1, d)
        x = self.fc_out(x)  # (b, t1, d)
        x = rearrange(x, 'b t1 (t2 d) -> b d (t1 t2)', t2=t2)  # (b, d, t1*t2)

        return x


class Patch2D(nn.Module):
    def __init__(self, patch_size: int, in_channels: int, out_channels: int):
        super().__init__()

        self.patch_size = patch_size

        self.fc_in = nn.Linear(in_features=in_channels, out_features=out_channels)
        self.fc_out = nn.Linear(in_features=out_channels, out_features=in_channels)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        r"""Patchify 2D data.

        b: batch_size
        d: latent_dim
        t1: patches_num_t
        t2: patch_size_t, e.g., 4
        f1: patches_num_f
        f2: patch_size_f, e.g., 4

        Args:
            x: (b, d, t1*t2, f1*t2)

        Outputs:
            x: (b, d, t1, f1)
        """

        t2, f2 = self.patch_size

        x = rearrange(x, 'b d (t1 t2) (f1 f2) -> b t1 f1 (t2 f2 d)', t2=t2, f2=f2)  # (b, t1, f1, t2*f2*d)
        x = self.fc_in(x)  # (b, t1, f1, d)
        x = rearrange(x, 'b t f d -> b d t f')  # (b, d, t1, f1)

        return x

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        r"""Unpatchify 1D data.

        Args:
            x: (b, d, t1, f1)

        Outputs:
            x: (b, d, t1*t2, f1*f2)
        """
        
        t2, f2 = self.patch_size
        
        x = rearrange(x, 'b d t f -> b t f d')  # (b, t1, f1, d)
        x = self.fc_out(x)  # (b, t1, f1, d)
        x = rearrange(x, 'b t1 f1 (t2 f2 d) -> b d (t1 t2) (f1 f2)', t2=t2, f2=f2)  # (b, d, t1*t2, f1*f2)

        return x