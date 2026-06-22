import json

import torch
import torch.nn as nn
from architts_vae12_5hz import ArchiTTSVAE12Hz
from torch import Tensor
from einops import rearrange


class ArchiTTSVAE(nn.Module):

    def __init__(self):
        super().__init__()

        self.vae = ArchiTTSVAE12Hz()
    
        self.dim = self.vae.dim
        self.sr = self.vae.sample_rate
        self.fps = self.vae.fps
        self.saveable = False
        
    def encode(self, audio: Tensor) -> Tensor:
        r"""Convert text into VAE latents.

        b: batch_size
        c: channels_num
        l: audio_samples
        d: dim
        t: time_steps

        Args:
            audio: (b, 2, l)

        Returns:
            latent: (b, t, d)
        """

        with torch.no_grad():
            self.vae.eval()
            latent = self.vae.encode(audio)  # (b, t, d)

        return latent

    def decode(self, latent: Tensor) -> Tensor:
        r"""

        Args:
            latent: (b, t, d)

        Returns:
            audio: (b, c, l)
        """

        with torch.no_grad():
            self.vae.eval()
            audio = self.vae.decode(latent)

        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)