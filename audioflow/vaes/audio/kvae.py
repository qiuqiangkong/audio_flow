import json

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from kvae_audio import KVAEAudio
from torch import Tensor
from einops import rearrange


class KVAE(nn.Module):

    def __init__(self):
        super().__init__()

        ckpt_path = hf_hub_download(
            repo_id="kandinskylab/KVAE-Audio",
            filename="kvae-audio.pt",
        )
        self.vae = KVAEAudio.load(ckpt_path, map_location="cpu")

        self.dim = self.vae.codebook_dim
        self.sr = self.vae.sample_rate
        self.fps = 50
        self.saveable = False
        
    def encode(self, audio: Tensor) -> Tensor:
        r"""Convert text into VAE latents.

        b: batch_size
        c: n_channels
        l: audio_samples
        d: dim
        t: time_steps

        Args:
            audio: (b, c, l)

        Returns:
            latent: (b, t, d)
        """

        # Encode each audio channel independently (mono VAE)
        x = rearrange(audio, 'b c l -> (b c) 1 l')  # (b*c, l)

        with torch.no_grad():
            self.vae.eval()
            z, _, _, _ = self.vae.encode(x, sample_rate=self.sr)
            
        latent = rearrange(z, '(b c) d t -> b t (c d)', c=audio.shape[1])  # (b, t, c*d)
        return latent

    def decode(self, latent: Tensor) -> Tensor:
        r"""

        Args:
            latent: (b, t, d)

        Returns:
            audio: (b, c, l)
        """

        # Decode each audio channel independently (mono VAE)
        x = rearrange(latent, 'b t (c d) -> (b c) d t', d=self.dim)

        with torch.no_grad():
            self.vae.eval()
            audio = self.vae.decode(x)

        audio = rearrange(audio, '(b c) 1 l -> b c l', b=latent.shape[0])
        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)