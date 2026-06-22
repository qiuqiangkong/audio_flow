import json

import torch
import torchaudio
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch import Tensor
from einops import rearrange

from mmaudio.model.utils.features_utils import FeaturesUtils


class MMAudioVAE(nn.Module):

    def __init__(self):
        super().__init__()

        # 16kHz VAE
        vae_path = hf_hub_download(
            repo_id="hkchengrex/MMAudio",
            filename="ext_weights/v1-16.pth",
        )

        # 16kHz BigVGAN
        bigvgan_path = hf_hub_download(
            repo_id="hkchengrex/MMAudio",
            filename="ext_weights/best_netG.pt",
        )

        self.mel_vae = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            bigvgan_vocoder_ckpt=bigvgan_path,
            synchformer_ckpt=None,
            enable_conditions=False,
            mode="16k",
            need_vae_encoder=True,
        )

        self.dim = 20
        self.sr = 16000
        self.fps = 31.25
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
        B, C, L = audio.shape
        x = audio.mean(dim=1)

        with torch.no_grad():
            self.mel_vae.eval()
            p = self.mel_vae.encode_audio(x)
            z = p.sample()  # (b, d, t)

        latent = rearrange(z, 'b d t -> b t d')
        return latent

    def decode(self, latent: Tensor) -> Tensor:
        r"""

        b: batch_size
        t: n_frames
        d: dim
        c: audio_channels
        l: audio_samples
        f: mel_bins
        t': mel_frames

        Args:
            latent: (b, t, d)

        Returns:
            audio: (b, c=1, l)
        """

        with torch.no_grad():
            self.mel_vae.eval()
            mel = self.mel_vae.decode(latent)  # (b, f, t')
            audio = self.mel_vae.vocode(mel)  # (b, c=1, l)

        return audio

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)