from __future__ import annotations

import dac
from einops import rearrange
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
import numpy as np


class DAC(nn.Module):
    def __init__(self, n_quantizers: int) -> None:
        super().__init__()

        model_path = dac.utils.download(model_type="44khz")
        self.codec = dac.DAC.load(model_path)

        self.n_quantizers = n_quantizers
        self.dim = self.codec.quantizer.codebook_size
        self.sr = self.codec.sample_rate
        self.fps = self.sr / 512
        
        self.saveable = False

    def encode(
        self, 
        audio: Tensor, 
    ) -> LongTensor:
        r"""Encode audio to discrete code.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: time_steps
        q: n_quantizers
        
        Args:
            audio: (b, c, l)

        Outputs:
            x: (b, t, d)
        """

        audio = torch.mean(audio, axis=1, keepdims=True)

        with torch.no_grad():
            self.codec.eval()
            _, codes, _, _, _ = self.codec.encode(
                audio_data=audio, 
                n_quantizers=self.n_quantizers
            )  # codes: (b, q, t), int, codebook indices

        codes = rearrange(codes, 'b q t -> b t q')
        from IPython import embed; embed(using=False); os._exit(0)
        return codes

    def decode(
        self, 
        codes: LongTensor, 
    ) -> Tensor:
        r"""Decode discrete code to audio.

        d: latent_dim

        Args:
            codes: (b, t, q)

        Returns:
            audio: (b, c, l)
        """

        codes = rearrange(codes, 'b t q -> b q t')

        with torch.no_grad():
            self.codec.eval()
            z, _, _ = self.codec.quantizer.from_codes(codes)  # (b, d, t)
            audio = self.codec.decode(z)  # (b, c, l)

        return audio

    def decode_latent_from_code(
        self, 
        codes: LongTensor
    ) -> Tensor:

        codes = rearrange(codes, 'b t q -> b q t')

        with torch.no_grad():
            self.codec.eval()
            z, _, _ = self.codec.quantizer.from_codes(codes)  # (b, d, t)

        z = rearrange(z, 'b d t -> b t d')  # (b, t, d)
        return z


    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)