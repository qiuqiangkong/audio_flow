from __future__ import annotations

import dac
import torch
import torch.nn as nn
from torch import LongTensor, Tensor
from einops import rearrange


class DAC(nn.Module):
    def __init__(self, n_quantizers: int = 2) -> None:
        super().__init__()

        model_path = dac.utils.download(model_type="44khz")
        self.model = dac.DAC.load(model_path)
        self.sr = self.model.sample_rate
        self.fps = self.sr / 2 / 4 / 8 / 8
        self.n_quantizers = n_quantizers
        self.saveable = False

    def encode(self, audio: Tensor) -> Tensor:
        r"""Convert audio into discrete code.

        b: batch_size
        c: channels_num
        l: audio_samples
        q: n_quantizers
        t: time_steps
        
        Args:
            audio: (b, c, l)

        Outputs:
            x: (b, t, q)
        """

        assert audio.shape[1] == 1

        with torch.no_grad():
            self.model.eval()
            _, code, _, _, _ = self.model.encode(
                audio_data=audio, 
                n_quantizers=self.n_quantizers
            )  # codes: (b, q, t), integer, codebook indices

            # latent, _, _ = self.model.quantizer.from_codes(code[:, 0 : self.n_quantizers, :])
        
        code = rearrange(code, 'b q t -> b t q')

        return code

    def decode(
        self, 
        code: LongTensor, 
    ) -> Tensor:
        r"""Decode discrete code to audio.

        d: latent_dim

        Args:
            codes: (b, t, q)

        Returns:
            audio: (b, c, l)
        """

        code = rearrange(code, 'b t q -> b q t')

        with torch.no_grad():
            self.model.eval()
            z, _, _ = self.model.quantizer.from_codes(code)  # (b, d, t)
            audio = self.model.decode(z)  # (b, c, l)

        return audio

    def code_to_latent(self, codes: LongTensor) -> Tensor:
        with torch.no_grad():
            self.model.eval()
            latent, _, _ = self.model.quantizer.from_codes(codes)  # (b, d, t)

        return latent

    def __call__(self, audio: Tensor) -> Tensor:
        return self.encode(audio)