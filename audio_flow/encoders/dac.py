from __future__ import annotations

import dac
import torch
import torch.nn as nn
from torch import LongTensor, Tensor


class DAC(nn.Module):
    def __init__(self, sr: float, n_quantizers: int) -> None:
        super().__init__()

        assert sr == 44100

        model_path = dac.utils.download(model_type="44khz")
        self.codec = dac.DAC.load(model_path)
        self.n_quantizers = n_quantizers

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
            x: (b, q, t)
        """

        assert audio.shape[1] == 1

        with torch.no_grad():
            self.codec.eval()
            _, codes, _, _, _ = self.codec.encode(
                audio_data=audio, 
                n_quantizers=self.n_quantizers
            )  # codes: (b, q, t), integer, codebook indices

        if self.n_quantizers:
            codes = codes[:, 0 : self.n_quantizers, :]  # (b, q, t)

        return codes

    def decode(
        self, 
        codes: LongTensor, 
    ) -> Tensor:
        r"""Decode discrete code to audio.

        d: latent_dim

        Args:
            codes: (b, q, t)

        Returns:
            audio: (b, c, l)
        """

        with torch.no_grad():
            self.codec.eval()
            z, _, _ = self.codec.quantizer.from_codes(codes)  # (b, d, t)
            audio = self.codec.decode(z)  # (b, c, l)

        return audio