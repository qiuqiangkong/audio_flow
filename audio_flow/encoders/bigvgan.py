import bigvgan
import torch
import torch.nn as nn
from bigvgan.meldataset import get_mel_spectrogram
from einops import rearrange
from torch import Tensor

from audio_flow.utils import suppress_print


class Mel_BigVGAN_44kHz(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)
        self.model.remove_weight_norm()

    def encode(self, audio: Tensor) -> Tensor:
        r"""Encode audio to mel spectrogram.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins

        Args:
            audio: (b, c, l)

        Outputs:
            x: (b, c, t, f)
        """

        with torch.no_grad() and suppress_print():
            self.model.eval()
            x = [get_mel_spectrogram(audio[:, i, :], self.model.h) for i in range(audio.shape[1])]
            x = torch.stack(x, dim=1)  # (b, c, f, t)

        x = rearrange(x, 'b c f t -> b c t f')  # (b, c, t, f)
        x = self.normalize(x)

        return x

    def decode(self, x: Tensor) -> Tensor:
        r"""Decode mel spectrogram to audio.

        Args:
            x: (b, c, t, f), mel spectrogram

        Outputs:
            out: (b, c, l), audio
        """
        
        x = self.denormalize(x)
        x = rearrange(x, 'b c t f -> b c f t')  # (b, c, f, t)

        with torch.no_grad():
            self.model.eval()
            x = torch.cat([self.model(x[:, i, :, :]) for i in range(x.shape[1])], dim=1)  # (b, c, l)

        return x

    def normalize(self, x: Tensor) -> Tensor:
        r"""Normalize log mel spectrogram to around -1 ~ +1. This can stabilize 
        the training of flow matching/diffusion models."""
        return (x + 5) / 5

    def denormalize(self, x: Tensor) -> Tensor:
        r"""Denormalize."""
        return x * 5 - 5