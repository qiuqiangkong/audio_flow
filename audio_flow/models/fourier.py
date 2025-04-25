import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from einops import rearrange


class Fourier(nn.Module):
    
    def __init__(self, 
        n_fft=2048, 
        hop_length=441, 
        return_complex=True, 
        normalized=True
    ):
        super(Fourier, self).__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.return_complex = return_complex
        self.normalized = normalized

    def stft(self, waveform: Tensor) -> Tensor:
        """Calculate the STFT of a waveform.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: time_steps
        f: freq_bins

        Args:
            waveform: (b, c, l)

        Returns:
            complex_sp: (b, c, t, f)
        """

        B, C, T = waveform.shape

        x = rearrange(waveform, 'b c l -> (b c) l')

        x = torch.stft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
            return_complex=self.return_complex
        )  # (b*c, f, t)

        complex_sp = rearrange(x, '(b c) f t -> b c t f', b=B, c=C)  # (b, c, t, f)

        return complex_sp

    def istft(self, complex_sp: Tensor) -> Tensor:
        """Convert complex spectrogram to waveform using ISTFT.

        Args:
            complex_sp: (b, c, t, f)

        Returns:
            waveform: (b, c, l)
        """

        B, C, T, F = complex_sp.shape

        x = rearrange(complex_sp, 'b c t f -> (b c) f t')

        x = torch.istft(
            input=x, 
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.n_fft).to(x.device),
            normalized=self.normalized,
        )  # (b*c, l)

        x = rearrange(x, '(b c) t -> b c t', b=B, c=C)  # (b, c, l)
        
        return x