import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz
from audio_flow.encoders.dac import DAC
from audio_flow.utils import fix_length


class Codec2Audio_Mel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocoder = Mel_BigVGAN_44kHz()
        self.dac = DAC(sr=44100, n_quantizers=1)

    def __call__(self, data: dict) -> tuple[Tensor, dict, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """
        
        name = data["dataset_name"][0]
        device = next(self.parameters()).device

        if name in ["MUSDB18HQ"]:
            
            # Mel spectrogram target
            audio = data["mixture"].to(device)  # (b, c, l)
            target = self.vocoder.encode(audio)  # (b, c, t, f)
            
            # Mel spectrogram of distorted audio as condition
            codes = self.dac.encode(torch.mean(audio, keepdim=True, dim=1))  # (b, 1, t)
            recon_audio = self.dac.decode(codes)  # (b, c, l)
            recon_audio = fix_length(recon_audio, audio.shape[-1])  # (b, c, l)
            
            cond_tf = self.vocoder.encode(recon_audio)  # (b, c, t, f)

            cond_dict = {
                "y": None,
                "c": None,
                "ct": None,
                "ctf": cond_tf,
                "cx": None
            }

            cond_sources = {
                "audio": recon_audio, 
            }
            
            return target, cond_dict, cond_sources

        else:
            raise ValueError(name)

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Ues vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, l)
        """
        return self.vocoder.decode(x)