import torch
import torch.nn as nn
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class Mono2Stereo_Mel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vocoder = Mel_BigVGAN_44kHz()

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
            stereo = data["mixture"].to(device)  # (b, 2, l)
            target = self.vocoder.encode(stereo)  # (b, 2, t, f)

            # Mel spectrogram of mono audio as condition
            mono = torch.mean(stereo, keepdim=True, dim=1)  # (b, 1, l)
            cond_tf = self.vocoder.encode(mono)  # (b, 1, t, f)

            cond_dict = {
                "y": None,
                "c": None,
                "ct": None,
                "ctf": cond_tf,
                "cx": None
            }

            cond_sources = {
                "audio": mono, 
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