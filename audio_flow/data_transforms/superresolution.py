import torch.nn as nn
import torchaudio
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class SuperResolution_Mel(nn.Module):
    def __init__(self, sr: float, distorted_sr: float):
        super().__init__()

        self.vocoder = Mel_BigVGAN_44kHz()
        self.sr = sr
        self.distorted_sr = distorted_sr

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
            distorted_audio = resample(x=audio, sr=self.sr, distorted_sr=self.distorted_sr)  # (b, c, l)
            cond_tf = self.vocoder.encode(distorted_audio)  # (b, c, t, f)

            cond_dict = {
                "y": None,
                "c": None,
                "ct": None,
                "ctf": cond_tf,
                "cx": None
            }

            cond_sources = {
                "audio": distorted_audio, 
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


def resample(x: Tensor, sr: float, distorted_sr: float) -> Tensor:
    x = torchaudio.functional.resample(x, orig_freq=sr, new_freq=distorted_sr)
    x = torchaudio.functional.resample(x, orig_freq=distorted_sr, new_freq=sr)
    return x