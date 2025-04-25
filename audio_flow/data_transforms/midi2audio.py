import torch.nn as nn
import torchaudio
from einops import rearrange
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class Midi2Audio_Mel(nn.Module):
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

        if name in ["MAESTRO"]:
            
            # Mel spectrogram target
            audio = data["audio"].to(device)  # (b, c, l)
            target = self.vocoder.encode(audio)  # (b, c, t, f)

            # Piano roll condition
            frame_roll = data["frame_roll"].to(device)  # (b, t, d)
            cond_t = rearrange(frame_roll, 'b t d -> b d t')  # (b, d, t)

            cond_t = torchaudio.functional.resample(
                waveform=cond_t.contiguous(), 
                orig_freq=cond_t.shape[2], 
                new_freq=target.shape[2]
            )  # (b, d, t)
            
            cond_dict = {
                "y": None,
                "c": None,
                "ct": cond_t,
                "ctf": None,
                "cx": None
            }

            cond_sources = {
                "image": frame_roll, 
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