import torch
import torch.nn as nn
from audidata.datasets import GTZAN
from torch import Tensor

from audio_flow.encoders.bigvgan import Mel_BigVGAN_44kHz


class Text2Music_Mel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.lb_to_ix = GTZAN.LB_TO_IX
        self.ix_to_lb = GTZAN.IX_TO_LB

        self.vocoder = Mel_BigVGAN_44kHz()

    def __call__(self, data: dict) -> tuple[Tensor, dict, dict]:
        r"""Transform data into latent representations and conditions.

        b: batch_size
        c: channels_num
        l: audio_samples
        t: frames_num
        f: mel bins
        """
        
        device = next(self.parameters()).device
        name = data["dataset_name"][0]

        if name in ["GTZAN"]:

            # Mel spectrogram target
            audio = data["audio"].to(device)  # (b, c, l)
            target = self.vocoder.encode(audio)  # (b, c, t, f)

            # Condition
            captions = data["label"]  # (b,)
            y = torch.LongTensor([self.lb_to_ix[lb] for lb in captions]).to(device)  # (b,)

            cond_dict = {
                "y": y,
                "c": None,
                "ct": None,
                "ctf": None,
                "cx": None
            }

            cond_sources = {
                "caption": captions,
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