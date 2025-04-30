import torch.nn as nn
import torchaudio
from einops import rearrange
from torch import Tensor
import torch.nn.functional as F

from audio_flow.encoders.bigvgan import Mel_BigVGAN_24kHz
from audio_flow.encoders.whisper import Whisper


class Whisper2Speech(nn.Module):
    def __init__(self, sr, trainable):
        super().__init__()

        self.train_mode = trainable
        self.vocoder = Mel_BigVGAN_24kHz()
        self.encoder = Whisper(sr=sr,trainable=self.train_mode)


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

        if name in ["LibriSpeech"]:
            
            # Mel spectrogram target
            audio = data["audio"].to(device)  # (b, c, l)
            target = self.vocoder.encode(audio)  # (b, c, t, f)

            # Whisper condition
            latent = self.encoder.encode(audio,train_mode=self.train_mode) # (b, t, d)
            cond_t = rearrange(latent, 'b t d -> b d t')  # (b, d, t)
            
            # use resample to deal with feature interpolate
            cond_t = torchaudio.functional.resample(
                waveform=cond_t.contiguous(), 
                orig_freq=cond_t.shape[2], 
                new_freq=target.shape[2]
            )  # (b, d, t)
            # cond_t = F.interpolate(cond_t, size=target.shape[2], mode='linear', align_corners=False)
            
            cond_dict = {
                "y": None,
                "c": None,
                "ct": cond_t,
                "ctf": None,
                "cx": None
            }

            cond_sources = {
                "audio": audio,  
            } # what cond_sources design for ?
            
            return target, cond_dict, cond_sources

        else:
            raise ValueError(name)

    def latent_to_audio(self, x: Tensor) -> Tensor:
        r"""Use vocoder to convert mel spectrogram to audio.

        Args:
            x: (b, c, t, f)

        Outputs:
            y: (b, c, l)
        """

        return self.vocoder.decode(x)