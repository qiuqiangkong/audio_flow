r"""Code from: https://github.com/qiuqiangkong/audio_understanding/blob/main/audio_understanding/audio_encoders/whisper.py"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import whisper

class Whisper(nn.Module):

    def __init__(self, sr: float, trainable: bool) -> None:
        r"""Whisper audio encoder [1]

        [1] A. Radford, et al., Robust Speech Recognition via Large-Scale Weak 
        Supervision, ICML, 2023

        Code: https://github.com/qiuqiangkong/piano_transcription_inference
        """

        super().__init__()

        self.audio_sr = sr
        self.model_sr = 16000  # Whisper sampling rate
        self.trainable = trainable
        
        self.clip_samples = round(self.model_sr * 30.)  # Whisper requires 30s as input
        self.fps = 50

        self.model = whisper.load_model("base")
        self.latent_dim = 512

        # Fix for parallel training
        self.model.register_buffer(
            name="alignment_heads", 
            tensor=self.model.get_buffer("alignment_heads").to_dense(), 
            persistent=False
        )

    def encode(self, audio: torch.Tensor, train_mode: bool) -> torch.Tensor:
        r"""Extract audio latent.

        Args:
            audio: (b, c, t)

        Returns:
            latent: (b, t, d)
        """

        # Resample audio
        audio = torchaudio.functional.resample(
            waveform=audio, 
            orig_freq=self.audio_sr, 
            new_freq=self.model_sr
        )

        # Pad to 30s required by Whisper
        T = audio.shape[-1]
        pad_len = self.clip_samples - T
        audio = F.pad(input=audio, pad=(0, pad_len), mode="constant", value=0)  # (b, c, t)

        # To mono
        audio = torch.mean(audio, dim=1)  # shape: (b, t)

        # Mel spectrogram
        mel = whisper.log_mel_spectrogram(audio)  # (b, f, t)

        if self.trainable and train_mode:
            self.model.train()
        else:
            self.model.eval()

        # Forward
        with torch.set_grad_enabled(self.trainable and train_mode):
            latent = self.model.encoder(mel)  # (b, t, d)

        # Clip to original length
        frames_num = round((T / self.model_sr) * self.fps)
        latent = latent[:, 0 : frames_num, :]  # (b, t, d)

        return latent