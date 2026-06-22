import random

import h5py
import librosa
import numpy as np

from audio_flow.utils import get_audio_latent_length, sample_start_frame, load_audio_latent


class TTMDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""
        
        task = meta["task"]
        prompt = meta["input"]["text"]["prompt"]
        latent_path = meta["target"]["audio"]["latent_path"]
        fps = meta["target"]["audio"]["fps"]
        
        clip_frames = round(self.clip_duration * fps)
        total_frames = get_audio_latent_length(latent_path)
        start = sample_start_frame(total_frames, clip_frames)

        latent, mask, length = load_audio_latent(latent_path, start, clip_frames)
        # latent: (l, d), mask: (l,)

        data = {
            "task": task,
            "prompt": prompt,
            "target_latent": latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data