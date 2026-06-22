import random

import h5py
import librosa
import numpy as np

from audio_flow.utils import get_audio_latent_length, sample_start_frame, load_audio_latent


class EditingDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""
        
        task = meta["task"]
        prompt = meta["input"]["text"]["prompt"]
        input_latent_path = meta["input"]["audio"]["latent_path"]
        target_latent_path = meta["target"]["audio"]["latent_path"]
        fps = meta["input"]["audio"]["fps"]
        
        clip_frames = round(self.clip_duration * fps)
        total_frames = get_audio_latent_length(input_latent_path)
        start = sample_start_frame(total_frames, clip_frames)

        input_latent, _, _ = load_audio_latent(input_latent_path, start, clip_frames)
        target_latent, mask, length = load_audio_latent(target_latent_path, start, clip_frames)
        # input_latent: (l, d)

        data = {
            "task": task,
            "prompt": prompt,
            "input_latent": input_latent,  # (l, d)
            "target_latent": target_latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data