import random

import h5py
import librosa
import numpy as np

from audio_flow.utils import get_audio_latent_length, sample_start_frame, load_audio_latent


class Video2AudioDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""
        
        task = meta["task"]
        prompt = meta["input"]["text"]["prompt"]
        input_latent_path = meta["input"]["video"]["latent_path"]
        target_latent_path = meta["target"]["audio"]["latent_path"]
      
        input_fps = meta["input"]["video"]["fps"]
        target_fps = meta["target"]["audio"]["fps"]
        assert input_fps == target_fps
        input_frames = int(meta["input"]["video"]["duration"] * input_fps)
        target_frames = int(meta["target"]["audio"]["duration"] * target_fps)
        total_frames = min(input_frames, target_frames)

        clip_frames = round(self.clip_duration * input_fps)
        start = sample_start_frame(total_frames, clip_frames)

        input_latent, _, _ = load_audio_latent(input_latent_path, start, clip_frames)
        target_latent, mask, length = load_audio_latent(target_latent_path, start, clip_frames)
        # input_latent: (l, d)

        data = {
            "task": task,
            "prompt": prompt,
            "input_latent_path": input_latent_path,
            "input_latent": input_latent,  # (l, d)
            "target_latent": target_latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data


class Video2AudioMaeDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""
        
        task = meta["task"]
        prompt = meta["input"]["text"]["prompt"]
        input_latent_path = meta["input"]["video"]["latent_path"]
        target_latent_path = meta["target"]["audio"]["latent_path"]
      
        input_fps = meta["input"]["video"]["fps"]
        target_fps = meta["target"]["audio"]["fps"]
        # assert input_fps == target_fps
        input_frames = int(meta["input"]["video"]["duration"] * input_fps)
        target_frames = int(meta["target"]["audio"]["duration"] * target_fps)
        # total_frames = min(input_frames, target_frames)

        input_clip_frames = round(self.clip_duration * input_fps)
        input_start = sample_start_frame(input_frames, input_clip_frames)
        target_start = input_start * round(target_fps / input_fps)
        target_clip_frames = input_clip_frames * round(target_fps / input_fps)

        input_latent, _, _ = load_audio_latent(input_latent_path, input_start, input_clip_frames)
        target_latent, mask, length = load_audio_latent(target_latent_path, target_start, target_clip_frames)
        # input_latent: (l, d)

        data = {
            "task": task,
            "prompt": prompt,
            "input_latent_path": input_latent_path,
            "input_latent": input_latent,  # (l, d)
            "target_latent": target_latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data