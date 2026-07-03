import random

import h5py
import librosa
import numpy as np

from audioflow.utils.xml import dict_to_xml
# from audioflow.utils.audio import get_latent_length, sample_start_frame, load_latent
from audioflow.utils.misc import sample_aligned_start_time, load_data_by_time


class V2ADataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_dur = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get video to audio data."""

        # Input text
        prompt = meta["input"]["text"]

        # Input video feature
        in_path = meta["input"]["video"]["path"]
        in_fps = meta["input"]["video"]["fps"]

        # Target audio latent
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]

        dur = min(meta["input"]["video"]["duration"], meta["target"]["audio"]["duration"])

        # Get start time
        start = meta.get("start_time", sample_aligned_start_time(dur, self.clip_dur, tgt_fps))
        
        # Load data
        in_feature, in_mask = load_data_by_time(in_path, start, self.clip_dur, in_fps)
        tgt_latent, tgt_mask = load_data_by_time(tgt_path, start, self.clip_dur, tgt_fps)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "input_feature": in_feature,  # (l, d)
            "input_mask": in_mask,  # (l,)
            "target_latent": tgt_latent,  # (l, d)
            "target_mask": tgt_mask,  # (l,)
        }
        
        return data


'''
class V2ADataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get video to audio data."""

        # Input text
        prompt = meta["input"]["text"]

        # Input video feature
        in_path = meta["input"]["video"]["path"]
        in_fps = meta["input"]["video"]["fps"]

        # Target audio latent
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]

        fps = min(in_fps, tgt_fps)
        
        clip_frames = round(self.clip_duration * fps)
        total_frames = min(get_latent_length(in_path), get_latent_length(tgt_path))
        start = sample_start_frame(total_frames, clip_frames)

        in_feature, in_mask = load_latent(in_path, start, clip_frames)
        tgt_latent, tgt_mask = load_latent(tgt_path, start, clip_frames)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "input_feature": in_feature,  # (l, d)
            "input_mask": in_mask,  # (l,)
            "target_latent": tgt_latent,  # (l, d)
            "target_mask": tgt_mask,  # (l,)
        }
        
        return data
'''

