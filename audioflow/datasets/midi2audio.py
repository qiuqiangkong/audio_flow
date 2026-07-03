import numpy as np

from audioflow.utils.misc import sample_aligned_start_time, load_data_by_time


class Midi2AudioDataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_dur = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        # Input text
        prompt = meta["input"]["text"]
        in_path = meta["input"]["midi"]["path"]
        in_fps = meta["input"]["midi"]["fps"]
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]
        dur = min(meta["input"]["midi"]["duration"], meta["target"]["audio"]["duration"])

        if "start_time" in meta:
            start = meta["start_time"]
        else:
            start = sample_aligned_start_time(dur, self.clip_dur, tgt_fps)
            
        in_feature, in_mask = load_data_by_time(in_path, start, self.clip_dur, in_fps)
        tgt_latent, tgt_mask = load_data_by_time(tgt_path, start, self.clip_dur, tgt_fps)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "input_feature": in_feature.astype(np.float32),  # (l, d)
            "input_mask": in_mask,  # (l,)
            "target_latent": tgt_latent,  # (l, d)
            "target_mask": tgt_mask,  # (l,)
        }
        
        return data
