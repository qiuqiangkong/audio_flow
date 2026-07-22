import numpy as np

from audioflow.utils.misc import sample_aligned_start_time, load_data_by_time


class MSSDataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_dur = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        # Input
        text = meta["input"]["text"]
        in_path = meta["input"]["audio"]["path"]
        in_fps = meta["input"]["audio"]["fps"]

        # Target
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]
        dur = min(meta["input"]["audio"]["duration"], meta["target"]["audio"]["duration"])

        # Load data
        start = meta.get("start_time", sample_aligned_start_time(dur, self.clip_dur, tgt_fps))
        in_feature, in_mask = load_data_by_time(in_path, start, self.clip_dur, in_fps)
        tgt_latent, tgt_mask = load_data_by_time(tgt_path, start, self.clip_dur, tgt_fps)
        # latent: (l, d), mask: (l,)

        data = {
            "text": text,
            # 
            "input_feature": in_feature,  # (l_in, d)
            "input_mask": in_mask,  # (l_in,)
            "input_fps": in_fps,
            #
            "target": tgt_latent,  # (l_tgt, d)
            "target_mask": tgt_mask,  # (l_tgt,)
            "target_fps": tgt_fps
        }
        
        return data
