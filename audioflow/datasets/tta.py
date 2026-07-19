import numpy as np

from audioflow.utils.misc import sample_aligned_start_time, load_data_by_time


class TTADataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_dur = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        # Input text
        text = meta["input"]["text"]

        # Target audio latent
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]
        dur = meta["target"]["audio"]["duration"]
        
        start = sample_aligned_start_time(dur, self.clip_dur, tgt_fps)
        tgt_latent, tgt_mask = load_data_by_time(tgt_path, start, self.clip_dur, tgt_fps)  # (l, d)

        data = {
            "text": text,
            #
            "target": tgt_latent,
            "target_mask": tgt_mask,
            "target_fps": tgt_fps
        }

        return data
