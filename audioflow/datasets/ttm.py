from audioflow.utils.misc import sample_aligned_start_time, load_data_by_time


class TTMDataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_dur = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        # Input text
        prompt = meta["input"]["text"]

        # Target audio latent
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]
        dur = meta["target"]["audio"]["duration"]
        
        start = sample_aligned_start_time(dur, self.clip_dur, tgt_fps)
        tgt_latent, tgt_mask = load_data_by_time(tgt_path, start, self.clip_dur, tgt_fps)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "target_latent": tgt_latent,  # (l, d)
            "target_mask": tgt_mask,  # (l,)
        }
        
        return data


'''
from audioflow.utils.audio import get_latent_length, load_latent, sample_start_frame


class TTMDataset:
    def __init__(self, clip_duration: float) -> None:
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""

        # Input text
        prompt = meta["input"]["text"]

        # Target audio latent
        tgt_path = meta["target"]["audio"]["path"]
        tgt_fps = meta["target"]["audio"]["fps"]
        
        clip_frames = round(self.clip_duration * tgt_fps)
        total_frames = get_latent_length(tgt_path)
        start = sample_start_frame(total_frames, clip_frames)
        
        tgt_latent, tgt_mask = load_latent(tgt_path, start, clip_frames)
        # latent: (l, d), mask: (l,)

        data = {
            "prompt": prompt, 
            "target_latent": tgt_latent,  # (l, d)
            "target_mask": tgt_mask,  # (l,)
        }
        
        return data
'''

