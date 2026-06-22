import random

import h5py
import librosa
import numpy as np

from audio_flow.datasets.ttm import TTMDataset
from audio_flow.datasets.tts import TTSDataset


class MetaDataset:
    def __init__(self, clip_duration: float):
        self.ttm = TTMDataset(clip_duration)
        self.tts = TTSDataset(clip_duration)
        
    def __getitem__(self, meta: dict) -> dict:
        r"""Load data from meta."""
        task = meta["task"]

        if task == "text to music":
            return self.ttm[meta]

        elif task == "text to speech":
            return self.tts[meta]
            
        elif task == "text to audio":
            return get_tta_data(meta, self.clip_duration)

        # elif task in ["music_source_separation", "mono_to_stereo", "super-resolution", "codec_to_music"]:
        #     return get_aligned_data(meta, self.clip_duration)

        else:
            raise ValueError(task)

'''
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
        total_frames = get_latent_length(latent_path)
        start = sample_start_frame(total_frames, clip_frames)

        latent, mask, length = load_latent(latent_path, start, clip_frames)
        # latent: (l, d), mask: (l,)

        data = {
            "task": task,
            "prompt": prompt,
            "target_latent": latent,  # (l, d)
            "target_mask": mask,  # (l,)
            "target_length": length
        }
        
        return data





def get_ttm_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to music data."""
    
    task = meta["task"]
    prompt = random.choice(meta["input"]["text"]["prompt"])
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, mask, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "prompt": prompt,
        "target_latent": latent.T,
        "target_mask": mask,
        "target_length": length
    }

    return data


def get_tts_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to speech data."""
    task = meta["task"]
    content = meta["input"]["text"]["content"]
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, mask, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "prompt": content,
        "target_latent": latent.T,
        "target_mask": mask,
        "target_length": length
    }

    # import pickle
    # pickle.dump(latent, open("_zz.pkl", "wb"))
    # from IPython import embed; embed(using=False); os._exit(0)
    return data


def get_tta_data(meta: dict, clip_duration: float) -> dict:
    r"""Get text to audio data."""
    task = meta["task"]
    prompt = random.choice(meta["input"]["text"]["prompt"])
    path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    bgn = random_bgn_frame(total_frames, clip_frames)
    latent, length = load_latent(path, bgn, clip_frames)

    data = {
        "task": task,
        "prompt": content,
        "target_latent": latent.T,
        "target_mask": mask,
        "target_length": length
    }
    return data


def get_aligned_data(meta: dict, clip_duration: float) -> dict:
    r"""Get input and target aligned data."""
    assert meta["input"]["audio"]["latent_type"] == meta["target"]["audio"]["latent_type"]

    task = meta["task"]
    instruction = meta["input"]["text"]["instruction"]
    input_path = meta["input"]["audio"]["path"]
    target_path = meta["target"]["audio"]["path"]
    fps = meta["target"]["audio"]["fps"]
    total_frames = meta["target"]["audio"]["num_frames"]
    clip_frames = round(clip_duration * fps)

    total_frames = get_latent_length(latent_path)
    bgn = random_bgn_frame(total_frames, clip_frames)
    input_latent, _ = load_latent(input_path, bgn, clip_frames)
    target_latent, length = load_latent(target_path, bgn, clip_frames)

    data = {
        "task": task,
        "instruction": instruction,
        "input_audio_latent": input_latent,
        "target_audio_latent": target_latent,
        "latent_length": length
    }
    return data


def get_latent_length(path: str) -> int:
    with h5py.File(path, 'r') as hf:
        return hf["latent"].shape[-1]


def sample_start_frame(total_frames: int, clip_frames: int) -> int:
    r"""Random sample a frame index."""
    max_start = max(total_frames - clip_frames, 0)
    return random.randint(0, max_start)


def load_latent(path: str, start: int, clip_frames: int) -> tuple[np.ndarray, np.ndarray, int]:
    r"""Load latent from hdf5."""
    with h5py.File(path, 'r') as hf:
        latent = hf["latent"][start : start + clip_frames, :]  # (l, d)
        length = latent.shape[0]

        latent = librosa.util.fix_length(data=latent, size=clip_frames, axis=0, constant_values=0.)
        mask = np.zeros(clip_frames, dtype=bool)
        mask[:length] = True

    return latent, mask, length
'''