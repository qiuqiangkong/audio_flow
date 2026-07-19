from __future__ import annotations

import numpy as np
import librosa
from xml.etree import ElementTree
import random
import math
import h5py


def augment_path(path: Path, i: int) -> Path:
    if i == 0:
        return path
    else:
        return path.parent / f"{path.stem}.aug{i:04d}{path.suffix}"


def get_single_value(lst: list):
    unique = list(set(lst))
    assert len(unique) == 1
    return unique[0]


def logmel(audio: np.ndarray, sr: float) -> np.ndarray:

    if audio.ndim == 2:
        audio = np.mean(audio, axis=0)

    return np.log10(librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=2048, 
        hop_length=round(sr * 0.01), 
        n_mels=128
    )).T  # (t, f)


def check_prompt(text: str) -> bool:
    try:
        ElementTree.fromstring(text)
        return True
    except ElementTree.ParseError:
        return False


def get_data_length(path: str) -> int:
    with h5py.File(path, 'r') as hf:
        return hf["data"].shape[0]


def load_data(
    path: str, 
    start_frame: int, 
    n_frames: int
) -> tuple[np.ndarray, np.ndarray]:
    r"""Load data from hdf5.

    l: n_frames
    
    Returns:
        x: (l, ...)
        mask (bool): (l,)
    """
    with h5py.File(path, 'r') as hf:
        x = hf["data"][start_frame : start_frame + n_frames, ...]  # (l, ...)

    if x.dtype == np.bool_:
        x = x.astype(np.float32)

    mask = np.zeros(n_frames, dtype=bool)  # (l,)
    mask[:x.shape[0]] = True  # (l,)
    x = librosa.util.fix_length(data=x, size=n_frames, axis=0, constant_values=0.)  # (l, ...)
        
    return x, mask


def load_data_by_time(
    path: str, 
    start_time: float, 
    clip_duration: float, 
    fps: float
) -> tuple[np.ndarray, np.ndarray]:
    r"""Load data from hdf5.

    l: n_frames

    Returns:
        x: (l, ...)
        mask (bool): (l,)
    """
    return load_data(
        path=path, 
        start_frame=int(start_time * fps), 
        n_frames=int(clip_duration * fps)
    )


def sample_start_time(total_duration: float, clip_duration: float) -> float:
    span = max(total_duration - clip_duration, 0.)
    return random.uniform(0., span)


def quantize_time(t: float, fps: float) -> float:
    return int(t * fps) / fps


def sample_aligned_start_time(total_duration: float, clip_duration: float, fps: float) -> float:
    return quantize_time(sample_start_time(total_duration, clip_duration), fps)