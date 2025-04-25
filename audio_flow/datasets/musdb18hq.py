from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from torch.utils.data import Dataset
from typing_extensions import Literal


class MUSDB18HQ(Dataset):
    r"""MUSDB18HQ [1] is a dataset containing 100 training audio files and 50 
    testing audio files, each with vocals, bass, drums, and other stems. The 
    total duration is 9.8 hours. The audio is stereo and sampled at 48,000 Hz. 
    After decompression, the dataset size is 30 GB.

    [1] https://zenodo.org/records/3338373

    The dataset looks like:

        musdb18hq (30 GB)
        ├── train (100 files)
        │   ├── A Classic Education - NightOwl
        │   │   ├── bass.wav
        │   │   ├── drums.wav
        │   │   ├── mixture.wav
        │   │   ├── other.wav
        │   │   └── vocals.wav
        │   ... 
        │   └── ...
        └── test (50 files)
            ├── Al James - Schoolboy Facination
            │   ├── bass.wav
            │   ├── drums.wav
            │   ├── mixture.wav
            │   ├── other.wav
            │   └── vocals.wav
            ... 
            └── ...
    """

    URL = "https://zenodo.org/records/3338373"

    DURATION = 35359.56  # Dataset duration (s), 9.8 hours, including training, 
    # valdiation, and testing

    def __init__(
        self,
        root: str = "/datasets/musdb18hq", 
        split: Literal["train", "test"] = "train",
        sr: int = 44100,
        crop: callable = RandomCrop(clip_duration=2.),
        target_stems: list[str] = ["vocals"],
        time_align: Literal["strict", "group", "random"] = "group",
        stem_transform: None | callable | list[callable] = None,
        group_transform: None | callable | list[callable] = None,
        mixture_transform: None | callable | list[callable] = None,
    ) -> None:
        r"""
        time_align: str. "strict" indicates all stems are aligned (from the 
            same song and have the same start time). "group" indictates 
            target stems / background stems are aligned. "random" indicates 
            all stems are from different songs with different start time.
        """

        self.stems = ["bass", "drums", "other", "vocals"]
        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.target_stems = target_stems
        self.bg_stems = list(set(self.stems) - set(self.target_stems))
        self.time_align = time_align
        self.stem_transform = group_transform
        self.group_transform = group_transform
        self.mixture_transform = group_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {MUSDB18HQ.URL}")

        self.audios_dir = Path(self.root, self.split)
        self.list_names = sorted(os.listdir(self.audios_dir))
        self.audios_num = len(self.list_names)
        
    def __getitem__(
        self, 
        index: Union[int, dict],
    ) -> dict:

        # Use different song indexes for different stems
        index_dict = self.get_index_dict(index)
        # E.g., {"bass": 94, "drums": 94, "other": 35, "vocals": 35}

        audio_names = {}
        audio_paths = {}
        start_times = {}
        clip_durations = {}

        for stem in self.stems:
            
            audio_names[stem] = self.list_names[index_dict[stem]]
            audio_paths[stem] = str(Path(self.audios_dir, audio_names[stem], "{}.wav".format(stem)))
            audio_duration = librosa.get_duration(path=audio_paths[stem])
            start_times[stem], clip_durations[stem] = self.crop(audio_duration=audio_duration)

        start_times = self.update_start_times(start_times)
        # E.g., {"bass": 44.86, "drums": 139.68, "other": 44.86, "vocals": 139.68}

        data = {
            "dataset_name": "MUSDB18HQ",
        }

        for stem in self.stems:

            # Load a clip
            data[stem] = load(
                path=audio_paths[stem], 
                sr=self.sr, 
                offset=start_times[stem], 
                duration=clip_durations[stem],
            )
            # shape: (channels, audio_samples)

            data["{}_audio_name".format(stem)] = audio_names[stem]
            data["{}_audio_path".format(stem)] = audio_paths[stem]
            data["{}_start_time".format(stem)] = start_times[stem]

            # Transform source
            if self.stem_transform is not None:
                data[stem] = self.stem_transform(data[stem])

        # Sum sources to target and background
        data["target"], data["background"] = self.sources_to_target_and_background(data)

        # Transform target and background
        if self.group_transform is not None:
            data["target"] = self.group_transform(data["target"])
            data["background"] = self.group_transform(data["background"])

        # Sum target and background to mixture
        data["mixture"] = data["target"] + data["background"]

        # Transform mixture
        if self.mixture_transform is not None:
            data["mixture"] = self.mixture_transform["mixture"]

        return data

    def __len__(self) -> int:
        return self.audios_num

    def get_index_dict(self, index: Union[int, dict]) -> dict:
        r"""Get song indexes of different stems."""

        if isinstance(index, int):
            # All sources have same indexes (from the same song)
            index_dict = {stem: index for stem in self.stems}

        elif isinstance(index, dict):
            index_dict = index

            if self.time_align == "strict":
                assert(self.equal_values(index_dict.values()))

            elif self.time_align == "group":
                assert(self.equal_values([index_dict[stem] for stem in self.target_stems]))
                assert(self.equal_values([index_dict[stem] for stem in self.bg_stems]))
                
            elif self.time_align == "random":
                pass

        else:
            raise TypeError(index)

        return index_dict

    def equal_values(self, x: list) -> bool:
        r"""Check if all values are the same."""
        return len(set(x)) == 1

    def update_start_times(self, start_times: dict) -> dict:
        r"""Update start times according to different time_align types."""

        if self.time_align == "strict":
            for stem in self.stems:
                start_times[stem] = start_times[self.stems[0]]
        
        elif self.time_align == "group":
            
            for stem in self.target_stems:
                start_times[stem] = start_times[self.target_stems[0]]
            
            for stem in self.bg_stems:
                start_times[stem] = start_times[self.bg_stems[0]]

        elif self.time_align == "random":
            pass

        else:
            raise ValueError(self.time_align)

        return start_times

    def sources_to_target_and_background(self, data: dict) -> tuple[np.ndarray, np.ndarray]:
        r"""Sum sources to target and background."""

        target = 0
        bg = 0

        for stem in self.target_stems:
            target += data[stem]

        for stem in self.bg_stems:
            bg += data[stem]

        return target, bg