r"""Copied from https://github.com/AudioFans/audidata/blob/main/audidata/datasets/librispeech.py"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.transforms.onehot import OneHot
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class LibriSpeech(Dataset):
    r"""LibriSpeech [1] is a speech dataset containing 292,367 English speech 
    files. The total duration is 1,000 hours. All audio files are mono and 
    sampled at 16,000 Hz. The durations of the audio files range from 0.83 s to 
    35.16 s, with 99.9% of the files having durations of less than 20 s. After 
    decompression, the dataset size is 60 GB.

    [1] Vassil Panayotov, et al., LibriSpeech: an ASR corpus based on public 
    domain audio books", ICASSP 2015

    The dataset looks like:

        librispeech (60 GB)
        ├── dev-clean (40 folders)
        │   ├── 1272 (3 folders)
        │   │   ├── 128104
        │   │   │   ├── 1272-128104-0000.flac
        │   │   │   ├── ...
        │   │   │   ├── 1272-128104-0014.flac
        │   │   │   └── 1272-128104.trans.txt
        │   │    ...
        │    ...
        ├── dev-other (33 folders)
        ├── test-clean (40 folders)
        ├── test-other (33 folders)
        ├── train-clean-100 (251 folders)
        ├── train-clean-360 (921 folders)
        ├── train-other-500 (1166 folders)
        ├── BOOKS.TXT
        ├── CHAPTERS.TXT
        ├── LICENSE.TXT
        ├── README.TXT
        └── SPEAKERS.TXT
    """

    # The original webpage http://marsyas.info/index.html is no longer available anymore.
    URL = "https://www.openslr.org/12"

    def __init__(
        self, 
        root: str = None, 
        split: str | list[str] = ["train-clean-100", "train-clean-360", "train-other-500"],
        sr: float = 16000,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=30.),
        transform: None | callable = Mono(),
        target_transform: None | callable = None,
    ) -> None:
    
        if isinstance(split, str):
            if split == "train":
                self.splits = ["train-clean-100", "train-clean-360", "train-other-500"]
            elif split == "dev":
                self.splits = ["dev-clean", "dev-other"]
            elif split == "test":
                self.splits = ["test-clean", "test-other"]
            else:
                raise ValueError

        elif isinstance(split, list):
            for name in split:
                assert name in ["train-clean-100", "train-clean-360", 
                    "train-other-500", "dev-clean", "dev-other", "test-clean", 
                    "test-other"]
            self.splits = split

        else:
            raise ValueError(split)


        self.root = root
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {LibriSpeech.URL}")

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = str(self.meta_dict["audio_path"][index])
        caption = self.meta_dict["caption"][index]

        full_data = {
            "dataset_name": "LibriSpeech",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        target_data = self.load_target_data(caption=caption)
        full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self) -> dict:
        r"""Load metadata of the GTZAN dataset.
        """

        meta_dict = {
            "audio_name": [],
            "audio_path": [],
            "caption": [],
        }

        for split in self.splits:

            audios_dir = Path(self.root, split)

            for caption_path in Path(audios_dir).rglob('*trans.txt'):

                df = pd.read_csv(caption_path, sep="\t", header=None)
                data = df[0].str.split(" ", n=1, expand=True)

                audio_names = [str(name) for name in data[0].values]
                audio_paths = [str(Path(caption_path.parent, f"{name}.flac")) for name in audio_names]
                captions = [caption for caption in data[1].values]

                meta_dict["audio_name"].extend(audio_names)
                meta_dict["audio_path"].extend(audio_paths)
                meta_dict["caption"].extend(captions)
             
        return meta_dict

    def load_audio_data(self, path: str) -> dict:

        audio_duration = librosa.get_duration(filename=path)
        
        if self.crop:
            start_time, clip_duration = self.crop(audio_duration=audio_duration)
        else:
            start_time = 0.
            clip_duration = audio_duration

        # Load a clip
        audio = load(
            path=path, 
            sr=self.sr, 
            offset=start_time, 
            duration=clip_duration
        )  # shape: (channels, audio_samples)

        # Transform audio
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration
        }

        return data

    def load_target_data(self, caption: str) -> dict:

        target = caption

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        data = {
            "caption": caption,
            "target": target,
        }

        return data