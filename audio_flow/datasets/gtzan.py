from __future__ import annotations

import os
import re
from pathlib import Path

import librosa
import numpy as np
from audidata.io.audio import load
from audidata.io.crops import StartCrop
from audidata.transforms.audio import Mono
from audidata.transforms.onehot import OneHot
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class GTZAN(Dataset):
    r"""GTZAN [1] is a music dataset containing 1,000 30-second audio clips. 
    The total duration is 8.3 hours. GTZAN includes 10 genres. All audio files 
    are mono sampled at 22,050 Hz. After decompression, the dataset size is 
    1.3 GB.

    [1] Tzanetakis, G., et al., Musical genre classification of audio signals. 2002

    The dataset looks like:

        gtzan (1.3 GB)
        └── genres
            ├── blues (100 files)
            ├── classical (100 files)
            ├── country (100 files)
            ├── disco (100 files)
            ├── hiphop (100 files)
            ├── jazz (100 files)
            ├── metal (100 files)
            ├── pop (100 files)
            ├── reggae (100 files)
            └── rock (100 files)
    """

    # The original webpage http://marsyas.info/index.html is no longer available anymore.
    URL = "https://huggingface.co/datasets/qiuqiangkong/gtzan/resolve/main/genres.tar.gz?download=true"

    DURATION = 30024.07  # Dataset duration (s), 8.3 hours

    LABELS = ["blues", "classical", "country", "disco", "hiphop", "jazz", 
        "metal", "pop", "reggae", "rock"]

    CLASSES_NUM = len(LABELS)
    LB_TO_IX = {lb: ix for ix, lb in enumerate(LABELS)}
    IX_TO_LB = {ix: lb for ix, lb in enumerate(LABELS)}

    def __init__(
        self, 
        root: str = None, 
        split: Literal["train", "test"] = "train",
        test_fold: int = 0,  # E.g., fold 0 is used for testing. Fold 1 - 9 are used for training.
        sr: float = 22050,  # Sampling rate
        crop: None | callable = StartCrop(clip_duration=30.),
        transform: None | callable = Mono(),
        target_transform: None | callable = OneHot(classes_num=CLASSES_NUM),
    ) -> None:
    
        self.root = root
        self.split = split
        self.test_fold = test_fold
        self.sr = sr
        self.crop = crop
        self.transform = transform
        self.target_transform = target_transform

        self.labels = GTZAN.LABELS
        self.lb_to_ix = GTZAN.LB_TO_IX
        self.ix_to_lb = GTZAN.IX_TO_LB

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {GTZAN.URL}")

        self.meta_dict = self.load_meta()

    def __getitem__(self, index: int) -> dict:

        audio_path = str(self.meta_dict["audio_path"][index])
        label = self.meta_dict["label"][index]

        full_data = {
            "dataset_name": "GTZAN",
            "audio_path": audio_path,
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        target_data = self.load_target_data(label=label)
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
            "label": [],
        }

        audios_dir = Path(self.root, "genres")

        for genre in self.labels:

            audio_names = sorted(os.listdir(Path(audios_dir, genre)))
            # E.g., len(audio_names) = 1000

            train_audio_names, test_audio_names = self.split_train_test(audio_names)
            # E.g., len(train_audio_names) = 900
            # E.g., len(test_audio_names) = 100

            if self.split == "train":
                filtered_audio_names = train_audio_names

            elif self.split == "test":
                filtered_audio_names = test_audio_names

            else:
                raise ValueError(self.split)

            for audio_name in filtered_audio_names:

                audio_path = str(Path(audios_dir, genre, audio_name))

                meta_dict["audio_name"].append(audio_name)
                meta_dict["audio_path"].append(audio_path)
                meta_dict["label"].append(genre)

        return meta_dict

    def split_train_test(self, audio_names: list) -> tuple[list, list]:

        train_audio_names = []
        test_audio_names = []

        test_ids = range(self.test_fold * 10, (self.test_fold + 1) * 10)
        # E.g., if test_fold = 3, then test_ids = [30, 31, 32, ..., 39]

        for audio_name in audio_names:

            audio_id = int(re.search(r'\d+', audio_name).group())
            # E.g., if audio_name is "blues.00037.au", then audio_id = 37

            if audio_id in test_ids:
                test_audio_names.append(audio_name)

            else:
                train_audio_names.append(audio_name)

        return train_audio_names, test_audio_names

    def load_audio_data(self, path: str) -> dict:

        audio_duration = librosa.get_duration(path=path)

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
        )
        # shape: (channels_num, audio_samples)

        # Transform audio
        if self.transform is not None:
            audio = call(transform=self.transform, x=audio)

        data = {
            "audio": audio, 
            "start_time": start_time,
            "duration": clip_duration
        }

        return data

    def load_target_data(self, label: str) -> dict:

        target = self.lb_to_ix[label]

        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        data = {
            "label": label,
            "target": target  # shape: (classes_num,)
        }

        return data