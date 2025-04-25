from __future__ import annotations

from pathlib import Path

import librosa
import pandas as pd
from audidata.io.audio import load
from audidata.io.crops import RandomCrop
from audidata.io.midi import read_single_track_midi, clip_notes
from audidata.transforms.audio import Mono
from audidata.transforms.midi import PianoRoll
from audidata.utils import call
from torch.utils.data import Dataset
from typing_extensions import Literal


class MAESTRO(Dataset):
    r"""MAESTRO [1] is a dataset containing 199 hours of audio from 1,276 files, 
    along with aligned MIDI files captured by a Yamaha Disklavier. All audio 
    files are stereo and sampled at 44,100 Hz. After decompression, the dataset 
    size is 131 GB.

    [1] C. Hawthorne, et al., Enabling Factorized Piano Music Modeling and 
    Generation with the MAESTRO Dataset, ICLR, 2019

    The dataset looks like:

        maestro-v3.0.0 (131 GB)
        ├── 2004 (132 songs, wav + flac + midi + tsv)
        ├── 2006 (115 songs, wav + flac + midi + tsv)
        ├── 2008 (147 songs, wav + flac + midi + tsv)
        ├── 2009 (125 songs, wav + flac + midi + tsv)
        ├── 2011 (163 songs, wav + flac + midi + tsv)
        ├── 2013 (127 songs, wav + flac + midi + tsv)
        ├── 2014 (105 songs, wav + flac + midi + tsv)
        ├── 2015 (129 songs, wav + flac + midi + tsv)
        ├── 2017 (140 songs, wav + flac + midi + tsv)
        ├── 2018 (93 songs, wav + flac + midi + tsv)
        ├── LICENSE
        ├── maestro-v3.0.0.csv
        ├── maestro-v3.0.0.json
        └── README
    """

    URL = "https://magenta.tensorflow.org/datasets/maestro"

    DURATION = 717232.49  # Dataset duration (s), 199 hours, including training, 
    # validation, and testing.

    def __init__(
        self, 
        root: str, 
        split: str = Literal["train", "validation", "test"],
        sr: float = 44100,
        crop: None | callable = RandomCrop(clip_duration=10., end_pad=9.9),
        transform: None | callable = Mono(),
        load_target: bool = True,
        extend_pedal: bool = True,
        target_transform: None | callable = PianoRoll(fps=100, pitches_num=128),
    ) -> None:

        self.root = root
        self.split = split
        self.sr = sr
        self.crop = crop
        self.load_target = load_target
        self.extend_pedal = extend_pedal
        self.transform = transform
        self.target_transform = target_transform

        if not Path(self.root).exists():
            raise Exception(f"{self.root} does not exist. Please download the dataset from {MAESTRO.URL}")

        meta_csv = Path(self.root, "maestro-v3.0.0.csv")

        self.meta_dict = self.load_meta(meta_csv)
        
    def __getitem__(self, index: int) -> dict:

        audio_path = self.meta_dict["audio_path"][index]
        midi_path = self.meta_dict["midi_path"][index]

        full_data = {
            "dataset_name": "MAESTRO",
            "audio_path": audio_path,
            "midi_path": midi_path
        }

        # Load audio data
        audio_data = self.load_audio_data(path=audio_path)
        full_data.update(audio_data)

        # Load target data
        if self.load_target:
            target_data = self.load_target_data(
                midi_path=midi_path,
                start_time=audio_data["start_time"],
                duration=audio_data["duration"]
            )
            full_data.update(target_data)

        return full_data

    def __len__(self) -> int:

        audios_num = len(self.meta_dict["audio_name"])

        return audios_num

    def load_meta(self, meta_csv: str) -> dict:
        r"""Load meta dict.
        """

        df = pd.read_csv(meta_csv, sep=',')

        indexes = df["split"].values == self.split

        audio_names = df["audio_filename"].values[indexes]
        midi_names = df["midi_filename"].values[indexes]
        durations = df["duration"].values[indexes]

        midi_paths = [str(Path(self.root, name)) for name in midi_names]
        audio_paths = [str(Path(self.root, name)) for name in audio_names]

        meta_dict = {
            "audio_name": audio_names,
            "audio_path": audio_paths,
            "midi_name": midi_names,
            "midi_path": midi_paths,
            "duration": durations
        }

        return meta_dict

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

    def load_target_data(
        self, 
        midi_path: str, 
        start_time: float, 
        duration: float
    ) -> dict:

        notes, pedals = read_single_track_midi(
            midi_path=midi_path, 
            extend_pedal=self.extend_pedal,
        )

        notes = clip_notes(notes, start_time, duration)
        pedals = clip_notes(pedals, start_time, duration)

        target = {
            "note": notes,
            "pedal": pedals,
            "start_time": start_time, 
            "duration": duration,
        }
        
        # Transform target
        if self.target_transform:
            target = call(transform=self.target_transform, x=target)

        return target