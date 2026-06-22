import random

import h5py
import librosa
import numpy as np

from audio_flow.utils import get_audio_latent_length, sample_start_frame, load_audio_latent


class Midi2AudioDataset:
    def __init__(self, clip_duration: float):
        self.clip_duration = clip_duration

    def __getitem__(self, meta: dict) -> dict:
        r"""Get text to music data."""
        
        task = meta["task"]
        input_latent_path = meta["input"]["audio"]["latent_path"]
        target_latent_path = meta["target"]["audio"]["latent_path"]
        input_fps = meta["input"]["audio"]["fps"]
        target_fps = meta["target"]["audio"]["fps"]
        
        clip_frames = round(self.clip_duration * target_fps)
        total_frames = get_audio_latent_length(target_latent_path)
        start = sample_start_frame(total_frames, clip_frames)

        target_latent, target_mask, target_length = load_audio_latent(
            target_latent_path, start, clip_frames
        )  # target_latent: (l, d), target_mask: (l,)

        assert input_fps % target_fps == 0
        r = int(input_fps // target_fps)
        input_latent, _, _ = load_audio_latent(input_latent_path, start * r, clip_frames * r)
        # input_latent: (l, d)

        data = {
            "task": task,
            "input_latent": input_latent,  # (l, d)
            "target_latent": target_latent,  # (l, d)
            "target_mask": target_mask,  # (l,)
            "target_length": target_length
        }

        '''
        with h5py.File(input_latent_path, 'r') as hf:
            midi_latent = hf["latent"][:]  # (l, d)

        with h5py.File(target_latent_path, 'r') as hf:
            vae_latent = hf["latent"][:]  # (l, d)

        import matplotlib.pyplot as plt
        import pickle

        
        plt.figure()
        # plt.matshow(midi_latent[0:3000].T, origin='lower', aspect='auto', cmap='jet')
        plt.matshow(input_latent.T, origin='lower', aspect='auto', cmap='jet')
        plt.savefig("_zz.pdf")

        
        # pickle.dump(vae_latent[0:750], open("_zz.pkl", "wb"))
        pickle.dump(target_latent, open("_zz.pkl", "wb"))

        from IPython import embed; embed(using=False); os._exit(0)
        '''

        return data