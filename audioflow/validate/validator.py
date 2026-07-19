import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from audioflow.datasets import get_dataset
from audioflow.decoders.audio import load_decoder as load_audio_decoder
from audioflow.inference.inference import sample_latent
from audioflow.solvers import get_solver
from audioflow.utils.json import read_jsonl
from audioflow.utils.misc import logmel
from audioflow.utils.xml import xml_to_text
from audioflow.utils.torch import to_device, trim_target_latent


class Validator:
    def __init__(self, configs: dict, model: nn.Module, device: torch.device) -> None:
        
        self.configs = configs
        self.model = model
        self.device = device

        self.dataset = get_dataset(configs["dataset"])
        self.cfg_scale = configs["solver"]["cfg_scale"]
        self.solver = get_solver(configs["solver"])

        self.in_decoder = None
        self.out_decoder = None

        self.in_modality = configs["validate"]["decode_input"]
        self.out_modality = configs["validate"]["decode_output"]
        
    def __call__(self, split: str, out_dir: str) -> None:

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for json_dict in self.configs["validate"][split]:
            jsonl_path = json_dict["path"]
            n_valid = json_dict["num"]
            
            metas = read_jsonl(jsonl_path)
            indices = np.linspace(0, len(metas) - 1, n_valid, dtype=int)
            metas = [metas[i] for i in indices]
            
            for i in range(len(metas)):
                meta = metas[i]

                # Lazy initialize decoders
                if self.in_decoder is None:
                    if self.in_modality in ["audio"]:
                        self.in_decoder = load_audio_decoder(meta["input"]["feature"]["type"]).to(self.device)

                if self.out_decoder is None:
                    if self.out_modality in ["audio"]:
                        self.out_decoder = load_audio_decoder(meta["target"]["latent"]["type"]).to(self.device)
                
                # Get data
                meta["start_time"] = max(meta["target"]["latent"]["duration"] - self.dataset.clip_dur, 0.) / 2
                data = self.dataset[meta]
                data = default_collate([data])  # list to batch
                data = trim_target_latent(data)  # Cut silense
                data = to_device(data, self.device)
                
                x_in = data["input_feature"] if meta["input"].get("feature") else None
                x_real = data["target"]  # (1, t, d)

                # Generate
                noise = torch.randn_like(x_real)  # (1, l, d)
                x_gen = sample_latent(self.model, noise, data, self.solver, self.cfg_scale)
                
                # Names
                name = f"{split},idx={i}," + xml_to_text(data["text"][0], sep=",")
                name = name[0 : 150]

                # Save results
                if self.in_decoder is not None:
                    if self.in_modality in ["audio"]:
                        audio_in = self.in_decoder.decode(x_in).cpu().numpy()[0]  # (c, l)
                        self.write_audio(audio_in, path=out_dir / f"{name},in.wav", sr=self.in_decoder.sr)

                    elif self.in_modality in ["video"]:
                        video_in = self.in_decoder.decode(x_in).cpu().numpy()[0]  # (c, l)
                else:
                    if x_in is not None:
                        self.write_hdf5(x_in.cpu().numpy()[0], meta["input"]["feature"]["type"], path=out_dir / f"{name},in.h5")

                if self.out_decoder is not None:
                    if self.out_modality in ["audio"]:
                        audio_gen = self.out_decoder.decode(x_gen).cpu().numpy()[0]  # (c, l)
                        audio_gt = self.out_decoder.decode(x_real).cpu().numpy()[0]  # (c, l)
                        self.write_audio(audio_gen, path=out_dir / f"{name},gen.wav", sr=self.out_decoder.sr)
                        self.write_audio(audio_gt, path=out_dir / f"{name},gt.wav", sr=self.out_decoder.sr)
                else:
                    self.write_hdf5(x_gen.cpu().numpy()[0], meta["target"]["latent"]["type"], path=out_dir / f"{name},gen.h5")
                    self.write_hdf5(x_real.cpu().numpy()[0], meta["target"]["latent"]["type"], path=out_dir / f"{name},gt.h5")
    
                # Plot
                fig, axs = plt.subplots(3, 1, figsize=(10, 10))

                if self.in_modality in ["audio"]:
                    logmel_in = logmel(audio_in, self.out_decoder.sr)
                    self.plot_logmel(axs[0], logmel_in)
                else:
                    axs[0].matshow(x_in.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
                    
                if self.out_modality in ["audio"]:
                    logmel_gen = logmel(audio_gen, self.out_decoder.sr)
                    logmel_gt = logmel(audio_gt, self.out_decoder.sr)
                    self.plot_logmel(axs[1], logmel_gen)
                    self.plot_logmel(axs[2], logmel_gt)

                axs[0].set_title("Input")
                axs[1].set_title("Generation")
                axs[2].set_title("Ground truth")
                axs[2].xaxis.tick_bottom()

                out_path = out_dir / f"{name}.png"
                plt.savefig(out_path)
                print(f"Write out to {out_path}")


    def plot_logmel(self, ax, x):
        vmin, vmax = -10, 5
        if x is not None:
            ax.matshow(x.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)

    def write_audio(self, audio: np.ndarray, path: str, sr) -> None:
        if audio is not None:
            soundfile.write(file=path, data=audio.T, samplerate=sr)
            print(f"Write out to {path}")

    def write_hdf5(self, data: np.ndarray, name: str, path: str) -> None:
        if data is not None:
            with h5py.File(path, 'w') as hf:
                hf.create_dataset("data", data=data, dtype=np.float32)
                hf.attrs.create("type", data=name)
            print(f"Write out to {path}")
