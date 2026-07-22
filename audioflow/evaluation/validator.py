import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import soundfile
import torch
import torch.nn as nn
from torch.utils.data._utils.collate import default_collate

from audioflow.data.datasets import get_dataset
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

        self.dataset = get_dataset(configs["data"]["dataset"])
        self.solver = get_solver(configs["sampling"]["solver"])
        self.n_valid = self.configs["validation"]["num"]
        self.cfg_scale = self.configs["sampling"]["cfg"]["scale"]

        self.decode_cond_bool = self.configs["validation"]["decode_condition"]
        self.decode_out_bool = self.configs["validation"]["decode_output"]
        self.cond_config = configs["non_text_condition"]["encoder"]
        self.out_config = configs["target"]["decoder"]

        # Lazy initialization
        self.cond_decoder = None
        self.out_decoder = None
        
    def __call__(self, split: str, out_dir: str) -> None:

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for json_dict in self.configs["data"]["validate"][split]:
            jsonl_path = json_dict["path"]
            metas = read_jsonl(jsonl_path)
            indices = np.linspace(0, len(metas) - 1, self.n_valid, dtype=int)
            metas = [metas[i] for i in indices]
            
            for i in range(len(metas)):
                meta = metas[i]

                # Lazy initialization
                if self.decode_cond_bool and (self.cond_decoder is None):
                    if self.cond_config["modality"] in ["audio"]:
                        self.cond_decoder = load_audio_decoder(self.cond_config["name"]).to(self.device)

                if self.decode_out_bool and (self.out_decoder is None):
                    if self.out_config["name"] == self.cond_config["name"]:
                        self.out_decoder = self.cond_decoder
                        
                    elif self.out_config["modality"] in ["audio"]:
                        self.out_decoder = load_audio_decoder(self.out_config["name"]).to(self.device)
                
                # Get data
                meta["start_time"] = max(meta["target"]["latent"]["duration"] - self.dataset.clip_dur, 0.) / 2
                data = self.dataset[meta]
                data = default_collate([data])  # list to batch
                data = trim_target_latent(data)  # Cut silense
                data = to_device(data, self.device)
                
                x_cond = data["input_feature"] if meta["input"].get("feature") else None
                x_real = data["target"]  # (1, t, d)

                # Generate
                noise = torch.randn_like(x_real)  # (1, l, d)
                x_gen = sample_latent(self.model, noise, data, self.solver, self.cfg_scale)
                
                # Names
                name = f"{split},idx={i}," + xml_to_text(data["text"][0], sep=",")
                name = name[0 : 150]

                # Save results
                if self.decode_cond_bool:
                    if self.cond_config["modality"] in ["audio"]:
                        audio_cond = self.cond_decoder.decode(x_cond).cpu().numpy()[0]  # (c, l)
                        self.write_audio(audio_cond, path=out_dir / f"{name},cond.wav", sr=self.cond_decoder.sr)

                    elif self.cond_config["modality"] in ["video"]:
                        video_cond = self.cond_decoder.decode(x_cond).cpu().numpy()[0]  # (c, l)
                else:
                    if x_cond is not None:
                        self.write_hdf5(x_cond.cpu().numpy()[0], meta["input"]["feature"]["type"], path=out_dir / f"{name},cond.h5")

                if self.decode_out_bool:
                    if self.out_config["modality"] in ["audio"]:
                        audio_gen = self.out_decoder.decode(x_gen).cpu().numpy()[0]  # (c, l)
                        audio_gt = self.out_decoder.decode(x_real).cpu().numpy()[0]  # (c, l)
                        self.write_audio(audio_gen, path=out_dir / f"{name},gen.wav", sr=self.out_decoder.sr)
                        self.write_audio(audio_gt, path=out_dir / f"{name},gt.wav", sr=self.out_decoder.sr)
                else:
                    self.write_hdf5(x_gen.cpu().numpy()[0], meta["target"]["latent"]["type"], path=out_dir / f"{name},gen.h5")
                    self.write_hdf5(x_real.cpu().numpy()[0], meta["target"]["latent"]["type"], path=out_dir / f"{name},gt.h5")
    
                # Plot
                fig, axes = plt.subplots(3, 1, figsize=(10, 10))
                [axes[i].xaxis.tick_bottom() for i in range(len(axes))]

                if self.decode_cond_bool:
                    if self.cond_config["modality"] in ["audio"]:
                        logmel_cond = logmel(audio_cond, self.out_decoder.sr)
                        self.plot_logmel(axes[0], logmel_cond)
                else:
                    axes[0].matshow(x_cond.cpu().numpy()[0].T, origin='lower', aspect='auto', cmap='jet')
                    
                if self.decode_out_bool:
                    if self.out_config["modality"] in ["audio"]:
                        logmel_gen = logmel(audio_gen, self.out_decoder.sr)
                        logmel_gt = logmel(audio_gt, self.out_decoder.sr)
                        self.plot_logmel(axes[1], logmel_gen)
                        self.plot_logmel(axes[2], logmel_gt)

                axes[0].set_title("Non-text Condition")
                axes[1].set_title("Generation")
                axes[2].set_title("Ground truth")

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
