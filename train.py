from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

import wandb
from audio_flow.samplers.jsonl_sampler import BatchJsonlSampler
# from audio_flow.datasets.dataset import MetaDataset
# from audio_flow.encoders.audio.levo_vae import LevoVAE

from audio_flow.utils import (load_vae, CombinedModel, LinearWarmUp, get_single_value,
                              load_jsonl, truncate_latent, to_device, logmel, parse_yaml, requires_grad,
                              update_ema, mean_pool, get_saveable_state_dict)
from audio_flow.solvers.euler import euler_solver


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Sampler
    batch_sampler = get_batch_sampler(configs)

    # Dataset
    train_dataset = get_dataset(configs)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=batch_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    # Model
    model = get_model(configs, ckpt_path).to(device)

    # VAE for validation
    # vae = load_vae("levo_vae").to(device)
    vae = load_vae(configs["validate"]["vae"]).to(device)

    # EMA (optional)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)
    
    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")
    
    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Data
        data = truncate_latent(data)
        data = to_device(data, device)

        x_real = data["target_latent"]
        noise = torch.randn_like(x_real)

        # 1.2 Get input and velocity
        t, xt, ut = fm.sample_location_and_conditional_flow(x0=noise, x1=x_real)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        controls = model.adapter(data)
        vt = model.base(t=t, x=xt, controls=controls)

        # 2.2 Loss
        loss = mean_pool((vt - ut) ** 2, data["target_mask"]).mean()

        # 2.3 Optimize
        optimizer.zero_grad()  # Reset all parameter.grad to 0
        loss.backward()  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        update_ema(ema, model, decay=0.999)

        # 2.4 Learning rate scheduler
        if scheduler:
            scheduler.step()

        if step % 100 == 0:
            print("train loss: {:.4f}".format(loss.item()))
        
        # ------ 3. Evaluation ------
        # 3.1 Evaluate
        if step % configs["train"]["test_every_n_steps"] == 0:

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    model=ema,
                    vae=vae,
                    split=split,
                    out_dir=Path("./results", filename, config_name, f"steps={step}_ema"),
                )

            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
           
            ckpt_path = Path(ckpts_dir, f"step={step}_ema.pth")
            torch.save(get_saveable_state_dict(ema), ckpt_path)
            print(f"Save model to {ckpt_path}")

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        


def get_dataset(configs: dict) -> Dataset:
    r"""Get dataset."""
    name = configs["dataset"]["name"]
    
    if name == "MetaDataset":
        return MetaDataset(configs["clip_duration"])

    elif name == "TTMDataset":
        from audio_flow.datasets.ttm import TTMDataset
        return TTMDataset(configs["clip_duration"])

    elif name == "TTSDataset":
        from audio_flow.datasets.tts import TTSDataset
        return TTSDataset(configs["clip_duration"])

    elif name == "TTADataset":
        from audio_flow.datasets.tta import TTADataset
        return TTADataset(configs["clip_duration"])

    elif name == "MSSDataset":
        from audio_flow.datasets.mss import MSSDataset
        return MSSDataset(configs["clip_duration"])

    elif name == "Vocals2MusicDataset":
        from audio_flow.datasets.vocals2music import Vocals2MusicDataset
        return Vocals2MusicDataset(configs["clip_duration"])

    elif name == "Mono2StereoDataset":
        from audio_flow.datasets.mono2stereo import Mono2StereoDataset
        return Mono2StereoDataset(configs["clip_duration"])

    elif name == "SuperResolutionDataset":
        from audio_flow.datasets.superresolution import SuperResolutionDataset
        return SuperResolutionDataset(configs["clip_duration"])

    elif name == "Codec2AudioDataset":
        from audio_flow.datasets.codec2audio import Codec2AudioDataset
        return Codec2AudioDataset(configs["clip_duration"])

    elif name == "Midi2AudioDataset":
        from audio_flow.datasets.midi2audio import Midi2AudioDataset
        return Midi2AudioDataset(configs["clip_duration"])

    elif name == "EditingDataset":
        from audio_flow.datasets.editing import EditingDataset
        return EditingDataset(configs["clip_duration"])

    elif name == "Video2AudioDataset":
        from audio_flow.datasets.video2audio import Video2AudioDataset
        return Video2AudioDataset(configs["clip_duration"])

    elif name == "Video2AudioMaeDataset":
        from audio_flow.datasets.video2audio import Video2AudioMaeDataset
        return Video2AudioMaeDataset(configs["clip_duration"])

    else:
        raise ValueError(name)


def get_batch_sampler(configs: dict) -> Iterable:
    r"""Get sampler."""
    name = configs["sampler"]["name"]
    batch_size = configs["train"]["batch_size_per_device"]

    if name == "BatchJsonlSampler":
        paths = [meta["path"] for meta in configs["train_jsonls"]]
        weights = [meta["weight"] for meta in configs["train_jsonls"]]
        return BatchJsonlSampler(paths, weights, batch_size)

    else:
        raise ValueError(name)


def get_model(configs: dict, ckpt_path: str) -> nn.Module:
    base = get_base(configs=configs)
    adapter = get_adapter(configs=configs)
    model = CombinedModel(base, adapter)

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt, strict=False)
        print(f"Load checkpoint from {ckpt_path}")

    return model


def get_base(
    configs: dict, 
) -> nn.Module:
    r"""Initialize base model."""
    name = configs["base"]["name"]

    if name == "Transformer":
        from audio_flow.models.transformer import Transformer
        return Transformer(**configs["base"])

    elif name == "Transformer2":
        from audio_flow.models.transformer2 import Transformer2
        return Transformer2(**configs["base"])

    elif name == "Transformer03a":
        from audio_flow.models.transformer_cfg import Transformer03a
        return Transformer03a(**configs["base"])

    else:
        raise ValueError(name)    


def get_adapter(
    configs: dict, 
) -> nn.Module:
    r"""Initialize adapter."""
    name = configs["adapter"]["name"]
    
    if name == "Adapter":
        from audio_flow.adapters.adapter import Adapter
        return Adapter(**configs["adapter"])

    elif name == "TTMAdapter":
        from audio_flow.adapters.ttm import TTMAdapter
        return TTMAdapter(**configs["adapter"])

    elif name == "TTSAdapter":
        from audio_flow.adapters.tts import TTSAdapter
        return TTSAdapter(**configs["adapter"])

    elif name == "TTAAdapter":
        from audio_flow.adapters.tta import TTAAdapter
        return TTAAdapter(**configs["adapter"])

    elif name == "TTAAdapter2":
        from audio_flow.adapters.tta2 import TTAAdapter2
        return TTAAdapter2(**configs["adapter"])

    elif name == "MSSAdapter":
        from audio_flow.adapters.mss import MSSAdapter
        return MSSAdapter(**configs["adapter"])

    elif name == "Vocals2MusicAdapter":
        from audio_flow.adapters.vocals2music import Vocals2MusicAdapter
        return Vocals2MusicAdapter(**configs["adapter"])

    elif name == "Mono2StereoAdapter":
        from audio_flow.adapters.mono2stereo import Mono2StereoAdapter
        return Mono2StereoAdapter(**configs["adapter"])

    elif name == "SuperResolutionAdapter":
        from audio_flow.adapters.superresolution import SuperResolutionAdapter
        return SuperResolutionAdapter(**configs["adapter"])

    elif name == "Codec2AudioAdapter":
        from audio_flow.adapters.codec2audio import Codec2AudioAdapter
        return Codec2AudioAdapter(**configs["adapter"])

    elif name == "Midi2AudioAdapter":
        from audio_flow.adapters.midi2audio import Midi2AudioAdapter
        return Midi2AudioAdapter(**configs["adapter"])

    elif name == "EditingAdapter":
        from audio_flow.adapters.editing import EditingAdapter
        return EditingAdapter(**configs["adapter"])

    elif name == "Video2AudioAdapter":
        from audio_flow.adapters.video2audio import Video2AudioAdapter
        return Video2AudioAdapter(**configs["adapter"])

    elif name == "Video2AudioMaeAdapter":
        from audio_flow.adapters.video2audio import Video2AudioMaeAdapter
        return Video2AudioMaeAdapter(**configs["adapter"])


    else:
        raise ValueError(name)    


def get_optimizer_and_scheduler(
    configs: dict, 
    params: list[torch.Tensor]
) -> tuple[optim.Optimizer, None | optim.lr_scheduler.LambdaLR]:
    r"""Get optimizer and scheduler."""

    lr = float(configs["train"]["lr"])
    warm_up_steps = configs["train"]["warm_up_steps"]
    optimizer_name = configs["train"]["optimizer"]

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(params=params, lr=lr)

    if warm_up_steps:
        lr_lambda = LinearWarmUp(warm_up_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    return optimizer, scheduler


def validate(
    configs: dict,
    model: nn.Module,
    vae: nn.Module,
    split: Literal["train", "test"],
    out_dir: str
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = configs[f"validate_{split}_jsonl"]["path"]
    n_valid = configs[f"validate_{split}_jsonl"]["num"]
    metas = load_jsonl(jsonl_path)

    skip_n = max(1, len(metas) // n_valid)
    metas = metas[0 :: skip_n]

    # Dataset
    dataset = get_dataset(configs)
    
    for i in range(len(metas)):

        # ------ 1. Data preparation ------
        # 1.1 Get Data
        data = dataset[metas[i]]
        data = default_collate([data])
        data = truncate_latent(data)
        data = to_device(data, device)

        # 2.1 Sample noise
        x_real = data["target_latent"]  # (b, t, d)
        noise = torch.randn_like(x_real)  # (b, l, d)

        # ------ 2. Forward with ODE ------
        # 2.1 Iteratively forward
        try:
            with torch.no_grad():
                model.eval()
                controls = model.adapter(data)
                x_gen = euler_solver(model.base, noise, controls, n_steps=100)  # (b, l, d)
        except:
            from IPython import embed; embed(using=False); os._exit(0)

        # Decode audio from VAE latents
        audio_gen = vae.decode(x_gen).data.cpu().numpy()[0]  # (c, l)
        audio_gt = vae.decode(x_real).data.cpu().numpy()[0]  # (c, l)
        
        task = data["task"][0]
        if task in ["music source separation", "vocals to music", 
            "mono to stereo", "super-resolution", "codec to music"]:
            x_in = data["input_latent"].to(device)
            audio_in = vae.decode(x_in).data.cpu().numpy()[0]  # (c, l)
        else:
            audio_in = None

        # ------ 3. Plot and Visualization ------
        # 3.1 Plot mel spectrogram
        if audio_in is not None:
            logmel_in = logmel(audio_in, vae.sr)
        logmel_gen = logmel(audio_gen, vae.sr)
        logmel_gt = logmel(audio_gt, vae.sr)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10))
        vmin, vmax = -10, 5

        if audio_in is not None:
            axs[0].matshow(logmel_in.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)

        elif task in ["midi to audio"]:
            x_in = data["input_latent"].cpu().numpy()[0]
            axs[0].matshow(x_in.T, origin='lower', aspect='auto', cmap='jet', vmin=0., vmax=1.)

        axs[1].matshow(logmel_gen.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2].matshow(logmel_gt.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title("Input")
        axs[1].set_title("Generation")
        axs[2].set_title("Ground truth")
        axs[2].xaxis.tick_bottom()

        strs = [split, f"idx={i}"]
        for key in ["task", "prompt"]:
            if key in data.keys():
                text = get_single_value(data[key])[0 : 150]
                strs.append("{}={}".format(key, text))
        stem = ",".join(strs)
        
        if task in ["video to audio"]:
            stem += ",id={}".format(Path(data["input_latent_path"][0]).stem)

        out_path = Path(out_dir, stem + ".png")
        plt.savefig(out_path)
        print(f"Write out to {out_path}")

        # 3.2 Save audio
        if audio_in is not None:
            out_path = Path(out_dir, stem + ",input.wav")
            soundfile.write(file=out_path, data=audio_in.T, samplerate=vae.sr)
            print(f"Write out to {out_path}") 

        out_path = Path(out_dir, stem + ",gen.wav")
        soundfile.write(file=out_path, data=audio_gen.T, samplerate=vae.sr)
        print(f"Write out to {out_path}")

        out_path = Path(out_dir, stem + ",gt.wav")
        soundfile.write(file=out_path, data=audio_gt.T, samplerate=vae.sr)
        print(f"Write out to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)