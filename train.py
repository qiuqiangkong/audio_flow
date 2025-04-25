from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

import matplotlib.pyplot as plt
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torchdiffeq
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from tqdm import tqdm

import wandb
from audio_flow.utils import LinearWarmUp, Logmel, parse_yaml, requires_grad, update_ema


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)
    device = configs["train"]["device"]

    # Checkpoints directory
    config_name = Path(config_path).stem
    ckpts_dir = Path("./checkpoints", filename, config_name)
    Path(ckpts_dir).mkdir(parents=True, exist_ok=True)

    # Datasets
    train_dataset = get_dataset(configs, split="train", mode="train")

    # Sampler
    train_sampler = get_sampler(configs, train_dataset)

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=configs["train"]["batch_size_per_device"], 
        sampler=train_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    # Data processor
    data_transform = get_data_transform(configs).to(device)

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    ).to(device)

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

    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 Transform data into latent representations and conditions
        target, cond_dict, _ = data_transform(data)

        # 1.2 Noise
        noise = torch.randn_like(target)

        # 1.3 Get input and velocity
        t, xt, ut = fm.sample_location_and_conditional_flow(x0=noise, x1=target)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        vt = model(t=t, x=xt, cond_dict=cond_dict)

        # 2.2 Loss
        loss = torch.mean((vt - ut) ** 2)

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
                    data_transform=data_transform,
                    model=model,
                    split=split,
                    out_dir=Path("results", filename, config_name, f"steps={step}")
                )

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    data_transform=data_transform,
                    model=ema,
                    split=split,
                    out_dir=Path("results", filename, config_name, f"steps={step}_ema")
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
            
            ckpt_path = Path(ckpts_dir, "step={}.pth".format(step))
            torch.save(model.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

            ckpt_path = Path(ckpts_dir, "step={}_ema.pth".format(step))
            torch.save(ema.state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        

def get_data_transform(configs: dict):
    r"""Transform data into latent representations and conditions."""

    name = configs["data_transform"]["name"]

    if name == "Text2Music_Mel":
        from audio_flow.data_transforms.text2music import Text2Music_Mel
        return Text2Music_Mel()

    elif name == "Codec2Audio_Mel":
        from audio_flow.data_transforms.codec2audio import Codec2Audio_Mel
        return Codec2Audio_Mel()

    elif name == "SuperResolution_Mel":
        from audio_flow.data_transforms.superresolution import \
            SuperResolution_Mel
        return SuperResolution_Mel(
            sr=configs["sample_rate"], 
            distorted_sr=configs["data_transform"]["distorted_sample_rate"]
        )

    elif name == "Mono2Stereo_Mel":
        from audio_flow.data_transforms.mono2stereo import Mono2Stereo_Mel
        return Mono2Stereo_Mel()

    elif name == "Midi2Audio_Mel":
        from audio_flow.data_transforms.midi2audio import Midi2Audio_Mel
        return Midi2Audio_Mel()

    elif name == "MSS_Mel":
        from audio_flow.data_transforms.mss import MSS_Mel
        return MSS_Mel()

    elif name == "Vocal2Music_Mel":
        from audio_flow.data_transforms.vocal2music import Vocal2Music_Mel
        return Vocal2Music_Mel()

    elif name == "Image2Audio_Mel":
        pass

    elif name == "Phase_STFT":
        pass

    else:
        raise ValueError(name)


def get_dataset(
    configs: dict, 
    split: Literal["train", "test"],
    mode: Literal["train", "test"]
) -> Dataset:
    r"""Get datasets."""

    sr = configs["sample_rate"]
    clip_duration = configs["clip_duration"]
    ds = f"{split}_datasets"

    for name in configs[ds].keys():

        if name == "GTZAN":

            from audidata.io.crops import RandomCrop, StartCrop
            from audio_flow.datasets.gtzan import GTZAN
            
            if mode == "train":
                crop = RandomCrop(clip_duration=clip_duration)
            elif mode == "test":
                crop = StartCrop(start=0., clip_duration=clip_duration)

            dataset = GTZAN(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                test_fold=0,
                sr=configs["sample_rate"],
                crop=RandomCrop(clip_duration=configs["clip_duration"])
            )
            return dataset
    
        elif name == "MUSDB18HQ":

            from audidata.io.crops import RandomCrop, StartCrop
            from audio_flow.datasets.musdb18hq import MUSDB18HQ
            
            if mode == "train":
                crop = RandomCrop(clip_duration=clip_duration)
            elif mode == "test":
                crop = StartCrop(start=60., clip_duration=clip_duration)

            dataset = MUSDB18HQ(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=crop,
                time_align=configs[ds][name]["time_align"],
                mixture_transform=None,
                group_transform=None,
                stem_transform=None
            )
            return dataset

        elif name == "MAESTRO":

            from audidata.io.crops import RandomCrop, StartCrop
            from audidata.transforms.midi import PianoRoll
            from audio_flow.datasets.maestro import MAESTRO
            from audio_flow.update_collate import default_collate_fn_map  # Change global variable

            if mode == "train":
                crop = RandomCrop(clip_duration=clip_duration)
            elif mode == "test":
                crop = StartCrop(start=60., clip_duration=clip_duration)

            dataset = MAESTRO(
                root=configs[ds][name]["root"],
                split=configs[ds][name]["split"],
                sr=sr,
                crop=crop,
                transform=None,
                load_target=True,
                extend_pedal=True,
                target_transform=PianoRoll(fps=100, pitches_num=128),
            )
            return dataset

        else:
            raise ValueError(name)

    else:
        raise ValueError("Do not support multiple datasets.")


def get_sampler(configs: dict, dataset: Dataset) -> Iterable:
    r"""Get sampler."""

    name = configs["sampler"]

    if name == "InfiniteSampler":
        from audio_flow.samplers.infinite_sampler import InfiniteSampler
        return InfiniteSampler(dataset)

    else:
        raise ValueError(name)


def get_model(
    configs: dict, 
    ckpt_path: str
) -> nn.Module:
    r"""Initialize model."""

    name = configs["model"]["name"]

    if name == "BSRoformerMel": 

        from audio_flow.models.bsroformer_mel import BSRoformerMel, Config

        config = Config(**configs["model"])
        model = BSRoformerMel(config)

    else:
        raise ValueError(name)    

    if ckpt_path:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt)

    return model


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
    data_transform: object,
    model: nn.Module,
    split: Literal["train", "test"],
    out_dir: str
) -> float:
    r"""Validate the model on part of data."""

    device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = configs["sample_rate"]
    valid_audios = configs["valid_audios"]

    dataset = get_dataset(configs, split=split, mode="test")

    # Evaluate only part of data
    if valid_audios:
        skip_n = max(1, len(dataset) // valid_audios)
    else:
        skip_n = 1

    for idx in range(0, len(dataset), skip_n):

        # ------ 1. Data preparation ------
        # 1.1 Get Data
        data = dataset[idx]
        data = default_collate([data])
        
        # 1.2 Transform data into latent representations and conditions
        target, cond_dict, cond_sources = data_transform(data)

        # 1.3 Noise
        noise = torch.randn_like(target)

        # ------ 2. Forward with ODE ------
        # 2.1 Iteratively forward
        with torch.no_grad():
            model.eval()
            traj = torchdiffeq.odeint(
                lambda t, x: model.forward(t, x, cond_dict),
                y0=noise,
                t=torch.linspace(0, 1, 2, device=device),
                atol=1e-4,
                rtol=1e-4,
                method="dopri5",
            )

        est_target = traj[-1]  # (b, c, t, f)

        # 2.2 Latent to audio
        est_audio = data_transform.latent_to_audio(est_target).data.cpu().numpy()  # (b, c, l)
        gt_audio = data_transform.latent_to_audio(target).data.cpu().numpy()  # (b, c, l)

        # ------ 3. Plot and Visualization ------
        # 3.1 Plot logmel spectrogram
        logmel_extractor = Logmel(sr=sr)
        est_logmel = logmel_extractor(est_audio[0, 0])
        gt_logmel = logmel_extractor(gt_audio[0, 0])

        if "caption" in cond_sources.keys():
            caption = cond_sources["caption"][0]
        else:
            caption = ""

        print(f"caption: {caption}")

        if "audio" in cond_sources.keys():
            cond_audio = cond_sources["audio"].data.cpu().numpy()  # (b, c, l)
            cond_logmel = logmel_extractor(cond_audio[0, 0])

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        vmin, vmax = -10, 5
        if "audio" in cond_sources.keys():
            axs[0].matshow(cond_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        if "image" in cond_sources.keys():
            image = cond_sources["image"].data.cpu().numpy()[0]
            axs[0].matshow(image.T, origin='lower', aspect='auto', cmap='jet')
        axs[1].matshow(est_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[2].matshow(gt_logmel.T, origin='lower', aspect='auto', cmap='jet', vmin=vmin, vmax=vmax)
        axs[0].set_title("Cond TF (if there are)")
        axs[1].set_title("Estimation")
        axs[2].set_title("Ground truth")
        axs[2].xaxis.tick_bottom()
        
        out_path = Path(out_dir, f"{split}_{idx}_{caption}.png")
        plt.savefig(out_path)
        print(f"Write out to {out_path}")

        # 3.2 Save audio
        if "audio" in cond_sources.keys():
            out_path = Path(out_dir, f"{split}_{idx}_cond_{caption}.wav")
            soundfile.write(file=out_path, data=cond_audio[0].T, samplerate=sr)
            print(f"Write out to {out_path}")

        out_path = Path(out_dir, f"{split}_{idx}_est_{caption}.wav")
        soundfile.write(file=out_path, data=est_audio[0].T, samplerate=sr)
        print(f"Write out to {out_path}")

        out_path = Path(out_dir, f"{split}_{idx}_gt_{caption}.wav")
        soundfile.write(file=out_path, data=gt_audio[0].T, samplerate=sr)
        print(f"Write out to {out_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)