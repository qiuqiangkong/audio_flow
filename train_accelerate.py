from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Literal

from accelerate import Accelerator
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
from train import (get_dataset, get_sampler, get_model, get_data_transform, 
    get_optimizer_and_scheduler, validate)


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    wandb_log = not args.no_log
    config_path = args.config
    filename = Path(__file__).stem
    
    # Configs
    configs = parse_yaml(config_path)

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
    data_transform = get_data_transform(configs)

    # Flow matching data processor
    fm = ConditionalFlowMatcher(sigma=0.)

    # Model
    model = get_model(
        configs=configs, 
        ckpt_path=configs["train"]["resume_ckpt_path"]
    )

    # EMA (optional)
    ema = deepcopy(model)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs, 
        params=model.parameters()
    )

    # Prepare for acceleration
    accelerator = Accelerator(mixed_precision=configs["train"]["precision"])

    data_transform, model, ema, optimizer, train_dataloader = accelerator.prepare(
        data_transform, model, ema, optimizer, train_dataloader)

    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    # Train
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
        accelerator.backward(loss)  # Update all parameter.grad
        optimizer.step()  # Update all parameters based on all parameter.grad
        update_ema(ema_model=ema, model=accelerator.unwrap_model(model))

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
                    data_transform=accelerator.unwrap_model(data_transform),
                    model=accelerator.unwrap_model(model),
                    split=split,
                    out_dir=Path("results", filename, config_name, f"steps={step}")
                )

            for split in ["train", "test"]:
                validate(
                    configs=configs,
                    data_transform=accelerator.unwrap_model(data_transform),
                    model=accelerator.unwrap_model(model),
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
            torch.save(accelerator.unwrap_model(model).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

            ckpt_path = Path(ckpts_dir, "step={}_ema.pth".format(step))
            torch.save(accelerator.unwrap_model(ema).state_dict(), ckpt_path)
            print("Save model to {}".format(ckpt_path))

        if step == configs["train"]["training_steps"]:
            break

        step += 1
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)