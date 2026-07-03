from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from audioflow.datasets import get_dataset
from audioflow.flows import get_flow
from audioflow.guidance.cfg import cfg_drop
from audioflow.validate.validator import Validator
from audioflow.models import get_model
from audioflow.optim import get_optimizer_and_scheduler
from audioflow.optim.ema import update_ema
from audioflow.samplers import get_batch_sampler
from audioflow.utils.torch import (mean, requires_grad, save, to_device,
                                   trim_target_latent)
from audioflow.utils.yaml import read_yaml


def train(args) -> None:
    r"""Train audio generation with flow matching."""

    # Arguments
    config_path = Path(args.config)
    wandb_log = not args.no_log
    filename = Path(__file__).stem
    
    # Configs
    configs = read_yaml(config_path)
    device = configs["train"]["device"]
    ckpt_path = configs["train"]["resume_ckpt_path"]

    # Checkpoints directory
    config_name = config_path.stem
    ckpts_dir = Path("./checkpoints") / filename / config_name
    ckpts_dir.mkdir(parents=True, exist_ok=True)

    # Sampler
    batch_sampler = get_batch_sampler(configs)

    # Dataset
    train_dataset = get_dataset(configs["dataset"])

    # Dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_sampler=batch_sampler,
        num_workers=configs["train"]["num_workers"], 
        pin_memory=True,
    )

    # Model
    model = get_model(configs["model"], ckpt_path).to(device)

    # EMA
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    ema.eval()  # EMA model should always be in eval mode

    # Flow matcher
    fm = get_flow(configs["flow"])

    # Optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(
        configs=configs["train"]["optimizer"], 
        params=model.parameters()
    )

    # Validator
    validator = Validator(configs, ema, device)

    # Logger
    if wandb_log:
        wandb.init(project="audio_flow", name=f"{filename}_{config_name}")

    for step, data in enumerate(tqdm(train_dataloader)):

        # ------ 1. Data preparation ------
        # 1.1 CFG drop
        data = cfg_drop(
            data=data, 
            p_full=configs["cfg"]["train"]["p_full"], 
            p_partial=configs["cfg"]["train"]["p_partial"]
        )

        # 1.2 Trim data
        data = trim_target_latent(data)  # Cut to max len in a batch
        data = to_device(data, device)

        # 1.3 Sample noise
        x_real = data["target_latent"]
        noise = torch.randn_like(x_real)  # (b, ...)

        # 1.4 Sample t. Compute input, and velocity
        t, x, u = fm.sample(x0=noise, x1=x_real)  # t: (b,), x: (b, ...), u: (b, ...)

        # ------ 2. Training ------
        # 2.1 Forward
        model.train()
        v = model(t, x, data)  # (b, ...)

        # 2.2 Loss
        loss = mean((v - u) ** 2, mask=data["target_mask"])

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
                out_dir = Path("./results") / filename / config_name / f"steps={step}_ema"
                validator(split=split, out_dir=out_dir)

            if wandb_log:
                wandb.log(
                    data={
                        "train_loss": loss.item()
                    },
                    step=step
                )
        
        # 3.2 Save model
        if step % configs["train"]["save_every_n_steps"] == 0:
            ckpt_path = ckpts_dir / f"step={step}_ema.pth"
            save(ema, ckpt_path)
            print(f"Save model to {ckpt_path}")
        
        if step == configs["train"]["training_steps"]:
            break


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path of config yaml.")
    parser.add_argument("--no_log", action="store_true", default=False)
    args = parser.parse_args()

    train(args)