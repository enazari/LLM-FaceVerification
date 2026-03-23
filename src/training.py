"""Shared training infrastructure for all tracks."""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
import torch.optim as optim
import torch.utils.checkpoint as _ckpt_mod
from accelerate import Accelerator
import yaml

from src.utils import cosine_lr, save_checkpoint, apply_overrides

# Force use_reentrant=False for gradient checkpointing (LoRA compatibility)
_orig_ckpt = _ckpt_mod.checkpoint
def _patched_ckpt(*args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _orig_ckpt(*args, **kwargs)
_ckpt_mod.checkpoint = _patched_ckpt


def parse_training_args(default_config: str):
    """Parse CLI args shared by all training scripts."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop after N gradient steps (0=full epoch)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip LFW/CFP evaluation after training")
    parser.add_argument("--override", nargs="*", default=[],
                        help="Config overrides: key.path=value")
    args = parser.parse_args()

    with open(f"configs/{args.config}.yaml") as f:
        cfg = yaml.safe_load(f)
    if args.override:
        apply_overrides(cfg, args.override)

    return args, cfg


def setup_accelerator(cfg: dict) -> Accelerator:
    """Create Accelerator from config.

    When launched via srun (SLURM-native), maps SLURM env vars to the
    PyTorch distributed env vars that Accelerate expects.
    """
    if "SLURM_PROCID" in os.environ and "RANK" not in os.environ:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    return Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "no"),
        gradient_accumulation_steps=grad_accum,
    )


def setup_output_dir(cfg: dict, config_name: str, accelerator: Accelerator) -> str:
    """Create output directory and copy config. Appends HHMM timestamp."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%H%M")
    output_dir = f"sessions/{cfg['session']}_{timestamp}"
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(f"configs/{config_name}.yaml", output_dir)
    return output_dir


def build_optimizer(params, cfg: dict):
    """Build optimizer from config."""
    opt_name = cfg["training"].get("optimizer", "sgd")
    if opt_name == "adamw":
        return optim.AdamW(
            params, lr=cfg["training"]["lr"],
            weight_decay=cfg["training"].get("weight_decay", 0.01),
        )
    return optim.SGD(
        params, lr=cfg["training"]["lr"],
        momentum=cfg["training"].get("momentum", 0.9),
        weight_decay=cfg["training"].get("weight_decay", 0.0005),
    )


def resume_checkpoint(backbone, optimizer, output_dir, accelerator):
    """Load last checkpoint if it exists. Returns start_epoch."""
    ckpt_path = os.path.join(output_dir, "last.pth")
    if not os.path.exists(ckpt_path):
        return 1

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    backbone.load_state_dict(ckpt["backbone"])
    start_epoch = ckpt["epoch"] + 1

    backbone, optimizer = accelerator.prepare(backbone, optimizer)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        if accelerator.is_main_process:
            print(f"Resumed from epoch {ckpt['epoch']}")

    return start_epoch


def run_training_loop(*, accelerator, backbone, optimizer, train_loader,
                      cfg, args, output_dir, train_epoch_fn, start_epoch=1):
    """Run the training loop with LR scheduling, CSV logging, and checkpointing.

    Args:
        train_epoch_fn: callable(backbone, loader, optimizer, accelerator, max_steps)
            → (loss, acc). Each script provides its own.
    """
    epochs = cfg["training"]["epochs"]
    warmup = cfg["training"].get("warmup_epochs", 2)
    base_lr = cfg["training"]["lr"]

    csv_file = None
    writer = None
    resuming = start_epoch > 1
    if accelerator.is_main_process:
        csv_path = os.path.join(output_dir, "metrics.csv")
        csv_file = open(csv_path, "a" if resuming else "w", newline="")
        writer = csv.writer(csv_file)
        if not resuming:
            writer.writerow(["epoch", "lr", "train_loss", "train_acc"])

    best_loss = float("inf")

    for epoch in range(start_epoch, epochs + 1):
        cosine_lr(optimizer, epoch - 1, epochs, warmup, base_lr)
        lr = optimizer.param_groups[0]['lr']
        if accelerator.is_main_process:
            print(f"Epoch {epoch}/{epochs}  lr={lr:.5f}")

        if hasattr(train_loader.dataset, "resample"):
            train_loader.dataset.resample()

        train_loss, train_acc = train_epoch_fn(
            backbone, train_loader, optimizer, accelerator,
            max_steps=args.max_steps,
        )
        if accelerator.is_main_process:
            print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(accelerator, backbone, optimizer, epoch, output_dir, "best")

        if accelerator.is_main_process:
            writer.writerow([epoch, f"{lr:.6f}", f"{train_loss:.6f}", f"{train_acc:.6f}"])
            csv_file.flush()

        save_checkpoint(accelerator, backbone, optimizer, epoch, output_dir, "last")

    if csv_file:
        csv_file.close()
