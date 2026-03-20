"""Shared training utilities."""

import math
import os
from pathlib import Path

import torch


def cosine_lr(optimizer, epoch: int, total_epochs: int, warmup_epochs: int,
              base_lr: float) -> None:
    """Cosine annealing with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for g in optimizer.param_groups:
        g['lr'] = lr


def save_checkpoint(accelerator, backbone, optimizer, epoch, output_dir, name,
                    head=None):
    """Save model checkpoint (main process only)."""
    if not accelerator.is_main_process:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pth")
    state = {
        "backbone": accelerator.unwrap_model(backbone).state_dict(),
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if head is not None:
        state["head"] = accelerator.unwrap_model(head).state_dict()
    torch.save(state, path)
    print(f"  saved → {path}")
