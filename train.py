"""Training script for face verification (embedding-based, Tracks 1 & 3)."""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.prepare import prepare, lmdb_paths
from src.backbones.factory import build_backbone
from src.data.dataset import LMDBFaceDataset
from src.data.sampler import PairSampler
from src.losses.factory import build_loss
from src.utils import cosine_lr, save_checkpoint


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(backbone, loss_fn, loader, optimizer, accelerator):
    backbone.train()
    total_loss = 0.0
    correct = 0
    total = 0

    params = list(backbone.parameters())

    for imgs, labels in tqdm(loader, desc="  train", leave=False,
                             disable=not accelerator.is_main_process):
        with accelerator.accumulate(backbone):
            optimizer.zero_grad()
            emb = backbone(imgs)
            emb_norm = F.normalize(emb, dim=1)

            loss = loss_fn(emb_norm, labels)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            # Pair accuracy: is the positive partner the top-1 nearest?
            with torch.no_grad():
                sim = emb_norm @ emb_norm.T
                sim.fill_diagonal_(float('-inf'))
                B = emb_norm.size(0)
                targets = torch.arange(B, device=sim.device)
                targets[0::2] = torch.arange(1, B, 2, device=sim.device)
                targets[1::2] = torch.arange(0, B, 2, device=sim.device)
                correct += (sim.argmax(1) == targets).sum().item()

            total_loss += loss.item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="internvit-lora")
    args = parser.parse_args()

    with open(f"configs/{args.config}.yaml") as f:
        cfg = yaml.safe_load(f)

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "no"),
        gradient_accumulation_steps=grad_accum,
    )
    accelerator.even_batches = False

    output_dir = f"sessions/{cfg['session']}"
    epochs = cfg["training"]["epochs"]
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(f"configs/{args.config}.yaml", output_dir)

    backbone = build_backbone(cfg)

    if epochs == 0:
        # Eval-only
        backbone = accelerator.prepare(backbone)
        save_checkpoint(accelerator, backbone, None, 0, output_dir, "best")
    else:
        # Prepare data
        if accelerator.is_main_process:
            prepare(cfg)
        accelerator.wait_for_everyone()

        train_path, _ = lmdb_paths(cfg)
        batch_size = cfg["training"]["batch_size"]
        num_workers = cfg["training"].get("num_workers", 4)
        train_dataset = LMDBFaceDataset(train_path)
        if accelerator.is_main_process:
            print(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} classes")

        pair_sampler = PairSampler(train_dataset.get_labels(), P=batch_size // 2)
        train_loader = DataLoader(train_dataset, batch_sampler=pair_sampler,
                                  num_workers=num_workers, pin_memory=True)

        loss_fn = build_loss(cfg).to(accelerator.device)

        # Optimizer (only trainable params — matters for LoRA)
        params = [p for p in backbone.parameters() if p.requires_grad]
        opt_name = cfg["training"].get("optimizer", "sgd")
        if opt_name == "adamw":
            optimizer = optim.AdamW(
                params, lr=cfg["training"]["lr"],
                weight_decay=cfg["training"].get("weight_decay", 0.01),
            )
        else:
            optimizer = optim.SGD(
                params, lr=cfg["training"]["lr"],
                momentum=cfg["training"].get("momentum", 0.9),
                weight_decay=cfg["training"].get("weight_decay", 0.0005),
            )

        # Resume from checkpoint
        start_epoch = 1
        ckpt_path = os.path.join(output_dir, "last.pth")
        ckpt = None
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(ckpt["backbone"])
            start_epoch = ckpt["epoch"] + 1

        backbone, optimizer, train_loader = accelerator.prepare(
            backbone, optimizer, train_loader
        )

        if ckpt is not None and "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if accelerator.is_main_process:
                print(f"Resumed from epoch {ckpt['epoch']}")
            ckpt = None

    # Training loop
    warmup = cfg["training"].get("warmup_epochs", 2)
    base_lr = cfg["training"]["lr"]

    csv_file = None
    writer = None
    start_epoch = start_epoch if epochs > 0 else 1
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

        train_loss, train_acc = train_epoch(
            backbone, loss_fn, train_loader, optimizer, accelerator
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

    # LFW + CFP evaluation on best model
    if accelerator.is_main_process:
        from src.eval.lfw import prepare_lfw, evaluate_lfw
        from src.eval.cfp import prepare_cfp, evaluate_cfp

        data_dir = os.environ.get("DATA_DIR", "data")
        eval_bs = cfg["training"].get("eval_batch_size", 64)

        prepare_lfw(data_dir)
        ckpt = torch.load(os.path.join(output_dir, "best.pth"),
                          map_location="cpu", weights_only=True)
        accelerator.unwrap_model(backbone).load_state_dict(ckpt["backbone"])
        mean_acc, std_acc, mean_thresh = evaluate_lfw(
            accelerator.unwrap_model(backbone), data_dir, accelerator.device,
            output_dir=output_dir, batch_size=eval_bs,
        )
        print(f"LFW 10-fold: acc={mean_acc:.4f} ± {std_acc:.4f}, "
              f"thresh@FAR0.001={mean_thresh:.4f}")
        if writer:
            writer.writerow(["lfw_eval", "", "", f"{mean_acc:.6f}"])

        try:
            prepare_cfp(data_dir)
            for protocol in ("FF", "FP"):
                acc, std, thresh = evaluate_cfp(
                    accelerator.unwrap_model(backbone), data_dir,
                    accelerator.device, protocol, output_dir=output_dir,
                    batch_size=eval_bs,
                )
                print(f"CFP-{protocol} 10-fold: acc={acc:.4f} ± {std:.4f}, "
                      f"thresh@FAR0.001={thresh:.4f}")
                if writer:
                    writer.writerow([f"cfp_{protocol.lower()}_eval", "", "", f"{acc:.6f}"])
        except FileNotFoundError as e:
            print(f"Skipping CFP evaluation: {e}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
