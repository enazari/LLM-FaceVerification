"""Training script for pairwise face verification (Track 2).

Feeds two face images + text prompt through InternVL2-2B.
Trains with cross-entropy on "Yes"/"No" answer tokens.
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as _ckpt_mod
from accelerate import Accelerator

# Force use_reentrant=False for gradient checkpointing (LoRA compatibility)
_orig_ckpt = _ckpt_mod.checkpoint
def _patched_ckpt(*args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return _orig_ckpt(*args, **kwargs)
_ckpt_mod.checkpoint = _patched_ckpt
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from src.data.prepare import prepare, lmdb_paths
from src.backbones.factory import build_backbone
from src.data.pair_dataset import PairDataset
from src.utils import cosine_lr, save_checkpoint, apply_overrides


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_epoch(backbone, loader, optimizer, accelerator, yes_id, no_id,
                max_steps=0):
    backbone.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps_done = 0

    for img_a, img_b, labels in tqdm(loader, desc="  train", leave=False,
                                      disable=not accelerator.is_main_process):
        with accelerator.accumulate(backbone):
            optimizer.zero_grad()
            logits = backbone(img_a, img_b)  # [B, vocab_size]

            # Target: Yes token for label=1, No token for label=0
            target_ids = torch.where(
                labels == 1,
                torch.full_like(labels, yes_id),
                torch.full_like(labels, no_id),
            )

            loss = F.cross_entropy(logits, target_ids)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=5.0)
            optimizer.step()

        with torch.no_grad():
            yes_no_logits = logits[:, [yes_id, no_id]]  # [B, 2]
            preds = yes_no_logits.argmax(dim=1)  # 0=Yes, 1=No
            targets = (labels == 0).long()
            correct += (preds == targets).sum().item()

        total_loss += loss.item()
        total += labels.size(0)
        steps_done += 1

        if max_steps > 0 and steps_done >= max_steps:
            break

    return total_loss / max(steps_done, 1), correct / max(total, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="internvl-pair-lora")
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

    grad_accum = cfg["training"].get("gradient_accumulation_steps", 1)
    accelerator = Accelerator(
        mixed_precision=cfg["training"].get("mixed_precision", "no"),
        gradient_accumulation_steps=grad_accum,
    )

    output_dir = f"sessions/{cfg['session']}"
    epochs = cfg["training"]["epochs"]
    if accelerator.is_main_process:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        shutil.copy(f"configs/{args.config}.yaml", output_dir)

    backbone = build_backbone(cfg)
    unwrapped = backbone  # keep ref before wrapping

    if epochs == 0:
        backbone = accelerator.prepare(backbone)
        save_checkpoint(accelerator, backbone, None, 0, output_dir, "best")
    else:
        if accelerator.is_main_process:
            prepare(cfg)
        accelerator.wait_for_everyone()

        train_path, _ = lmdb_paths(cfg)
        batch_size = cfg["training"]["batch_size"]
        num_workers = cfg["training"].get("num_workers", 4)
        pairs_per_epoch = cfg["training"].get("pairs_per_epoch", 100_000)

        train_dataset = PairDataset(train_path, pairs_per_epoch=pairs_per_epoch)
        if accelerator.is_main_process:
            print(f"Train: {len(train_dataset)} pairs per epoch")

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

        params = [p for p in backbone.parameters() if p.requires_grad]
        opt_name = cfg["training"].get("optimizer", "adamw")
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

    yes_id = unwrapped.yes_token_id
    no_id = unwrapped.no_token_id

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

        if hasattr(train_loader.dataset, "resample"):
            train_loader.dataset.resample()

        train_loss, train_acc = train_epoch(
            backbone, train_loader, optimizer, accelerator, yes_id, no_id,
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

    # Pairwise evaluation
    if accelerator.is_main_process and not args.skip_eval:
        from src.eval.lfw import prepare_lfw, evaluate_lfw_pairwise
        from src.eval.cfp import prepare_cfp, evaluate_cfp_pairwise

        data_dir = os.environ.get("DATA_DIR", "data")
        eval_bs = cfg["training"].get("eval_batch_size", 8)

        prepare_lfw(data_dir)
        ckpt = torch.load(os.path.join(output_dir, "best.pth"),
                          map_location="cpu", weights_only=True)
        accelerator.unwrap_model(backbone).load_state_dict(ckpt["backbone"])
        mean_acc, std_acc, mean_thresh = evaluate_lfw_pairwise(
            accelerator.unwrap_model(backbone), data_dir, accelerator.device,
            output_dir=output_dir, batch_size=eval_bs,
        )
        print(f"LFW 10-fold (pairwise): acc={mean_acc:.4f} ± {std_acc:.4f}, "
              f"thresh@FAR0.001={mean_thresh:.4f}")
        if writer:
            writer.writerow(["lfw_eval", "", "", f"{mean_acc:.6f}"])

        try:
            prepare_cfp(data_dir)
            for protocol in ("FF", "FP"):
                acc, std, thresh = evaluate_cfp_pairwise(
                    accelerator.unwrap_model(backbone), data_dir,
                    accelerator.device, protocol, output_dir=output_dir,
                    batch_size=eval_bs,
                )
                print(f"CFP-{protocol} 10-fold (pairwise): acc={acc:.4f} ± {std:.4f}, "
                      f"thresh@FAR0.001={thresh:.4f}")
                if writer:
                    writer.writerow([f"cfp_{protocol.lower()}_eval", "", "", f"{acc:.6f}"])
        except FileNotFoundError as e:
            print(f"Skipping CFP evaluation: {e}")

    if csv_file:
        csv_file.close()


if __name__ == "__main__":
    main()
