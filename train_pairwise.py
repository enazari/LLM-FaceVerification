"""Training script for pairwise face verification (Track 2).

Feeds two face images + text prompt through InternVL2-2B.
Trains with cross-entropy on "Yes"/"No" answer tokens.
"""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.factory import build_backbone
from src.data.prepare import prepare, lmdb_paths
from src.data.pair_dataset import PairDataset
from src.utils import save_checkpoint
from src.training import (parse_training_args, setup_accelerator,
                           setup_output_dir, build_optimizer,
                           run_training_loop)


def train_epoch(backbone, loader, optimizer, accelerator, max_steps=0,
                yes_id=None, no_id=None):
    backbone.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps_done = 0

    for img_a, img_b, labels in tqdm(loader, desc="  train", leave=False,
                                      disable=not accelerator.is_main_process):
        with accelerator.accumulate(backbone):
            optimizer.zero_grad()
            logits = backbone(img_a, img_b)

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
            yes_no_logits = logits[:, [yes_id, no_id]]
            preds = yes_no_logits.argmax(dim=1)
            targets = (labels == 0).long()
            correct += (preds == targets).sum().item()

        total_loss += loss.item()
        total += labels.size(0)
        steps_done += 1

        if max_steps > 0 and steps_done >= max_steps:
            break

    return total_loss / max(steps_done, 1), correct / max(total, 1)


def main():
    args, cfg = parse_training_args(default_config="internvl-pair-lora")
    accelerator = setup_accelerator(cfg)
    output_dir = setup_output_dir(cfg, args.config, accelerator)
    epochs = cfg["training"]["epochs"]

    backbone = build_backbone(cfg)
    unwrapped = backbone

    if epochs == 0:
        backbone = accelerator.prepare(backbone)
        save_checkpoint(accelerator, backbone, None, 0, output_dir, "best")
        return

    # Data
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

    # Optimizer + resume
    params = [p for p in backbone.parameters() if p.requires_grad]
    optimizer = build_optimizer(params, cfg)

    start_epoch = 1
    ckpt_path = os.path.join(output_dir, "last.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(ckpt["backbone"])
        start_epoch = ckpt["epoch"] + 1

    backbone, optimizer, train_loader = accelerator.prepare(
        backbone, optimizer, train_loader
    )

    if start_epoch > 1:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            if accelerator.is_main_process:
                print(f"Resumed from epoch {ckpt['epoch']}")

    yes_id = unwrapped.yes_token_id
    no_id = unwrapped.no_token_id

    # Train
    def _train_epoch(backbone, loader, optimizer, accelerator, max_steps=0):
        return train_epoch(backbone, loader, optimizer, accelerator,
                           max_steps=max_steps, yes_id=yes_id, no_id=no_id)

    run_training_loop(
        accelerator=accelerator, backbone=backbone, optimizer=optimizer,
        train_loader=train_loader, cfg=cfg, args=args,
        output_dir=output_dir, train_epoch_fn=_train_epoch,
        start_epoch=start_epoch,
    )

    # Evaluation
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
        except FileNotFoundError as e:
            print(f"Skipping CFP evaluation: {e}")


if __name__ == "__main__":
    main()
