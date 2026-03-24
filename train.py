"""Training script for face verification (embedding-based, Tracks 1 & 3)."""

import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.backbones.factory import build_backbone
from src.data.prepare import prepare, lmdb_paths
from src.data.dataset import LMDBFaceDataset, FirLMDBFaceDataset
from src.data.sampler import PairSampler
from src.losses.factory import build_loss
from src.utils import save_checkpoint
from src.training import (parse_training_args, setup_accelerator,
                           setup_output_dir, build_optimizer,
                           run_training_loop)


def train_epoch(backbone, loader, optimizer, accelerator, max_steps=0,
                loss_fn=None):
    backbone.train()
    total_loss = 0.0
    correct = 0
    total = 0
    steps_done = 0

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
            steps_done += 1

            if max_steps > 0 and steps_done >= max_steps:
                break

    return total_loss / max(steps_done, 1), correct / max(total, 1)


def main():
    args, cfg = parse_training_args(default_config="internvit-lora")
    accelerator = setup_accelerator(cfg)
    accelerator.even_batches = False
    output_dir = setup_output_dir(cfg, args.config, accelerator)
    epochs = cfg["training"]["epochs"]

    backbone = build_backbone(cfg)

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
    if os.environ.get("FIR_LMDB"):
        train_dataset = FirLMDBFaceDataset(
            train_path, split="train",
            num_identities=cfg["data"]["num_identities"],
            val_fraction=cfg["data"].get("val_fraction", 0.2),
        )
    else:
        train_dataset = LMDBFaceDataset(train_path)
    if accelerator.is_main_process:
        print(f"Train: {len(train_dataset)} images, {train_dataset.num_classes} classes")

    pair_sampler = PairSampler(train_dataset.get_labels(), P=batch_size // 2)
    train_loader = DataLoader(train_dataset, batch_sampler=pair_sampler,
                              num_workers=num_workers, pin_memory=True)

    loss_fn = build_loss(cfg).to(accelerator.device)

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

    # Train
    def _train_epoch(backbone, loader, optimizer, accelerator, max_steps=0):
        return train_epoch(backbone, loader, optimizer, accelerator,
                           max_steps=max_steps, loss_fn=loss_fn)

    run_training_loop(
        accelerator=accelerator, backbone=backbone, optimizer=optimizer,
        train_loader=train_loader, cfg=cfg, args=args,
        output_dir=output_dir, train_epoch_fn=_train_epoch,
        start_epoch=start_epoch,
    )

    # Evaluation
    if accelerator.is_main_process and not args.skip_eval:
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
        except FileNotFoundError as e:
            print(f"Skipping CFP evaluation: {e}")


if __name__ == "__main__":
    main()
