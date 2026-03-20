"""Evaluate a saved checkpoint on LFW and CFP benchmarks."""

import argparse
import os

import torch
import yaml

from src.backbones.factory import build_backbone
from src.eval.lfw import prepare_lfw, evaluate_lfw
from src.eval.cfp import prepare_cfp, evaluate_cfp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name (without .yaml)")
    parser.add_argument("--checkpoint", default="best", help="Checkpoint name (best or last)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    with open(f"configs/{args.config}.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = f"sessions/{cfg['session']}"

    # Load model
    backbone = build_backbone(cfg)
    ckpt_path = os.path.join(output_dir, f"{args.checkpoint}.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        backbone.load_state_dict(ckpt["backbone"])
        print(f"Loaded {ckpt_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"No checkpoint at {ckpt_path}, evaluating with pretrained weights")
    backbone = backbone.to(device)

    # LFW
    prepare_lfw(args.data_dir)
    acc, std, thresh = evaluate_lfw(
        backbone, args.data_dir, device,
        output_dir=output_dir, batch_size=args.batch_size,
    )
    print(f"\nLFW: {acc:.4f} ± {std:.4f}  thresh={thresh:.4f}")

    # CFP
    try:
        prepare_cfp(args.data_dir)
        for protocol in ("FF", "FP"):
            acc, std, thresh = evaluate_cfp(
                backbone, args.data_dir, device, protocol,
                output_dir=output_dir, batch_size=args.batch_size,
            )
            print(f"CFP-{protocol}: {acc:.4f} ± {std:.4f}  thresh={thresh:.4f}")
    except FileNotFoundError as e:
        print(f"Skipping CFP: {e}")


if __name__ == "__main__":
    main()
