"""Build backbone from config."""

import torch.nn as nn


def build_backbone(cfg: dict) -> nn.Module:
    name = cfg["backbone"]["name"]

    if name == "resnet50":
        from src.backbones.resnet import iresnet50
        return iresnet50(
            embedding_dim=cfg["backbone"]["embedding_dim"],
            dropout=cfg["backbone"].get("dropout", 0.0),
        )

    elif name == "internvit":
        from src.backbones.internvit import build_internvit
        return build_internvit(cfg)

    elif name == "internvl":
        from src.backbones.internvl import build_internvl
        return build_internvl(cfg)

    elif name == "internvl_pair":
        from src.backbones.internvl_pair import build_internvl_pair
        return build_internvl_pair(cfg)

    else:
        raise ValueError(f"Unknown backbone: {name}")
