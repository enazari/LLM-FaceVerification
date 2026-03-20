"""Build loss from config."""

import torch.nn as nn

from src.losses.infonce import InfoNCELoss


def build_loss(cfg: dict) -> nn.Module:
    name = cfg["loss"]["name"]

    if name == "infonce":
        return InfoNCELoss(temperature=cfg["loss"]["temperature"])

    raise ValueError(f"Unknown loss: {name}")
