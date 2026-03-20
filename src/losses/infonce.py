"""InfoNCE (CLIP-style paired contrastive) loss.

Batch structure: P identities × 2 images (PairSampler).
For each image i, its positive is the paired image of the same identity.
All other 2P-2 images are negatives.

Operates on L2-normalized embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        B = embeddings.size(0)
        sim = embeddings @ embeddings.T / self.temperature  # [2P, 2P]

        # Mask self-similarity
        sim.fill_diagonal_(float('-inf'))

        # Positive partner indices: 0↔1, 2↔3, 4↔5, ...
        targets = torch.arange(B, device=sim.device)
        targets[0::2] = torch.arange(1, B, 2, device=sim.device)
        targets[1::2] = torch.arange(0, B, 2, device=sim.device)

        return F.cross_entropy(sim, targets)
