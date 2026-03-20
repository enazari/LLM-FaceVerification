"""Pairwise face dataset for Track 2 (MLLM binary classifier).

Wraps LMDBFaceDataset to yield (img_a, img_b, label) triples:
  - 50% positive pairs (same identity)
  - 50% negative pairs (different identities, re-sampled each epoch)
"""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.dataset import LMDBFaceDataset


class PairDataset(Dataset):
    """Yields (img_a, img_b, label) from an LMDB face dataset.

    Positive pairs: two random images of the same identity.
    Negative pairs: one image each from two different identities.
    Negatives are re-sampled each epoch via `resample()`.
    """

    def __init__(self, lmdb_path: str, pairs_per_epoch: int = 100_000):
        self.base = LMDBFaceDataset(lmdb_path)
        self.pairs_per_epoch = pairs_per_epoch

        # Build identity → indices mapping
        labels = self.base.get_labels()
        self.id_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.id_to_indices[int(label)].append(idx)

        self.identities = list(self.id_to_indices.keys())
        # Filter identities with at least 2 images for positive pairs
        self.pos_identities = [k for k, v in self.id_to_indices.items() if len(v) >= 2]

        self.pairs: list[tuple[int, int, int]] = []
        self.resample()

    def resample(self):
        """Re-generate pairs for a new epoch."""
        pairs = []
        n_pos = self.pairs_per_epoch // 2
        n_neg = self.pairs_per_epoch - n_pos

        # Positive pairs
        for _ in range(n_pos):
            ident = random.choice(self.pos_identities)
            a, b = random.sample(self.id_to_indices[ident], 2)
            pairs.append((a, b, 1))

        # Negative pairs
        for _ in range(n_neg):
            id1, id2 = random.sample(self.identities, 2)
            a = random.choice(self.id_to_indices[id1])
            b = random.choice(self.id_to_indices[id2])
            pairs.append((a, b, 0))

        random.shuffle(pairs)
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_a, idx_b, label = self.pairs[idx]
        img_a, _ = self.base[idx_a]
        img_b, _ = self.base[idx_b]
        return img_a, img_b, label
