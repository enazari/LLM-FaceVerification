"""LMDB-backed face dataset."""

import io
import os
import pickle
import random
from collections import defaultdict

import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class LMDBFaceDataset(Dataset):
    """Face dataset backed by an LMDB written by prepare_data.py."""

    def __init__(self, lmdb_path: str, transform=None):
        self.lmdb_path = lmdb_path
        self.transform = transform or default_transform()
        self._env = None  # opened lazily per worker

        # Read metadata without keeping env open
        env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with env.begin() as txn:
            self._len       = int(txn.get(b"__len__").decode())
            self._nclasses  = int(txn.get(b"__nclasses__").decode())
        env.close()

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

    @property
    def num_classes(self) -> int:
        return self._nclasses

    def get_labels(self) -> np.ndarray:
        """Return all labels as a numpy array (used by PKSampler)."""
        env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        labels = np.empty(self._len, dtype=np.int64)
        with env.begin() as txn:
            for i in range(self._len):
                value = txn.get(f"{i:09d}".encode())
                labels[i] = pickle.loads(value)["label"]
        env.close()
        return labels

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._open_env()
        key   = f"{idx:09d}".encode()
        with self._env.begin() as txn:
            value = txn.get(key)
        record = pickle.loads(value)
        image  = Image.open(io.BytesIO(record["jpeg"])).convert("RGB")
        return self.transform(image), record["label"]


class FirLMDBFaceDataset(Dataset):
    """Reads the pre-existing MTCNN-processed MS1M LMDB on Fir.

    Avoids any data conversion by doing the train/val identity split in
    memory. Index/label arrays are cached to disk after the first scan
    so subsequent workers and runs are fast.

    Set env var FIR_LMDB=<path> to activate this code path.
    """

    def __init__(self, lmdb_path: str, split: str = "train",
                 num_identities: int = 85742, val_fraction: float = 0.2,
                 seed: int = 42, transform=None):
        self.lmdb_path = lmdb_path
        self.split = split
        self.transform = transform or default_transform()
        self._env = None

        cache_path = f"{lmdb_path}.{num_identities}_{split}.idx.pkl"
        if os.path.exists(cache_path):
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            self._indices = data["indices"]   # list of int (original LMDB key)
            self._labels  = data["labels"]    # remapped 0..N-1
            self._nclasses = data["nclasses"]
        else:
            self._indices, self._labels, self._nclasses = \
                self._build_split(lmdb_path, split, num_identities,
                                  val_fraction, seed, cache_path)

    @staticmethod
    def _build_split(lmdb_path, split, num_identities, val_fraction, seed,
                     cache_path):
        print(f"FirLMDBFaceDataset: scanning {lmdb_path} ...", flush=True)
        env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=1)
        label_to_indices = defaultdict(list)
        with env.begin(buffers=True) as txn:
            cur = txn.cursor()
            i = 0
            for key, val in cur.iternext(keys=True, values=True):
                rec = pickle.loads(bytes(val))
                if not rec.get("mtcnn_success", True):
                    i += 1
                    continue
                label_to_indices[rec["label"]].append(i)
                i += 1
        env.close()

        # Select top-N by image count, remap labels
        sorted_ids = sorted(label_to_indices, key=lambda k: len(label_to_indices[k]),
                            reverse=True)[:num_identities]
        label_remap = {old: new for new, old in enumerate(sorted_ids)}

        # Split identities 80/20
        rng = random.Random(seed)
        shuffled = sorted_ids[:]
        rng.shuffle(shuffled)
        n_val = int(len(shuffled) * val_fraction)
        split_ids = set(shuffled[:n_val]) if split == "val" else set(shuffled[n_val:])

        indices, labels = [], []
        for orig_id in sorted_ids:
            if orig_id not in split_ids:
                continue
            for src_idx in label_to_indices[orig_id]:
                indices.append(src_idx)
                labels.append(label_remap[orig_id])

        nclasses = len(split_ids)
        print(f"  {split}: {len(indices):,} images, {nclasses:,} identities", flush=True)

        with open(cache_path, "wb") as f:
            pickle.dump({"indices": indices, "labels": labels,
                         "nclasses": nclasses}, f, protocol=4)
        return indices, labels, nclasses

    def _open_env(self):
        if self._env is None:
            self._env = lmdb.open(self.lmdb_path, readonly=True, lock=False)

    @property
    def num_classes(self) -> int:
        return self._nclasses

    def get_labels(self) -> np.ndarray:
        return np.array(self._labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        self._open_env()
        src_idx = self._indices[idx]
        key = f"{src_idx:08d}".encode()
        with self._env.begin() as txn:
            value = txn.get(key)
        record = pickle.loads(value)
        image = Image.open(io.BytesIO(record["image"])).convert("RGB")
        return self.transform(image), self._labels[idx]
