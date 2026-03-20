"""CFP 10-fold evaluation (FF and FP protocols) with LMDB caching.

Expects the CFP dataset at data/cfp-dataset/ (manual download from Kaggle).
Detects + aligns faces with RetinaFace (reuses _detect_align from lfw.py),
caches in LMDB, evaluates with 10-fold cross-validation at FAR@0.001.
"""

import os
import pickle
import shutil

import lmdb
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .common import compute_10fold, write_results_csv
from .lfw import _detect_align, _embed_faces, _pairwise_scores

LMDB_NAME_FF = "cfp_ff_10fold_retinaface.lmdb"
LMDB_NAME_FP = "cfp_fp_10fold_retinaface.lmdb"
CFP_SUBDIR = "cfp-dataset"

_LMDB_NAMES = {"FF": LMDB_NAME_FF, "FP": LMDB_NAME_FP}


# ---------------------------------------------------------------------------
# Parse CFP protocol
# ---------------------------------------------------------------------------

def _load_pair_list(path):
    base_dir = os.path.dirname(path)
    index_to_path = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            idx = int(parts[0])
            rel_path = parts[1]
            index_to_path[idx] = os.path.normpath(os.path.join(base_dir, rel_path))
    return index_to_path


def _parse_cfp_protocol(cfp_dir, protocol):
    proto_dir = os.path.join(cfp_dir, "Protocol")
    frontal_map = _load_pair_list(os.path.join(proto_dir, "Pair_list_F.txt"))
    map_b = frontal_map
    if protocol == "FP":
        map_b = _load_pair_list(os.path.join(proto_dir, "Pair_list_P.txt"))

    folds = []
    for fold_idx in range(1, 11):
        fold = []
        fold_dir = os.path.join(proto_dir, "Split", protocol, f"{fold_idx:02d}")
        for fname, label in [("same.txt", 1), ("diff.txt", 0)]:
            with open(os.path.join(fold_dir, fname)) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    a_str, b_str = line.split(",")
                    fold.append((frontal_map[int(a_str)], map_b[int(b_str)], label))
        folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# LMDB preparation
# ---------------------------------------------------------------------------

def prepare_cfp(data_dir="data"):
    cfp_dir = os.path.join(data_dir, CFP_SUBDIR)
    if not os.path.isdir(cfp_dir):
        raise FileNotFoundError(
            f"CFP dataset not found at {cfp_dir}. "
            "Download from https://www.kaggle.com/datasets/chinafax/cfpw-dataset "
            "and extract to data/cfp-dataset/"
        )

    for protocol, lmdb_name in [("FF", LMDB_NAME_FF), ("FP", LMDB_NAME_FP)]:
        lmdb_path = os.path.join(data_dir, lmdb_name)
        if os.path.exists(lmdb_path):
            print(f"CFP-{protocol} LMDB exists: {lmdb_path}")
            continue

        folds = _parse_cfp_protocol(cfp_dir, protocol)
        print(f"Detecting + aligning CFP-{protocol} faces...")

        align_cache = {}
        failed_paths = set()

        def get_aligned(path):
            if path not in align_cache:
                img = np.array(Image.open(path).convert("RGB"))
                result = _detect_align(img)
                align_cache[path] = result
                if result is None:
                    failed_paths.add(path)
            return align_cache[path]

        kept_records = []
        fold_sizes = []
        failed = 0

        for fold_idx, fold in enumerate(folds):
            fold_kept = 0
            for path_a, path_b, label in tqdm(fold, desc=f"CFP-{protocol} fold {fold_idx+1}/10", leave=False):
                aligned_a = get_aligned(path_a)
                aligned_b = get_aligned(path_b)
                if aligned_a is None or aligned_b is None:
                    failed += 1
                    continue
                kept_records.append((aligned_a, aligned_b, label))
                fold_kept += 1
            fold_sizes.append(fold_kept)

        total = sum(len(f) for f in folds)
        print(f"CFP-{protocol}: aligned {len(kept_records)}/{total} pairs ({failed} failed)")

        if failed_paths:
            fail_dir = os.path.join(data_dir, f"cfp_{protocol.lower()}_failed")
            if os.path.isdir(fail_dir):
                shutil.rmtree(fail_dir)
            os.makedirs(fail_dir)
            for path in sorted(failed_paths):
                shutil.copy(path, fail_dir)
            print(f"Saved {len(failed_paths)} failed images → {fail_dir}/")

        os.makedirs(data_dir, exist_ok=True)
        map_size = len(kept_records) * 112 * 112 * 3 * 2 * 2
        env = lmdb.open(lmdb_path, map_size=max(map_size, 1 << 30))
        with env.begin(write=True) as txn:
            for i, (fa, fb, lbl) in enumerate(kept_records):
                key = f"{i:06d}".encode()
                txn.put(key, pickle.dumps({"face_a": fa, "face_b": fb, "label": lbl}))
            txn.put(b"__len__", str(len(kept_records)).encode())
            txn.put(b"__fold_sizes__", pickle.dumps(fold_sizes))
        env.close()
        print(f"Wrote {lmdb_path} ({len(kept_records)} pairs, folds={fold_sizes})")


# ---------------------------------------------------------------------------
# Load pairs from LMDB
# ---------------------------------------------------------------------------

def _load_cfp(data_dir, protocol):
    lmdb_path = os.path.join(data_dir, _LMDB_NAMES[protocol])
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        n = int(txn.get(b"__len__").decode())
        fold_sizes = pickle.loads(txn.get(b"__fold_sizes__"))
        faces_a, faces_b, labels = [], [], []
        for i in range(n):
            rec = pickle.loads(txn.get(f"{i:06d}".encode()))
            faces_a.append(rec["face_a"])
            faces_b.append(rec["face_b"])
            labels.append(rec["label"])
    env.close()
    return faces_a, faces_b, np.array(labels), fold_sizes


# ---------------------------------------------------------------------------
# Evaluation (embedding-based, Tracks 1 & 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cfp(backbone, data_dir, device, protocol, output_dir=None,
                  batch_size=64):
    faces_a, faces_b, labels, fold_sizes = _load_cfp(data_dir, protocol)
    tag = f"CFP-{protocol}"
    print(f"{tag} eval: {len(labels)} pairs, folds={fold_sizes}")

    backbone.eval()
    embs_a = _embed_faces(backbone, faces_a, device, batch_size)
    embs_b = _embed_faces(backbone, faces_b, device, batch_size)
    sims = (embs_a * embs_b).sum(dim=1).numpy()

    results = compute_10fold(sims, labels, fold_sizes)
    if output_dir:
        write_results_csv(output_dir, f"cfp_{protocol.lower()}_results.csv", results)
    return results["mean_acc"], results["std_acc"], results["mean_thresh"]


# ---------------------------------------------------------------------------
# Evaluation (pairwise MLLM, Track 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cfp_pairwise(backbone, data_dir, device, protocol,
                           output_dir=None, batch_size=8):
    faces_a, faces_b, labels, fold_sizes = _load_cfp(data_dir, protocol)
    tag = f"CFP-{protocol}"
    print(f"{tag} pairwise eval: {len(labels)} pairs, folds={fold_sizes}")

    backbone.eval()
    sims = _pairwise_scores(backbone, faces_a, faces_b, device, batch_size)

    results = compute_10fold(sims, labels, fold_sizes)
    if output_dir:
        write_results_csv(output_dir, f"cfp_{protocol.lower()}_pairwise_results.csv", results)
    return results["mean_acc"], results["std_acc"], results["mean_thresh"]
