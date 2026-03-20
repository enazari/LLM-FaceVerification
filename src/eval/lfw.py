"""LFW 10-fold evaluation with self-reliant LMDB creation.

Downloads LFW (250x250 originals) from figshare, detects + aligns faces with
RetinaFace, caches in LMDB, evaluates with 10-fold cross-validation at FAR@0.001.
"""

import os
import pickle
import tarfile
import urllib.request

import cv2
import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.transform import SimilarityTransform
from torchvision import transforms
from tqdm import tqdm

from .common import compute_10fold, write_results_csv

# ArcFace reference landmarks for 112x112 alignment
ARCFACE_REF = np.array([
    [38.2946, 51.6963],   # left eye (viewer)
    [73.5318, 51.5014],   # right eye (viewer)
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth (viewer)
    [70.7299, 92.2041],   # right mouth (viewer)
], dtype=np.float32)

LMDB_NAME = "lfw_10fold_original_retinaface.lmdb"

# Same figshare URLs used by scikit-learn
_LFW_TGZ_URL = "https://ndownloader.figshare.com/files/5976018"
_PAIRS_URL = "https://ndownloader.figshare.com/files/5976006"


# ---------------------------------------------------------------------------
# Download + parse
# ---------------------------------------------------------------------------

def _download_file(url, dest, desc="Downloading"):
    resp = urllib.request.urlopen(url)
    total = int(resp.headers.get("Content-Length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=desc) as bar:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)
            bar.update(len(chunk))


def _ensure_lfw_downloaded(cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    lfw_dir = os.path.join(cache_dir, "lfw")
    pairs_path = os.path.join(cache_dir, "pairs.txt")

    if not os.path.exists(pairs_path):
        _download_file(_PAIRS_URL, pairs_path, "pairs.txt")

    if not os.path.isdir(lfw_dir):
        tgz_path = os.path.join(cache_dir, "lfw.tgz")
        if not os.path.exists(tgz_path):
            _download_file(_LFW_TGZ_URL, tgz_path, "lfw.tgz (~173 MB)")
        print("Extracting lfw.tgz...")
        with tarfile.open(tgz_path) as tf:
            tf.extractall(cache_dir)
        os.remove(tgz_path)

    return lfw_dir, pairs_path


def _parse_pairs(pairs_path, lfw_dir):
    folds = []
    with open(pairs_path) as f:
        header = f.readline().strip().split()
        n_folds, n_per_class = int(header[0]), int(header[1])
        for _ in range(n_folds):
            fold = []
            for _ in range(n_per_class):
                parts = f.readline().strip().split("\t")
                name, i1, i2 = parts[0], int(parts[1]), int(parts[2])
                fold.append((
                    os.path.join(lfw_dir, name, f"{name}_{i1:04d}.jpg"),
                    os.path.join(lfw_dir, name, f"{name}_{i2:04d}.jpg"),
                    1,
                ))
            for _ in range(n_per_class):
                parts = f.readline().strip().split("\t")
                name1, i1, name2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
                fold.append((
                    os.path.join(lfw_dir, name1, f"{name1}_{i1:04d}.jpg"),
                    os.path.join(lfw_dir, name2, f"{name2}_{i2:04d}.jpg"),
                    0,
                ))
            folds.append(fold)
    return folds


# ---------------------------------------------------------------------------
# Face detection + alignment
# ---------------------------------------------------------------------------

def _detect_align(img_rgb, padding=150):
    from retinaface import RetinaFace

    h, w = img_rgb.shape[:2]
    padded = np.full((h + 2 * padding, w + 2 * padding, 3), 128, dtype=np.uint8)
    padded[padding:padding + h, padding:padding + w] = img_rgb

    padded_bgr = padded[:, :, ::-1].copy()
    faces = RetinaFace.detect_faces(padded_bgr, threshold=0.5)
    if not isinstance(faces, dict) or len(faces) == 0:
        return None

    cx, cy = padding + w / 2, padding + h / 2
    best_key, best_dist = None, float("inf")
    for key, face in faces.items():
        x1, y1, x2, y2 = face["facial_area"]
        fx, fy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = (fx - cx) ** 2 + (fy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_key = key

    lms = faces[best_key]["landmarks"]
    landmarks = np.array([
        lms["right_eye"],    # viewer left eye
        lms["left_eye"],     # viewer right eye
        lms["nose"],
        lms["mouth_right"],  # viewer left mouth
        lms["mouth_left"],   # viewer right mouth
    ], dtype=np.float32)
    landmarks -= padding

    tform = SimilarityTransform()
    tform.estimate(landmarks, ARCFACE_REF)
    M = tform.params[:2]
    aligned = cv2.warpAffine(img_rgb, M, (112, 112), borderValue=0.0)
    return aligned.astype(np.uint8)


# ---------------------------------------------------------------------------
# LMDB preparation
# ---------------------------------------------------------------------------

def prepare_lfw(data_dir="data"):
    lmdb_path = os.path.join(data_dir, LMDB_NAME)
    if os.path.exists(lmdb_path):
        print(f"LFW LMDB exists: {lmdb_path}")
        return

    cache_dir = os.path.join(data_dir, "lfw_raw")
    lfw_dir, pairs_path = _ensure_lfw_downloaded(cache_dir)
    folds = _parse_pairs(pairs_path, lfw_dir)

    print("Detecting + aligning faces (250x250 originals)...")
    kept_records = []
    fold_sizes = []
    failed = 0

    for fold_idx, fold in enumerate(folds):
        fold_kept = 0
        for path_a, path_b, label in tqdm(fold, desc=f"fold {fold_idx+1}/10", leave=False):
            img_a = np.array(Image.open(path_a).convert("RGB"))
            img_b = np.array(Image.open(path_b).convert("RGB"))
            aligned_a = _detect_align(img_a)
            aligned_b = _detect_align(img_b)
            if aligned_a is None or aligned_b is None:
                failed += 1
                continue
            kept_records.append((aligned_a, aligned_b, label))
            fold_kept += 1
        fold_sizes.append(fold_kept)

    total = sum(len(f) for f in folds)
    print(f"Aligned {len(kept_records)}/{total} pairs ({failed} failed)")

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

def _load_lfw(data_dir="data"):
    lmdb_path = os.path.join(data_dir, LMDB_NAME)
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
# Embedding helpers
# ---------------------------------------------------------------------------

def _embed_faces(backbone, face_list, device, batch_size=64):
    """Extract L2-normalized embeddings from a list of face arrays."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    embs = []
    buf = []
    for face in face_list:
        buf.append(transform(Image.fromarray(face)))
        if len(buf) == batch_size:
            t = torch.stack(buf).to(device)
            e = F.normalize(backbone(t), dim=1)
            embs.append(e.cpu())
            buf = []
    if buf:
        t = torch.stack(buf).to(device)
        e = F.normalize(backbone(t), dim=1)
        embs.append(e.cpu())
    return torch.cat(embs)


def _pairwise_scores(backbone, faces_a, faces_b, device, batch_size=8):
    """Compute P("Yes") for all pairs using a pairwise MLLM backbone."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    n = len(faces_a)
    scores = []
    for i in tqdm(range(0, n, batch_size), desc="  pairwise scoring", leave=False):
        end = min(i + batch_size, n)
        batch_a = torch.stack([transform(Image.fromarray(faces_a[j])) for j in range(i, end)]).to(device)
        batch_b = torch.stack([transform(Image.fromarray(faces_b[j])) for j in range(i, end)]).to(device)
        logits = backbone(batch_a, batch_b)
        p_yes = backbone.get_yes_no_scores(logits)
        scores.append(p_yes.cpu())
    return torch.cat(scores).numpy()


# ---------------------------------------------------------------------------
# Evaluation (embedding-based, Tracks 1 & 3)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_lfw(backbone, data_dir, device, output_dir=None, batch_size=64):
    faces_a, faces_b, labels, fold_sizes = _load_lfw(data_dir)
    print(f"LFW eval: {len(labels)} pairs, folds={fold_sizes}")

    backbone.eval()
    embs_a = _embed_faces(backbone, faces_a, device, batch_size)
    embs_b = _embed_faces(backbone, faces_b, device, batch_size)
    sims = (embs_a * embs_b).sum(dim=1).numpy()

    results = compute_10fold(sims, labels, fold_sizes)
    if output_dir:
        write_results_csv(output_dir, "lfw_results.csv", results)
    return results["mean_acc"], results["std_acc"], results["mean_thresh"]


# ---------------------------------------------------------------------------
# Evaluation (pairwise MLLM, Track 2)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_lfw_pairwise(backbone, data_dir, device, output_dir=None,
                           batch_size=8):
    faces_a, faces_b, labels, fold_sizes = _load_lfw(data_dir)
    print(f"LFW pairwise eval: {len(labels)} pairs, folds={fold_sizes}")

    backbone.eval()
    sims = _pairwise_scores(backbone, faces_a, faces_b, device, batch_size)

    results = compute_10fold(sims, labels, fold_sizes)
    if output_dir:
        write_results_csv(output_dir, "lfw_pairwise_results.csv", results)
    return results["mean_acc"], results["std_acc"], results["mean_thresh"]
