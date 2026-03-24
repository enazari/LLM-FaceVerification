#!/bin/bash
#SBATCH --job-name=convert_lmdb
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpubase_bynode_b3
#SBATCH --output=/scratch/enazari/slurm-logs/%j-%x.out

# Convert MTCNN-processed MS1M LMDB to LLM-FaceVerification format.
# Source: /project/def-pbranco/enazari/datasets/MTCNN-processed_MS1M_ArcFace.lmdb
# Output: data/ms1m_85742_train.lmdb and data/ms1m_85742_val.lmdb

PROJECT=/project/def-pbranco/enazari/mllm-fv/LLM-FaceVerification
SRC_LMDB=/project/def-pbranco/enazari/datasets/MTCNN-processed_MS1M_ArcFace.lmdb

module load StdEnv/2023 gcc/12.3 python/3.11
source /project/def-pbranco/enazari/arcface_env/bin/activate

cd $PROJECT
mkdir -p data

python3 - << 'PYEOF'
import lmdb
import pickle
import random
from collections import defaultdict

SRC = "/project/def-pbranco/enazari/datasets/MTCNN-processed_MS1M_ArcFace.lmdb"
OUT_TRAIN = "data/ms1m_85742_train.lmdb"
OUT_VAL   = "data/ms1m_85742_val.lmdb"
VAL_FRAC  = 0.2
SEED      = 42

print("Pass 1: scanning labels...", flush=True)
src_env = lmdb.open(SRC, readonly=True, lock=False, max_readers=8)
label_to_indices = defaultdict(list)
with src_env.begin(buffers=True) as txn:
    cur = txn.cursor()
    i = 0
    for key, val in cur.iternext(keys=True, values=True):
        if key[:2] == b'__':
            continue
        rec = pickle.loads(bytes(val))
        if not rec.get('mtcnn_success', True):
            i += 1
            continue
        label_to_indices[rec['label']].append(i)
        i += 1
        if i % 500000 == 0:
            print(f"  scanned {i:,} entries, {len(label_to_indices):,} labels", flush=True)

all_labels = sorted(label_to_indices.keys())
print(f"Total: {i:,} entries, {len(all_labels):,} unique labels", flush=True)

# Remap labels to 0..N-1
label_remap = {old: new for new, old in enumerate(all_labels)}

# Split identities into train/val
random.seed(SEED)
random.shuffle(all_labels)
n_val = int(len(all_labels) * VAL_FRAC)
val_labels = set(all_labels[:n_val])
train_labels = set(all_labels[n_val:])

train_indices = []
val_indices   = []
for lbl in train_labels:
    for idx in label_to_indices[lbl]:
        train_indices.append((idx, label_remap[lbl]))
for lbl in val_labels:
    for idx in label_to_indices[lbl]:
        val_indices.append((idx, label_remap[lbl]))

print(f"Train: {len(train_indices):,} images from {len(train_labels):,} identities", flush=True)
print(f"Val:   {len(val_indices):,} images from {len(val_labels):,} identities", flush=True)

def write_lmdb(out_path, indices_labels, src_env, name):
    map_size = 60 * 1024**3  # 60 GB
    out_env = lmdb.open(out_path, map_size=map_size)
    with src_env.begin(buffers=True) as src_txn:
        with out_env.begin(write=True) as out_txn:
            out_txn.put(b"__len__",      str(len(indices_labels)).encode())
            out_txn.put(b"__nclasses__", str(len(set(lbl for _, lbl in indices_labels))).encode())
            for new_idx, (src_idx, label) in enumerate(indices_labels):
                src_key = f"{src_idx:08d}".encode()
                raw = src_txn.get(src_key)
                rec = pickle.loads(bytes(raw))
                out_rec = {"jpeg": rec["image"], "label": label}
                out_key = f"{new_idx:09d}".encode()
                out_txn.put(out_key, pickle.dumps(out_rec, protocol=4))
                if new_idx % 200000 == 0:
                    print(f"  {name}: wrote {new_idx:,}/{len(indices_labels):,}", flush=True)
    out_env.close()
    print(f"{name} done.", flush=True)

print("Pass 2: writing train LMDB...", flush=True)
write_lmdb(OUT_TRAIN, train_indices, src_env, "train")
print("Pass 3: writing val LMDB...", flush=True)
write_lmdb(OUT_VAL, val_indices, src_env, "val")
src_env.close()
print("Conversion complete.", flush=True)
PYEOF

echo "Exit code: $?"
