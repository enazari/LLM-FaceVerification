# LLM-FaceVerification

LoRA-finetuning a multimodal LLM (InternVL2-2B) for face verification — exploring whether language-aware representations can learn identity.

## Motivation

Multimodal LLMs like InternVL2 combine a vision encoder (InternViT-300M) with an LLM (InternLM2-1.8B) trained on billions of image-text pairs. But face verification is a metric learning problem: it requires fine-grained identity discrimination, not broad visual understanding.

- Zero-shot MLLMs plateau at ~81% on LFW (vs. ArcFace at ~99.7%)
- No published work applies LoRA to an MLLM with a contrastive face verification loss
- **Question:** Can parameter-efficient finetuning close this gap?

## Three Experimental Tracks

| Track | What | Input | Loss | Output | Hypothesis |
|-------|------|-------|------|--------|------------|
| 1 | ViT encoder only | 1 image &rarr; embedding | InfoNCE | `[B, 1024]` | Does instruction-tuned InternViT beat standalone DINOv2/CLIP? |
| 2 | Full MLLM classifier | 2 images + prompt &rarr; Yes/No | Softmax (CE) | P("Yes") as score | Can the MLLM learn verification through its natural text interface? |
| 3 | Full MLLM embedding | 1 image + prompt &rarr; LLM hidden state | InfoNCE | `[B, 2048]` | Are LLM hidden states useful as face embeddings? |

All tracks use LoRA (rank 8) on ViT FFN layers (`fc1`, `fc2`) and/or LLM FFN layers (`w1`, `w2`, `w3`).

## Architecture

```
InternVL2-2B
├── InternViT-300M          (304M params, 24 layers)
│   └── LoRA on fc1, fc2
├── MLP projector            (12.6M params)
│   └── pixel unshuffle + Linear
└── InternLM2-1.8B          (1.8B params, 24 layers)
    └── LoRA on w1, w2, w3
```

**Data flow:** 112x112 face &rarr; resize 448 &rarr; ImageNet norm &rarr; ViT &rarr; pixel unshuffle &rarr; 256 tokens &rarr; MLP &rarr; LLM &rarr; hidden states

## Setup

```bash
conda env create -f environment.yaml
conda activate face-verification
cp .env.example .env   # fill in dataset paths + HPC credentials
```

### Data

- **MS1M-ArcFace** (training): Download from [Kaggle](https://www.kaggle.com/datasets/jadesag3/ms1m-arcface/data). Set `MS1M_DIR` in `.env`. LMDBs are built automatically on first run.
- **LFW** (eval): Auto-downloaded and cached on first eval.
- **CFP** (eval): Download from [Kaggle](https://www.kaggle.com/datasets/chinafax/cfpw-dataset) to `data/cfp-dataset/`.

### Model weights

```bash
python scripts/download_internvl.py   # downloads InternVL2-2B (~4.4 GB)
```

## Training

```bash
# Track 1: ViT encoder + LoRA
accelerate launch train.py --config internvit-lora

# Track 2: MLLM pairwise classifier
accelerate launch train_pairwise.py --config internvl-pair-lora

# Track 3: MLLM siamese embeddings
accelerate launch train.py --config internvl-lora
```

## HPC (SLURM)

```bash
cd hpc
bash estimate.sh            # validate all tracks + estimate training time
bash internvit-lora.sh      # submit Track 1
bash internvl-lora.sh       # submit Track 3
bash internvl-pair-lora.sh  # submit Track 2
```

## Evaluation

```bash
python eval_checkpoint.py --config internvit-lora
```

Evaluates on LFW and CFP-FF/FP with 10-fold cross-validation at FAR@0.001.

## Project Structure

```
train.py                  # Track 1 & 3 training (embedding-based)
train_pairwise.py         # Track 2 training (pairwise classifier)
eval_checkpoint.py        # Standalone evaluation
configs/                  # YAML configs per experiment
src/
  backbones/              # InternViT, InternVL, InternVL-Pair, LoRA
  data/                   # LMDB dataset, pair sampler, MS1M preparation
  eval/                   # LFW/CFP evaluation with 10-fold CV
  losses/                 # InfoNCE loss
  training.py             # Shared training scaffold
  utils.py                # LR scheduling, checkpointing, config overrides
hpc/                      # SLURM job scripts
scripts/                  # Model download, analysis tools
tests/                    # Unit tests (27 tests)
```
