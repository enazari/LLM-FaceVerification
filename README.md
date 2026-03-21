
Track	What	Input	Loss	Output	Question
1	ViT encoder only	1 image → embedding	InfoNCE	[B, 1024]	Does instruction-tuned InternViT beat standalone DINOv2/CLIP?
2	Full MLLM classifier	2 images + prompt → Yes/No	Softmax (CE)	P("Yes") as score	Can the MLLM learn verification through its natural text interface?
3	Full MLLM embedding	1 image + prompt → LLM hidden state	InfoNCE	[B, 2048]	Are LLM hidden states useful as face embeddings?

## Setup

```bash
conda env create -f environment.yaml
conda activate face-verification
cp .env.example .env  # fill in your dataset paths and HPC credentials
```

## Data

### MS1M-ArcFace (training)

1. Download from Kaggle: https://www.kaggle.com/datasets/jadesag3/ms1m-arcface/data
2. Extract &mdash; you need `train.rec` and `train.idx` (RecordIO format).
3. Set `MS1M_DIR` in `.env` to the extracted folder.

`train.py` automatically converts RecordIO to LMDB on first run.

### LFW (evaluation)

No manual download needed. `train.py` auto-downloads LFW from figshare, runs RetinaFace alignment, and caches the result as `data/lfw_10fold_original_retinaface.lmdb`.

### CFP (evaluation)

Download from Kaggle: https://www.kaggle.com/datasets/chinafax/cfpw-dataset and extract to `data/cfp-dataset/`. LMDBs are prepared automatically on first run.

## Run

```bash
accelerate launch train.py --config arc
```

## SLURM cluster

```bash
sbatch hpc/_requirements_installation.sh   # one-time env setup
cd hpc && bash _download_models.sh          # pre-download weights (login node, has internet)
sbatch hpc/arc.sh                           # submit training job
```