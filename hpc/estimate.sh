#!/bin/bash
#SBATCH --job-name=estimate
#SBATCH --time=00:45:00
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --mail-type=ALL

# Validates all 3 tracks on 2 nodes × 4 GPUs and estimates full training time.

# --- Self-submit: run `bash hpc/estimate.sh` from hpc/ ---
if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env"; exit 1; }
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
    exit $?
fi

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

cd ..
echo "Project directory: $(pwd)"

NUM_GPUS=$((SLURM_NNODES * 4))
echo "Nodes: $SLURM_NNODES, GPUs/node: 4, Total GPUs: $NUM_GPUS"
echo "Nodelist: $SLURM_JOB_NODELIST"

# Copy data to fast local storage
mkdir -p $SLURM_TMPDIR/data
cp -r data/checkpoints $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_val.lmdb $SLURM_TMPDIR/data/

export DATA_DIR=$SLURM_TMPDIR/data
source ../face-verification-env/bin/activate
export HF_HUB_OFFLINE=1

# Multi-node rendezvous
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

STEPS=50
PASS=0
FAIL=0

echo ""
echo "================================================================"
echo "  Training Time Estimator — $STEPS steps, $NUM_GPUS GPUs"
echo "================================================================"

run_estimate() {
    local name="$1"
    local script="$2"
    local config="$3"
    local steps_per_epoch_1gpu="$4"
    local epochs="$5"
    shift 5

    echo ""
    echo "=== $name ==="

    START=$(date +%s)
    if srun accelerate launch \
        --multi_gpu \
        --num_processes=$NUM_GPUS \
        --num_machines=$SLURM_NNODES \
        --main_process_ip=$MASTER_ADDR \
        --main_process_port=$MASTER_PORT \
        --mixed_precision=bf16 \
        "$script" --config "$config" --max-steps "$STEPS" --skip-eval \
        --override data.num_identities=1000 \
                   training.epochs=1 \
                   training.warmup_epochs=0 \
                   training.num_workers=4 \
                   "session=_estimate/$config" \
                   "$@" 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        PASS=$((PASS + 1))

        TIME_PER_STEP=$(echo "scale=3; $ELAPSED / $STEPS" | bc)

        # With DDP, steps/epoch divides by num_gpus
        STEPS_PER_EPOCH=$((steps_per_epoch_1gpu / NUM_GPUS))
        EPOCH_SECS=$(echo "scale=0; $STEPS_PER_EPOCH * $ELAPSED / $STEPS" | bc)
        EPOCH_MIN=$(echo "scale=1; $EPOCH_SECS / 60" | bc)
        TOTAL_SECS=$(echo "scale=0; $EPOCH_SECS * $epochs" | bc)
        TOTAL_H=$(echo "scale=1; $TOTAL_SECS / 3600" | bc)

        echo ""
        echo "  PASS: $name"
        echo "  -------------------------------------------------------"
        echo "  Measured: ${ELAPSED}s for $STEPS steps (${TIME_PER_STEP} s/step)"
        echo "  Steps/epoch ($NUM_GPUS GPUs): $STEPS_PER_EPOCH"
        echo "  Est. per epoch: ${EPOCH_MIN} min"
        echo "  Est. total ($epochs epochs, $NUM_GPUS GPUs): ${TOTAL_H} hours"
        echo "  -------------------------------------------------------"
    else
        END=$(date +%s)
        ELAPSED=$((END - START))
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name (exit code $?, ${ELAPSED}s)"
    fi
}

# Track 1: InternViT-LoRA
# PairSampler: batch=200, P=100 → ~23285 steps/epoch (1 GPU), 15 epochs
run_estimate "Track 1: InternViT-LoRA" \
    train.py internvit-lora \
    23285 15

# Track 3: InternVL-LoRA (full MLLM siamese)
# PairSampler: batch=32, P=16 → ~146617 steps/epoch (1 GPU), 10 epochs
run_estimate "Track 3: InternVL-LoRA" \
    train.py internvl-lora \
    146617 10

# Track 2: InternVL-Pair-LoRA (pairwise classifier)
# 100k pairs / batch=16 = 6250 steps/epoch (1 GPU), 10 epochs
run_estimate "Track 2: InternVL-Pair-LoRA" \
    train_pairwise.py internvl-pair-lora \
    6250 10

echo ""
echo "================================================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "================================================================"

rm -rf sessions/_estimate
echo "Done."

[ "$FAIL" -eq 0 ]
