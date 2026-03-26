#!/bin/bash
#SBATCH --job-name=estimate_narval
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=a100:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=0
#SBATCH --mail-type=ALL

# Estimates InternVL-LoRA training time on Narval (2 nodes × 4 A100 GPUs).
# ntasks-per-node=4: SLURM launches 1 process per GPU, Accelerator() initializes from env.

# --- Self-submit: run `bash estimate_narval.sh` from hpc/ ---
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

GPUS_PER_NODE=4
NUM_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))
echo "Nodes: $SLURM_NNODES, GPUs/node: $GPUS_PER_NODE, Total GPUs: $NUM_GPUS"
echo "Nodelist: $SLURM_JOB_NODELIST"

# Copy data to fast local storage on EVERY node
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c '
    mkdir -p $SLURM_TMPDIR/data
    cp -r data/checkpoints $SLURM_TMPDIR/data/
    cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/
    cp -r data/ms1m_1000_val.lmdb $SLURM_TMPDIR/data/
    echo "Node $(hostname): data copied to $SLURM_TMPDIR/data"
'

export DATA_DIR=$SLURM_TMPDIR/data
source ../face-verification-env/bin/activate
export HF_HUB_OFFLINE=1

# Multi-node rendezvous
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500

STEPS=200
PASS=0
FAIL=0

echo ""
echo "================================================================"
echo "  InternVL-LoRA Estimate — $STEPS steps, $NUM_GPUS GPUs (Narval)"
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
    if srun python \
        $script --config $config --max-steps $STEPS --skip-eval \
        --override data.num_identities=1000 \
                   training.epochs=1 \
                   training.warmup_epochs=0 \
                   training.num_workers=4 \
                   session=_estimate/$config \
                   $@ 2>&1; then
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

# InternVL-LoRA (full MLLM siamese)
run_estimate "InternVL-LoRA" \
    train.py internvl-lora \
    146617 10

echo ""
echo "================================================================"
echo "  Results: $PASS passed, $FAIL failed"
echo "================================================================"

rm -rf sessions/_estimate
echo "Done."

[ "$FAIL" -eq 0 ]
