#!/bin/bash
#SBATCH --job-name=fv_test
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --output=/scratch/enazari/slurm-logs/%j-%x.out
#SBATCH --mail-type=ALL

# Single-node smoke test for Track 1 (internvit-lora).
# Runs 50 steps to verify training works and estimate time on Trillium GPU nodes.

if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found."; exit 1; }
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
    exit $?
fi

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

echo "Project directory: $(pwd)"
echo "Nodes: $SLURM_NNODES, Nodelist: $SLURM_JOB_NODELIST"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4

mkdir -p $SLURM_TMPDIR/data
cp -r data/checkpoints $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_val.lmdb $SLURM_TMPDIR/data/
echo "Data copied."

export DATA_DIR=$SLURM_TMPDIR/data
source /scratch/enazari/face-verification-env/bin/activate
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/enazari/hf_cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

STEPS=50
NUM_GPUS=4
echo ""
echo "=== Track 1: InternViT-LoRA — $STEPS steps smoke test ==="

START=$(date +%s)
srun python train.py --config internvit-lora --max-steps $STEPS --skip-eval \
    --override data.num_identities=1000 \
               training.epochs=1 \
               training.warmup_epochs=0 \
               training.num_workers=0 \
               session=_test/internvit-lora
STATUS=$?
END=$(date +%s)
ELAPSED=$((END - START))

if [ $STATUS -eq 0 ]; then
    TIME_PER_STEP=$(echo "scale=3; $ELAPSED / $STEPS" | bc)
    STEPS_PER_EPOCH=$((23285 / NUM_GPUS))
    EPOCH_SECS=$(echo "scale=0; $STEPS_PER_EPOCH * $ELAPSED / $STEPS" | bc)
    TOTAL_H=$(echo "scale=1; $EPOCH_SECS * 15 / 3600" | bc)
    echo ""
    echo "PASS: ${ELAPSED}s for $STEPS steps (${TIME_PER_STEP} s/step)"
    echo "Est. steps/epoch (4 GPUs): $STEPS_PER_EPOCH"
    echo "Est. total (15 epochs, 4 GPUs): ${TOTAL_H} hours"
    rm -rf sessions/_test
    exit 0
else
    echo "FAIL: exit code $STATUS after ${ELAPSED}s"
    exit 1
fi
