#!/bin/bash
#SBATCH --job-name=internvl-lora
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --partition=compute_full_node
#SBATCH --gpus-per-node=4
#SBATCH --output=/scratch/enazari/slurm-logs/%j-%x.out
#SBATCH --mail-type=ALL

# Trillium GPU — InternVL-LoRA full training.
# Resubmitting this script resumes automatically from the last checkpoint.

CONFIG_NAME="internvl-lora"
export SESSION_DIR="sessions/internvl_lora"

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
echo "Config: $CONFIG_NAME | Session: $SESSION_DIR"
echo "Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST) | GPUs: $SLURM_NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4

N=$(grep "num_identities" configs/$CONFIG_NAME.yaml | awk '{print $2}')

# Copy data to fast local storage on every node
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c "
    mkdir -p \$SLURM_TMPDIR/data
    cp -r data/checkpoints \$SLURM_TMPDIR/data/
    cp -r data/ms1m_${N}_train.lmdb \$SLURM_TMPDIR/data/
    cp -r data/ms1m_${N}_val.lmdb \$SLURM_TMPDIR/data/
    cp -r data/lfw_10fold_original_retinaface.lmdb \$SLURM_TMPDIR/data/ 2>/dev/null || true
    cp -r data/cfp_ff_10fold_retinaface.lmdb \$SLURM_TMPDIR/data/ 2>/dev/null || true
    cp -r data/cfp_fp_10fold_retinaface.lmdb \$SLURM_TMPDIR/data/ 2>/dev/null || true
    echo \"Node \$(hostname): data ready (\$(du -sh \$SLURM_TMPDIR/data | cut -f1))\"
"

export DATA_DIR=$SLURM_TMPDIR/data
source /scratch/enazari/face-verification-env/bin/activate
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/enazari/hf_cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

[ -f "$SESSION_DIR/last.pth" ] && echo "Resuming from $SESSION_DIR/last.pth" || echo "Starting fresh"

srun python train.py --config "$CONFIG_NAME"
STATUS=$?

[ $STATUS -eq 0 ] && echo "Done." || echo "FAILED with exit code $STATUS"
exit $STATUS
