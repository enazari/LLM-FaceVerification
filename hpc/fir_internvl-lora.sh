#!/bin/bash
#SBATCH --job-name=internvl-lora
#SBATCH --time=7-00:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --partition=gpubase_bynode_b5
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=/scratch/enazari/slurm-logs/%j-%x.out
#SBATCH --mail-type=ALL

# Fir — InternVL-LoRA full training (2 nodes × 4 H100s).
# Resubmitting this script resumes automatically from last checkpoint.

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
module load StdEnv/2023 gcc/12.3 cuda/12.6 cudnn python/3.11 opencv/4.8.1

cd /home/enazari/projects/rrg-pbranco/enazari/mllm-fv/LLM-FaceVerification

echo "Project directory: $(pwd)"
echo "Config: $CONFIG_NAME | Session: $SESSION_DIR"
echo "Nodes: $SLURM_NNODES ($SLURM_JOB_NODELIST) | GPUs: $SLURM_NTASKS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -4

# Copy model checkpoint to fast local NVMe on every node
srun --ntasks-per-node=1 --ntasks=$SLURM_NNODES bash -c "
    mkdir -p \$SLURM_TMPDIR/data/checkpoints
    cp -r data/checkpoints/InternVL2-2B \$SLURM_TMPDIR/data/checkpoints/
    echo \"Node \$(hostname): checkpoint ready\"
"

export DATA_DIR=$SLURM_TMPDIR/data
export FIR_LMDB=/project/def-pbranco/enazari/datasets/MTCNN-processed_MS1M_ArcFace.lmdb
source /scratch/enazari/fir-face-verification-env/bin/activate
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
