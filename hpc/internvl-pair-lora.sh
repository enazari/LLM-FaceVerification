#!/bin/bash
#SBATCH --job-name=internvl-pair-lora
#SBATCH --time=16:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

# --- Self-submit: run `bash hpc/internvl-pair-lora.sh` from project root ---
if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env"; exit 1; }
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
    exit $?
fi

CONFIG_NAME="internvl-pair-lora"

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

cd ..
echo "Project directory: $(pwd)"

N=$(grep "num_identities" configs/$CONFIG_NAME.yaml | awk '{print $2}')
echo "Config: $CONFIG_NAME, num_identities: $N"

# Copy data to fast local storage
mkdir -p $SLURM_TMPDIR/data

# Copy checkpoints (InternVL2-2B needed offline)
cp -r data/checkpoints $SLURM_TMPDIR/data/

cp -r data/ms1m_${N}_train.lmdb $SLURM_TMPDIR/data/
cp -r data/ms1m_${N}_val.lmdb $SLURM_TMPDIR/data/
cp -r data/lfw_10fold_original_retinaface.lmdb $SLURM_TMPDIR/data/ 2>/dev/null
cp -r data/cfp_ff_10fold_retinaface.lmdb $SLURM_TMPDIR/data/ 2>/dev/null
cp -r data/cfp_fp_10fold_retinaface.lmdb $SLURM_TMPDIR/data/ 2>/dev/null

echo "Local data:"
du -sh $SLURM_TMPDIR/data/*

export DATA_DIR=$SLURM_TMPDIR/data

source ../face-verification-env/bin/activate
export HF_HUB_OFFLINE=1

echo "Starting training: $CONFIG_NAME"
accelerate launch train_pairwise.py --config $CONFIG_NAME

echo "Done."
