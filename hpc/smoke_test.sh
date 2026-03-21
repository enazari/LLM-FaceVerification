#!/bin/bash
#SBATCH --job-name=smoke-test
#SBATCH --time=00:25:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

# --- Self-submit: run `bash hpc/smoke_test.sh` from hpc/ or project root ---
if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env"; exit 1; }
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
    exit $?
fi

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/.."
echo "Project directory: $(pwd)"

# Copy minimal data to fast local storage
mkdir -p $SLURM_TMPDIR/data
cp -r data/checkpoints $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/

export DATA_DIR=$SLURM_TMPDIR/data
source ../face-verification-env/bin/activate
export HF_HUB_OFFLINE=1

PASS=0
FAIL=0

run_track() {
    local name="$1" script="$2" config="$3"
    shift 3
    echo ""
    echo "=== $name ==="
    if python "$script" --config "$config" --max-steps 3 --skip-eval \
        --override data.num_identities=1000 \
                   training.batch_size=4 \
                   training.epochs=1 \
                   training.warmup_epochs=0 \
                   training.num_workers=2 \
                   training.gradient_accumulation_steps=1 \
                   "session=_smoke/$config" \
                   "$@" 2>&1; then
        echo "  PASS: $name"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $name (exit code $?)"
        FAIL=$((FAIL + 1))
    fi
}

run_track "Track 1: InternViT-LoRA" train.py internvit-lora
run_track "Track 3: InternVL-LoRA" train.py internvl-lora training.batch_size=2
run_track "Track 2: InternVL-Pair-LoRA" train_pairwise.py internvl-pair-lora training.batch_size=2 training.pairs_per_epoch=100

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

rm -rf sessions/_smoke
echo "Done."

[ "$FAIL" -eq 0 ]
