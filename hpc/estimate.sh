#!/bin/bash
#SBATCH --job-name=estimate
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL

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

# Copy data to fast local storage (1000-subset for loading, checkpoints for model)
mkdir -p $SLURM_TMPDIR/data
cp -r data/checkpoints $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_val.lmdb $SLURM_TMPDIR/data/

export DATA_DIR=$SLURM_TMPDIR/data
source ../face-verification-env/bin/activate
export HF_HUB_OFFLINE=1

STEPS=50
PASS=0
FAIL=0

echo ""
echo "================================================================"
echo "  Training Time Estimator — $STEPS steps per track"
echo "================================================================"

run_estimate() {
    local name="$1"
    local script="$2"
    local config="$3"
    local steps_per_epoch="$4"
    local epochs="$5"
    shift 5

    echo ""
    echo "=== $name ==="

    START=$(date +%s)
    if python "$script" --config "$config" --max-steps "$STEPS" --skip-eval \
        --override data.num_identities=1000 \
                   training.epochs=1 \
                   training.warmup_epochs=0 \
                   training.num_workers=4 \
                   "session=_estimate/$config" \
                   "$@" 2>&1; then
        END=$(date +%s)
        ELAPSED=$((END - START))
        PASS=$((PASS + 1))

        # Extrapolate
        TIME_PER_STEP=$(echo "scale=2; $ELAPSED / $STEPS" | bc)
        EPOCH_SECS=$(echo "scale=0; $steps_per_epoch * $ELAPSED / $STEPS" | bc)
        TOTAL_SECS=$(echo "scale=0; $EPOCH_SECS * $epochs" | bc)
        TOTAL_HOURS=$(echo "scale=1; $TOTAL_SECS / 3600" | bc)
        EPOCH_MINS=$(echo "scale=1; $EPOCH_SECS / 60" | bc)

        echo ""
        echo "  PASS: $name"
        echo "  -------------------------------------------------------"
        echo "  Measured: ${ELAPSED}s for $STEPS steps (${TIME_PER_STEP}s/step)"
        echo "  Steps/epoch: $steps_per_epoch"
        echo "  Est. per epoch: ${EPOCH_MINS} min"
        echo "  Est. total ($epochs epochs): ${TOTAL_HOURS} hours"
        echo "  -------------------------------------------------------"
    else
        END=$(date +%s)
        ELAPSED=$((END - START))
        FAIL=$((FAIL + 1))
        echo "  FAIL: $name (exit code $?, ${ELAPSED}s)"
    fi
}

# Track 1: InternViT-LoRA
# PairSampler: ~85742 identities, P=100 (batch=200) → ~23285 steps/epoch, 15 epochs
run_estimate "Track 1: InternViT-LoRA" \
    train.py internvit-lora \
    23285 15

# Track 3: InternVL-LoRA (full MLLM siamese)
# PairSampler: ~85742 ids, P=16 (batch=32) → ~146617 steps/epoch, 10 epochs
# With grad_accum=8: effective optimizer steps = 146617/8 ≈ 18327, but wall time = 146617 forward passes
run_estimate "Track 3: InternVL-LoRA" \
    train.py internvl-lora \
    146617 10

# Track 2: InternVL-Pair-LoRA (pairwise classifier)
# 100k pairs / batch_size=16 = 6250 steps/epoch, 10 epochs
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
