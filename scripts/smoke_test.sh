#!/usr/bin/env bash
# Smoke test: run 3 gradient steps for each track locally before pushing to HPC.
# Usage: bash scripts/smoke_test.sh
set -uo pipefail
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
SMOKE_DIR="sessions/_smoke"

run_track() {
    local name="$1" script="$2" config="$3"
    shift 3
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
        echo "  FAIL: $name"
        FAIL=$((FAIL + 1))
    fi
    echo ""
}

# Track 1: InternViT encoder + LoRA + InfoNCE
run_track "Track 1: InternViT-LoRA" train.py internvit-lora

# Track 3: Full InternVL siamese + LoRA + InfoNCE (batch=2 for VRAM)
run_track "Track 3: InternVL-LoRA" train.py internvl-lora \
    training.batch_size=2

# Track 2: Full InternVL pairwise classifier (batch=2, few pairs)
run_track "Track 2: InternVL-Pair-LoRA" train_pairwise.py internvl-pair-lora \
    training.batch_size=2 training.pairs_per_epoch=100

echo "=== Results: $PASS passed, $FAIL failed ==="

# Cleanup
rm -rf "$SMOKE_DIR"

[ "$FAIL" -eq 0 ]
