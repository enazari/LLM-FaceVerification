#!/bin/bash
# Pull training results from Trillium GPU cluster.
# Run this after maintenance ends to get the latest checkpoints and logs.
# Usage: bash results/sync_results.sh [session_dir]

REMOTE="enazari@trillium-gpu.alliancecan.ca"
PROJ="/project/def-pbranco/enazari/mllm-fv/LLM-FaceVerification"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Sessions to sync (pass as arg or sync all)
SESSION="${1:-}"

echo "=== Syncing SLURM logs ==="
rsync -avz --progress \
    -e "ssh -o ControlPath=~/.ssh/trillium-gpu.sock" \
    "$REMOTE:/scratch/enazari/slurm-logs/" \
    "$LOCAL_DIR/slurm-logs/"

echo ""
echo "=== Syncing training sessions ==="
if [ -n "$SESSION" ]; then
    rsync -avz --progress \
        -e "ssh -o ControlPath=~/.ssh/trillium-gpu.sock" \
        "$REMOTE:$PROJ/sessions/$SESSION/" \
        "$LOCAL_DIR/sessions/$SESSION/"
else
    rsync -avz --progress --exclude='*.lmdb' \
        -e "ssh -o ControlPath=~/.ssh/trillium-gpu.sock" \
        "$REMOTE:$PROJ/sessions/" \
        "$LOCAL_DIR/sessions/"
fi

echo ""
echo "=== Done. Results in: $LOCAL_DIR ==="
echo ""
echo "Sessions synced:"
ls "$LOCAL_DIR/sessions/" 2>/dev/null | sed 's/^/  /'
