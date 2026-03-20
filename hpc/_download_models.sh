#!/bin/bash
# Pre-download model weights for offline use on compute nodes.
# Run ONCE on a login node (has internet) before submitting training jobs.
#
# Usage:
#   cd hpc && bash _download_models.sh

cd ..
source ../face-verification-env/bin/activate

echo "=== Downloading model weights to data/checkpoints/ ==="

# InternVL2-2B (~4.4GB) — used by all three InternVL tracks
python scripts/download_internvl.py

echo ""
echo "=== Cached files ==="
du -sh data/checkpoints/*/ 2>/dev/null
echo ""
echo "=== All models cached. Ready for offline compute nodes. ==="
