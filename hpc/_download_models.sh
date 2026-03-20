#!/bin/bash
# Pre-download model weights for offline use on compute nodes.
# Run ONCE on a login node (has internet) before submitting training jobs.
#
# Usage (from project root):
#   bash hpc/_download_models.sh

cd "$(dirname "$0")/.."
echo "Project directory: $(pwd)"

# Use venv if available, otherwise fall back to system Python
if [ -f "../face-verification-env/bin/activate" ]; then
    source ../face-verification-env/bin/activate
fi

# Ensure huggingface_hub is available
python -c "import huggingface_hub" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "huggingface_hub not found, installing..."
    pip install --user huggingface_hub
fi

echo "=== Downloading model weights to data/checkpoints/ ==="

# InternVL2-2B (~4.4GB) — used by all three InternVL tracks
python scripts/download_internvl.py

echo ""
echo "=== Cached files ==="
du -sh data/checkpoints/*/ 2>/dev/null
echo ""
echo "=== All models cached. Ready for offline compute nodes. ==="
