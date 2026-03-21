#!/bin/bash
#SBATCH --job-name=fv_install
#SBATCH --time=00:45:00
#SBATCH --mem=15G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL

# --- Self-submit: run `bash hpc/_requirements_installation.sh` from project root ---
if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env"; exit 1; }
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" "$0"
    exit $?
fi

echo "=== FACE-VERIFICATION ENVIRONMENT INSTALLATION ==="

# Step 1: Load required modules
echo "Loading required modules..."
module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

echo "Loaded modules:"
module list

# Step 2: Navigate to project parent (env lives alongside project dir)
cd ../..
PERSISTENT_DIR=$(pwd)/face-verification-env
echo "Current directory: $(pwd)"
echo "Persistent venv will be: $PERSISTENT_DIR"

# Remove existing environment if it exists
if [ -d "$PERSISTENT_DIR" ]; then
    echo "Removing existing virtual environment..."
    rm -rf "$PERSISTENT_DIR"
fi

# Step 3: Clear PYTHONPATH
echo "Clearing PYTHONPATH to avoid conflicts..."
unset PYTHONPATH
export PYTHONPATH=""

# Step 4: Build venv on SLURM_TMPDIR (fast local NVMe)
BUILD_ENV=$SLURM_TMPDIR/face-verification-env
echo "Creating virtual environment in SLURM_TMPDIR..."
python -m venv --system-site-packages $BUILD_ENV

if [ ! -d "$BUILD_ENV" ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

source $BUILD_ENV/bin/activate
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: VIRTUAL_ENV not set - activation failed"
    exit 1
fi
echo "Build venv activated: $VIRTUAL_ENV"

# Step 5: Upgrade pip
echo "Upgrading pip..."
pip install --no-index --upgrade pip
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

# Step 6: Install packages

echo "Installing numpy (<2 for opencv module compatibility)..."
pip install --no-index 'numpy<2'
if [ $? -ne 0 ]; then echo "ERROR: Failed to install numpy"; exit 1; fi

echo "Installing PyTorch ecosystem..."
pip install --no-index torch torchvision
if [ $? -ne 0 ]; then echo "ERROR: Failed to install torch/torchvision"; exit 1; fi

echo "Installing scikit-learn..."
pip install --no-index scikit-learn
if [ $? -ne 0 ]; then echo "ERROR: Failed to install scikit-learn"; exit 1; fi

echo "Installing HPC-cached packages..."
pip install --no-index lmdb pyyaml pillow accelerate tqdm scikit-image
if [ $? -ne 0 ]; then
    echo "WARNING: Some HPC-cached packages failed, continuing..."
fi

# transformers + sentencepiece needed for InternVL2-2B
echo "Installing transformers and sentencepiece..."
pip install 'transformers>=4.37.2,<5.0.0' sentencepiece
if [ $? -ne 0 ]; then
    echo "WARNING: transformers/sentencepiece install failed. Try with internet."
fi

# timm + einops needed by InternVL2-2B custom code
echo "Installing timm and einops..."
pip install timm einops
if [ $? -ne 0 ]; then
    echo "WARNING: timm/einops install failed."
fi

# Step 7: Verify packages
echo "Verifying package compatibility..."
python -c "
import numpy as np
print(f'NumPy {np.__version__}')

import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')

import torchvision
print(f'Torchvision {torchvision.__version__}')

import lmdb
print(f'lmdb {lmdb.version()}')

import transformers
print(f'transformers {transformers.__version__}')

import sentencepiece
print(f'sentencepiece OK')

from torchvision.transforms import ToTensor
from PIL import Image
import numpy as np
test_array = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
result = ToTensor()(Image.fromarray(test_array))
print(f'ToTensor transform: {result.shape}')

print('All critical packages working.')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Package verification failed"
    exit 1
fi

# Step 8: Copy venv from SLURM_TMPDIR to persistent Lustre storage
deactivate
echo "Copying venv to persistent storage: $PERSISTENT_DIR ..."
cp -a $BUILD_ENV $PERSISTENT_DIR

# Fix hardcoded paths
echo "Fixing venv paths..."
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.csh
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate.fish
find $PERSISTENT_DIR/bin -type f -exec grep -l "$BUILD_ENV" {} + 2>/dev/null | \
    xargs -r sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g"

# Verify the persistent copy works
source $PERSISTENT_DIR/bin/activate
echo "Persistent venv activated: $VIRTUAL_ENV"
python -c "import torch; print(f'torch {torch.__version__} OK')"
if [ $? -ne 0 ]; then
    echo "ERROR: Persistent venv verification failed"
    exit 1
fi

# Step 9: Summary
echo "=== INSTALLATION SUMMARY ==="
echo "Virtual environment: $VIRTUAL_ENV"
echo "Python: $(which python)"
echo ""
echo "Installed packages:"
pip list
echo ""
echo "=== INSTALLATION COMPLETED ==="
