#!/bin/bash
#SBATCH --job-name=fv_install
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --mail-type=ALL

# Narval variant of requirements installation.
# Run: bash hpc/_requirements_installation_narval.sh  (from project root)

if [ -z "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
    source "$PROJECT_ROOT/.env" 2>/dev/null || { echo "ERROR: .env not found. cp .env.example .env"; exit 1; }
    LOG_DIR="$SCRATCH/slurm-logs"
    mkdir -p "$LOG_DIR"
    sbatch --account="$SLURM_ACCOUNT" --mail-user="$SLURM_MAIL_USER" \
           --output="$LOG_DIR/%x-%j.out" "$0"
    exit $?
fi

echo "=== FACE-VERIFICATION ENVIRONMENT INSTALLATION (Narval) ==="

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1
echo "Loaded modules:"; module list

PERSISTENT_DIR=$SCRATCH/narval-face-verification-env
echo "Persistent venv: $PERSISTENT_DIR"

if [ -d "$PERSISTENT_DIR" ]; then
    echo "Removing existing venv..."
    rm -rf "$PERSISTENT_DIR"
fi

unset PYTHONPATH; export PYTHONPATH=""

# Build on fast local NVMe, copy to scratch when done
BUILD_ENV=$SLURM_TMPDIR/narval-face-verification-env
python -m venv --system-site-packages $BUILD_ENV
source $BUILD_ENV/bin/activate
echo "Build venv: $VIRTUAL_ENV"

pip install --no-index --upgrade pip

echo "Installing numpy..."
pip install --no-index 'numpy<2'

echo "Installing torch 2.9.x (flash-attn binary requires <2.10)..."
pip install --no-index 'torch<2.10'

echo "Installing torchvision to match torch 2.9.x..."
pip install --no-index 'torchvision<0.25'

echo "Installing flash-attn..."
pip install --no-index flash-attn

echo "Installing ML packages..."
pip install --no-index accelerate lmdb pyyaml pillow tqdm scikit-learn scikit-image einops

echo "Installing transformers + sentencepiece..."
pip install --no-index 'transformers>=4.37.2,<5.0.0' sentencepiece

echo "Installing timm (no-deps to avoid torch upgrade)..."
pip install --no-index --no-deps timm

echo "Verifying..."
python -c "
import torch, torchvision, flash_attn, transformers, lmdb
print(f'torch {torch.__version__}')
print(f'torchvision {torchvision.__version__}')
print(f'flash_attn {flash_attn.__version__}')
print(f'transformers {transformers.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print('All OK.')
"
if [ $? -ne 0 ]; then echo "ERROR: verification failed"; exit 1; fi

deactivate
echo "Copying to $PERSISTENT_DIR ..."
cp -a $BUILD_ENV $PERSISTENT_DIR

# Fix hardcoded paths
sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g" $PERSISTENT_DIR/bin/activate
find $PERSISTENT_DIR/bin -type f -exec grep -l "$BUILD_ENV" {} + 2>/dev/null | \
    xargs -r sed -i "s|$BUILD_ENV|$PERSISTENT_DIR|g"

source $PERSISTENT_DIR/bin/activate
python -c "import torch, flash_attn; print(f'Persistent venv OK — torch {torch.__version__}, flash_attn {flash_attn.__version__}')"

echo "=== INSTALLATION COMPLETE: $PERSISTENT_DIR ==="
