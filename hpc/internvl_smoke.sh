#!/bin/bash
#SBATCH --job-name=ivl_smoke
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --partition=debug
#SBATCH --gpus-per-node=4
#SBATCH --output=/scratch/enazari/slurm-logs/%j-%x.out

module purge
module load StdEnv/2023 gcc cuda/12.2 cudnn python/3.11 opencv/4.8.1

cd ~/links/projects/def-pbranco/enazari/mllm-fv/LLM-FaceVerification

mkdir -p $SLURM_TMPDIR/data
cp -r data/checkpoints $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_train.lmdb $SLURM_TMPDIR/data/
cp -r data/ms1m_1000_val.lmdb $SLURM_TMPDIR/data/

export DATA_DIR=$SLURM_TMPDIR/data
source /scratch/enazari/face-verification-env/bin/activate
export HF_HUB_OFFLINE=1
export HF_HOME=/scratch/enazari/hf_cache
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

STEPS=20
START=$(date +%s)
srun python train.py --config internvl-lora --max-steps $STEPS --skip-eval \
    --override data.num_identities=1000 training.epochs=1 training.warmup_epochs=0 \
               training.num_workers=0 session=_smoke/internvl-lora
END=$(date +%s)
ELAPSED=$((END - START))
TIME_PER_STEP=$(echo "scale=2; $ELAPSED / $STEPS" | bc)
echo "SMOKE: ${ELAPSED}s for $STEPS steps (${TIME_PER_STEP} s/step)"
