#!/bin/bash
#
#SBATCH --job-name=n16-atlb-b3           # Job name
#SBATCH --partition=gecko,falcon          # Prefer gecko (A10 GPUs), fallback to falcon
#SBATCH --ntasks=1                        # One task per job
#SBATCH --cpus-per-task=12               # CPU threads
#SBATCH --gres=gpu:1                     # 1 GPU per job
#SBATCH --mem=32000                      # 32GB RAM
#SBATCH --time=2-00:00:00               # 48 hours max
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=training/logs/n16_atlb_50k_75k_%j.out
#SBATCH --error=training/logs/n16_atlb_50k_75k_%j.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate spm
module load CUDA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPM_MODELS_DIR=/dcs/large/u5514611/models

DATASET="Tournament_n16_ATL_B_T5_1e4_m1e7_b210_p0.5"
SEED=0

export SPM_LOAD_CKPT="8x8x256-${DATASET}_iter50000"
export SPM_SAVE_ITERS="75000 100000"
export SPM_BATCH_SIZE=256
export SPM_GRAD_ACCUM_STEPS=4

echo "================================================================"
echo "n=16 ATL-B TRAINING (50k -> 75k)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $DATASET"
echo "Load checkpoint: $SPM_LOAD_CKPT"
echo "Node: $SLURM_NODELIST"
echo "================================================================"

WORKSPACE_DIR="/dcs/23/u5514611/cs310/third-year-project"
LARGE_DATA="/dcs/large/u5514611/data"
WORKSPACE_DATA="$WORKSPACE_DIR/data"

mkdir -p "$WORKSPACE_DATA"

DATASET_LINK="$WORKSPACE_DATA/$DATASET"
DATASET_SOURCE="$LARGE_DATA/$DATASET"

if [ ! -e "$DATASET_LINK" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "Creating symlink: $DATASET_LINK -> $DATASET_SOURCE"
    ln -sfn "$DATASET_SOURCE" "$DATASET_LINK"
elif [ -d "$DATASET_SOURCE" ]; then
    echo "Dataset symlink/directory already exists: $DATASET_LINK"
else
    echo "ERROR: Dataset not found at $DATASET_SOURCE"
    exit 1
fi

cd /dcs/23/u5514611/cs310/third-year-project/training
bash train_single.sh "$DATASET" "$SEED"

echo "Training 50k-75k completed successfully"
