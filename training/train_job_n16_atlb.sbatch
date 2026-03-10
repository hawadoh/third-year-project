#!/bin/bash
#
#SBATCH --job-name=n16-atlb              # Job name
#SBATCH --partition=gecko,falcon          # Prefer gecko (A10 GPUs), fallback to falcon
#SBATCH --array=1                         # Which batch to run (1=0-25k, 2=25k-50k, 3=50k-75k, 4=75k-100k)
#SBATCH --ntasks=1                        # One task per job
#SBATCH --cpus-per-task=12               # CPU threads
#SBATCH --gres=gpu:1                     # 1 GPU per job
#SBATCH --mem=32000                      # 32GB RAM
#SBATCH --time=2-00:00:00               # 48 hours max
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=training/logs/n16_atlb_job_%A_batch_%a.out
#SBATCH --error=training/logs/n16_atlb_job_%A_batch_%a.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate spm
module load CUDA

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET="Tournament_n16_ATL_B_T5_1e4_m1e7_b210_p0.5"
SEED=0
BATCH=$SLURM_ARRAY_TASK_ID
RUN_NAME="8x8x256-${DATASET}"

# Checkpoint to resume from (batches 2-4 load previous batch's checkpoint)
CHECKPOINTS=(
    ""                              # batch 1: train from scratch
    "${RUN_NAME}_iter25000"         # batch 2: resume from 25k
    "${RUN_NAME}_iter50000"         # batch 3: resume from 50k
    "${RUN_NAME}_iter75000"         # batch 4: resume from 75k
)

export SPM_LOAD_CKPT="${CHECKPOINTS[$(( BATCH - 1 ))]}"
export SPM_SAVE_ITERS="25000 50000 75000 100000"
export SPM_BATCH_SIZE=256
export SPM_GRAD_ACCUM_STEPS=4

echo "================================================================"
echo "n=16 ATL-B TRAINING (batch $BATCH/4)"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
echo "Node: $SLURM_NODELIST"
echo "Load checkpoint: ${SPM_LOAD_CKPT:-none (fresh start)}"
echo "Save iters: $SPM_SAVE_ITERS"
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

echo "Batch $BATCH completed successfully"
