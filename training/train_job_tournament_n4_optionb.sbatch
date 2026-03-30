#!/bin/bash
#
#SBATCH --job-name=tournament-n4-optionb  # Job name
#SBATCH --partition=gecko,falcon          # Prefer gecko (A10 GPUs), fallback to falcon
#SBATCH --array=1-3                       # 1 dataset × 3 seeds = 3 jobs
#SBATCH --ntasks=1                        # One task per job
#SBATCH --cpus-per-task=12               # CPU threads
#SBATCH --gres=gpu:1                     # 1 GPU per job
#SBATCH --mem=32000                      # 32GB RAM
#SBATCH --time=2-00:00:00               # 48 hours max
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=training/logs/tournament_n4_optionb_job_%A_task_%a.out
#SBATCH --error=training/logs/tournament_n4_optionb_job_%A_task_%a.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

conda activate spm
module load CUDA

DATASET="Tournament_n4_ATL_B_T6_1e4_m1e7_b210"
SEEDS=(0 1 2)
SEED=${SEEDS[$(( $SLURM_ARRAY_TASK_ID - 1 ))]}

echo "================================================================"
echo "TOURNAMENT GCD OPTION B TRAINING (n=4)"
echo "================================================================"
echo "SLURM Array Task: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset: $DATASET"
echo "Seed: $SEED"
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

echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
