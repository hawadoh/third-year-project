#!/bin/bash
#
#SBATCH --job-name=tournament-n4         # Job name
#SBATCH --partition=gecko,falcon         # Prefer gecko (A10 GPUs), fallback to falcon
#SBATCH --array=1-9                      # 3 datasets × 3 seeds = 9 jobs
#SBATCH --ntasks=1                       # One task per job
#SBATCH --cpus-per-task=12               # CPU threads
#SBATCH --gres=gpu:1                     # 1 GPU per job
#SBATCH --mem=32000                      # 32GB RAM
#SBATCH --time=2-00:00:00                # 48 hours max
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=training/logs/tournament_n4_job_%A_task_%a.out
#SBATCH --error=training/logs/tournament_n4_job_%A_task_%a.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

# Activate conda environment
conda activate spm

# Load CUDA
module load CUDA

# Define Tournament n=4 datasets
DATASETS=(
    "Tournament_n4_Baseline_1e4_m1e7_b210"
    "Tournament_n4_TL_1e4_m1e7_b210"
    "Tournament_n4_ATL_1e4_m1e7_b210"
)
SEEDS=(0 1 2)

# Map SLURM_ARRAY_TASK_ID to (dataset, seed) combination
# Task IDs 1-9 map to: seed_idx = (task_id - 1) / 3, dataset_idx = (task_id - 1) % 3
# This processes all datasets for seed 0, then seed 1, then seed 2
TASK_ID=$SLURM_ARRAY_TASK_ID
SEED_IDX=$(( ($TASK_ID - 1) / 3 ))
DATASET_IDX=$(( ($TASK_ID - 1) % 3 ))

DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "================================================================"
echo "TOURNAMENT GCD TRAINING (n=4)"
echo "================================================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Index: $DATASET_IDX -> $DATASET"
echo "Seed Index: $SEED_IDX -> Seed $SEED"
echo "Node: $SLURM_NODELIST"
echo "================================================================"

# Ensure symlink to /dcs/large data exists in workspace
WORKSPACE_DIR="/dcs/23/u5514611/cs310/third-year-project"
LARGE_DATA="/dcs/large/u5514611/data"
WORKSPACE_DATA="$WORKSPACE_DIR/data"

# Create workspace data directory if it doesn't exist
mkdir -p "$WORKSPACE_DATA"

# Create symlink for this dataset if needed
DATASET_LINK="$WORKSPACE_DATA/$DATASET"
DATASET_SOURCE="$LARGE_DATA/$DATASET"

if [ ! -e "$DATASET_LINK" ] && [ -d "$DATASET_SOURCE" ]; then
    echo "Creating symlink: $DATASET_LINK -> $DATASET_SOURCE"
    ln -sfn "$DATASET_SOURCE" "$DATASET_LINK"
elif [ -d "$DATASET_SOURCE" ]; then
    echo "Dataset symlink/directory already exists: $DATASET_LINK"
else
    echo "ERROR: Dataset not found at $DATASET_SOURCE"
    echo "Generate tournament data first with:"
    echo "  SPM_DATA_DIR=/dcs/large/u5514611/data python -m spm.data.tournament.generate_tournament_data --n_values 4"
    exit 1
fi

# Run training script
cd /dcs/23/u5514611/cs310/third-year-project/training
bash train_single.sh "$DATASET" "$SEED"

echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
