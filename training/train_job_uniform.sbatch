#!/bin/bash
#
#SBATCH --job-name=spm-uniform            # Job name
#SBATCH --partition=gecko,falcon          # Prefer gecko (A10 GPUs), fallback to falcon
#SBATCH --array=1-9                       # 3 datasets × 3 seeds = 9 jobs
#SBATCH --ntasks=1                        # One task per job
#SBATCH --cpus-per-task=12                # CPU threads
#SBATCH --gres=gpu:1                      # 1 GPU per job
#SBATCH --mem=32000                       # 32GB RAM
#SBATCH --time=2-00:00:00                 # 48 hours max
#
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=logs/uniform_job_%A_task_%a.out
#SBATCH --error=logs/uniform_job_%A_task_%a.err

## Initialisation ##
source /etc/profile.d/modules.sh
source /etc/profile.d/conda.sh

# Activate conda environment
conda activate spm

# Load CUDA
module load CUDA

# Define UNIFORM datasets (3 datasets for distribution investigation)
DATASETS=("Baseline_uniform_1e4_m1e7_b210" "TL_uniform_1e4_m1e7_b210" "ATL7_uniform_1e4_m1e7_b210")
SEEDS=(0 1 2)

# Map SLURM_ARRAY_TASK_ID to (dataset, seed) combination
# Task IDs 1-9 map to: dataset_idx = (task_id - 1) / 3, seed_idx = (task_id - 1) % 3
TASK_ID=$SLURM_ARRAY_TASK_ID
DATASET_IDX=$(( ($TASK_ID - 1) / 3 ))
SEED_IDX=$(( ($TASK_ID - 1) % 3 ))

DATASET=${DATASETS[$DATASET_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "================================================================"
echo "UNIFORM DISTRIBUTION TRAINING"
echo "================================================================"
echo "SLURM Job Array Task: $SLURM_ARRAY_TASK_ID of $SLURM_ARRAY_TASK_COUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Dataset Index: $DATASET_IDX -> $DATASET"
echo "Seed Index: $SEED_IDX -> Seed $SEED"
echo "Node: $SLURM_NODELIST"
echo "================================================================"

# Run training script
cd /dcs/23/u5514611/cs310/third-year-project/training
bash train_single.sh "$DATASET" "$SEED"

echo "Task $SLURM_ARRAY_TASK_ID completed successfully"
