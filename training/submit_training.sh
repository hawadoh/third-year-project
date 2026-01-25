#!/bin/bash
# Submit training jobs for any dataset with full dependency chaining
# Usage: ./submit_training.sh <dataset_name>
#   or: ./submit_training.sh all  (to submit all datasets sequentially)
#
# Available datasets: Baseline, TL, ATL2, ATL3, ATL4, ATL5, ATL6, ATL7
#
# Jobs run sequentially: seed 0 → seed 1 → seed 2 → next dataset seed 0 → ...

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

# Dataset to task ID mapping (from train_job.sbatch array configuration)
declare -A DATASET_TASKS=(
    ["Baseline"]="1 2 3"
    ["TL"]="4 5 6"
    ["ATL2"]="7 8 9"
    ["ATL3"]="10 11 12"
    ["ATL4"]="13 14 15"
    ["ATL5"]="16 17 18"
    ["ATL6"]="19 20 21"
    ["ATL7"]="22 23 24"
)

# Function to submit one dataset's training jobs with dependency on previous job
submit_dataset() {
    local dataset=$1
    local prev_job_id=$2  # Empty string if first dataset
    local full_name="${dataset}_1e4_m1e7_b210"

    # Get task IDs for this dataset
    local tasks=(${DATASET_TASKS[$dataset]})

    echo "----------------------------------------"
    echo "Dataset: $full_name"
    echo "  Task ${tasks[0]}: Seed 0 (saves checkpoints)"
    echo "  Task ${tasks[1]}: Seed 1"
    echo "  Task ${tasks[2]}: Seed 2"

    # Submit seed 0
    if [ -z "$prev_job_id" ]; then
        # First dataset - no dependency
        JOB_ID_0=$(sbatch --parsable --array=${tasks[0]} train_job.sbatch)
        echo "  ✓ Seed 0: Job $JOB_ID_0"
    else
        # Depends on previous dataset's last seed
        JOB_ID_0=$(sbatch --parsable --dependency=afterok:$prev_job_id --array=${tasks[0]} train_job.sbatch)
        echo "  ✓ Seed 0: Job $JOB_ID_0 (after $prev_job_id)"
    fi

    # Submit seed 1 (depends on seed 0)
    JOB_ID_1=$(sbatch --parsable --dependency=afterok:$JOB_ID_0 --array=${tasks[1]} train_job.sbatch)
    echo "  ✓ Seed 1: Job $JOB_ID_1 (after $JOB_ID_0)"

    # Submit seed 2 (depends on seed 1)
    JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --array=${tasks[2]} train_job.sbatch)
    echo "  ✓ Seed 2: Job $JOB_ID_2 (after $JOB_ID_1)"

    # Return the last job ID for chaining
    echo "$JOB_ID_2"
}

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name|all>"
    echo ""
    echo "Available datasets:"
    for ds in "${!DATASET_TASKS[@]}"; do
        echo "  - $ds"
    done | sort
    echo "  - all (submit all datasets sequentially)"
    exit 1
fi

DATASET_ARG=$1

# Handle "all" case
if [ "$DATASET_ARG" == "all" ]; then
    echo "========================================="
    echo "Submitting ALL datasets SEQUENTIALLY"
    echo "========================================="
    echo ""
    echo "Total: 24 jobs (8 datasets × 3 seeds)"
    echo "Estimated time: ~24 hours total"
    echo ""
    echo "Chain: Baseline s0→s1→s2 → TL s0→s1→s2 → ATL2 s0→s1→s2 → ..."
    echo ""

    read -p "Submit all jobs? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    echo ""
    echo "Submitting jobs..."
    echo ""

    # Submit all datasets in order with chaining
    LAST_JOB_ID=""
    for dataset in Baseline TL ATL2 ATL3 ATL4 ATL5 ATL6 ATL7; do
        LAST_JOB_ID=$(submit_dataset "$dataset" "$LAST_JOB_ID")
    done

    echo ""
    echo "========================================="
    echo "✓ All 24 jobs submitted!"
    echo "========================================="
    echo ""
    echo "Monitor: squeue -u $USER"
    echo ""
    exit 0
fi

# Handle single dataset case
if [ ! ${DATASET_TASKS[$DATASET_ARG]+_} ]; then
    echo "Error: Unknown dataset '$DATASET_ARG'"
    echo ""
    echo "Available datasets:"
    for ds in "${!DATASET_TASKS[@]}"; do
        echo "  - $ds"
    done | sort
    exit 1
fi

echo "========================================="
echo "Training: ${DATASET_ARG}_1e4_m1e7_b210"
echo "========================================="
echo ""
echo "Estimated time: ~3 hours total (1 hour per seed)"
echo ""

read -p "Submit $DATASET_ARG training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo ""

submit_dataset "$DATASET_ARG" "" > /dev/null

echo ""
echo "========================================="
echo "✓ $DATASET_ARG training submitted!"
echo "========================================="
echo ""
echo "Monitor: squeue -u $USER"
echo ""
