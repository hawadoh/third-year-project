#!/bin/bash
# Submit training jobs for Part 1 datasets with full dependency chaining
# Usage: ./submit_training_part1.sh <dataset_name>
#   or: ./submit_training_part1.sh all  (to submit all Part 1 datasets sequentially)
#
# Part 1 datasets: TL, ATL2, ATL3, ATL4 (12 jobs total)
#
# Jobs run sequentially: seed 0 → seed 1 → seed 2 → next dataset seed 0 → ...

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

# Dataset to task ID mapping (from train_job.sbatch array configuration)
declare -A DATASET_TASKS=(
    ["TL"]="4 5 6"
    ["ATL2"]="7 8 9"
    ["ATL3"]="10 11 12"
    ["ATL4"]="13 14 15"
)

# Function to submit one dataset's training jobs with dependency on previous job
submit_dataset() {
    local dataset=$1
    local prev_job_id=$2  # Empty string if first dataset
    local full_name="${dataset}_1e4_m1e7_b210"

    # Get task IDs for this dataset
    local tasks=(${DATASET_TASKS[$dataset]})

    echo "----------------------------------------" >&2
    echo "Dataset: $full_name" >&2
    echo "  Task ${tasks[0]}: Seed 0 (saves checkpoints)" >&2
    echo "  Task ${tasks[1]}: Seed 1" >&2
    echo "  Task ${tasks[2]}: Seed 2" >&2

    # Submit seed 0
    if [ -z "$prev_job_id" ]; then
        # First dataset - no dependency
        JOB_ID_0=$(sbatch --parsable --array=${tasks[0]} train_job.sbatch)
        echo "  ✓ Seed 0: Job $JOB_ID_0" >&2
    else
        # Depends on previous dataset's last seed
        JOB_ID_0=$(sbatch --parsable --dependency=afterok:$prev_job_id --array=${tasks[0]} train_job.sbatch)
        echo "  ✓ Seed 0: Job $JOB_ID_0 (after $prev_job_id)" >&2
    fi

    # Submit seed 1 (depends on seed 0)
    JOB_ID_1=$(sbatch --parsable --dependency=afterok:$JOB_ID_0 --array=${tasks[1]} train_job.sbatch)
    echo "  ✓ Seed 1: Job $JOB_ID_1 (after $JOB_ID_0)" >&2

    # Submit seed 2 (depends on seed 1)
    JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --array=${tasks[2]} train_job.sbatch)
    echo "  ✓ Seed 2: Job $JOB_ID_2 (after $JOB_ID_1)" >&2

    # Return the last job ID for chaining
    echo "$JOB_ID_2"
}

# Parse arguments
AFTER_JOB=""
DATASET_ARG=""

while [ $# -gt 0 ]; do
    case "$1" in
        --after=*)
            AFTER_JOB="${1#*=}"
            shift
            ;;
        *)
            DATASET_ARG="$1"
            shift
            ;;
    esac
done

if [ -z "$DATASET_ARG" ]; then
    echo "Usage: $0 [--after=JOBID] <dataset_name|all>"
    echo ""
    echo "Options:"
    echo "  --after=JOBID    Wait for specified job to complete before starting"
    echo ""
    echo "Available datasets (Part 1):"
    for ds in "${!DATASET_TASKS[@]}"; do
        echo "  - $ds"
    done | sort
    echo "  - all (submit all Part 1 datasets sequentially)"
    exit 1
fi

# Handle "all" case
if [ "$DATASET_ARG" == "all" ]; then
    echo "========================================="
    echo "Submitting Part 1 datasets SEQUENTIALLY"
    echo "========================================="
    echo ""
    echo "Total: 12 jobs (4 datasets × 3 seeds)"
    echo "Estimated time: ~12 hours total"
    echo ""
    echo "Chain: TL s0→s1→s2 → ATL2 s0→s1→s2 → ATL3 s0→s1→s2 → ATL4 s0→s1→s2"
    echo ""

    read -p "Submit all Part 1 jobs? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi

    echo ""
    echo "Submitting jobs..."
    echo ""

    # Submit all Part 1 datasets in order with chaining
    LAST_JOB_ID="$AFTER_JOB"
    if [ -n "$AFTER_JOB" ]; then
        echo "Will wait for job $AFTER_JOB to complete before starting"
        echo ""
    fi
    for dataset in TL ATL2 ATL3 ATL4; do
        LAST_JOB_ID=$(submit_dataset "$dataset" "$LAST_JOB_ID")
    done

    echo ""
    echo "========================================="
    echo "✓ All 12 Part 1 jobs submitted!"
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
    echo "Available datasets (Part 1):"
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

if [ -n "$AFTER_JOB" ]; then
    echo "Will wait for job $AFTER_JOB to complete before starting"
    echo ""
fi

submit_dataset "$DATASET_ARG" "$AFTER_JOB" > /dev/null

echo ""
echo "========================================="
echo "✓ $DATASET_ARG training submitted!"
echo "========================================="
echo ""
echo "Monitor: squeue -u $USER"
echo ""
