#!/bin/bash
# Train ATL7_1e4_m1e7_b210 with all 3 seeds sequentially
# Seed 0 → Seed 1 → Seed 2

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

DATASET="ATL7_1e4_m1e7_b210"
TASKS=(22 23 24)  # seed 0, 1, 2

echo "========================================="
echo "Training: $DATASET"
echo "========================================="
echo ""
echo "Will train 3 seeds sequentially:"
echo "  Task 22: Seed 0 (saves checkpoints)"
echo "  Task 23: Seed 1"
echo "  Task 24: Seed 2"
echo ""
echo "Estimated time: ~3 hours total (1 hour per seed)"
echo ""

read -p "Submit $DATASET training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs with dependencies..."
echo ""

# Submit seed 0 (no dependency)
JOB_ID_0=$(sbatch --parsable --array=${TASKS[0]} train_job.sbatch)
echo "✓ Seed 0 submitted: Job ID $JOB_ID_0"

# Submit seed 1 (depends on seed 0)
JOB_ID_1=$(sbatch --parsable --dependency=afterok:$JOB_ID_0 --array=${TASKS[1]} train_job.sbatch)
echo "✓ Seed 1 submitted: Job ID $JOB_ID_1 (after $JOB_ID_0)"

# Submit seed 2 (depends on seed 1)
JOB_ID_2=$(sbatch --parsable --dependency=afterok:$JOB_ID_1 --array=${TASKS[2]} train_job.sbatch)
echo "✓ Seed 2 submitted: Job ID $JOB_ID_2 (after $JOB_ID_1)"

echo ""
echo "========================================="
echo "✓ $DATASET training submitted!"
echo "========================================="
echo ""
echo "Job IDs: $JOB_ID_0, $JOB_ID_1, $JOB_ID_2"
echo ""
echo "Monitor: squeue -u $USER"
echo ""
