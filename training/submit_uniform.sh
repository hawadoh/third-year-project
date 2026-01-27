#!/bin/bash
# Submit uniform distribution training jobs for distribution investigation
# Trains: Baseline_uniform, TL_uniform, ATL7_uniform (3 datasets × 3 seeds = 9 jobs)
#
# Prerequisites:
# 1. Generate uniform training data:
#    python data-uniform/generate_uniform_data.py --n_train 10240000 --datasets Baseline TL ATL7
#
# 2. Either:
#    a. Run data generation on DCS directly, OR
#    b. SCP the generated data/ directories to DCS

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p logs

DATASETS=("Baseline_uniform_1e4_m1e7_b210" "TL_uniform_1e4_m1e7_b210" "ATL7_uniform_1e4_m1e7_b210")

echo "========================================="
echo "UNIFORM DISTRIBUTION TRAINING"
echo "========================================="
echo ""
echo "Datasets to train:"
for d in "${DATASETS[@]}"; do
    echo "  - $d"
done
echo ""
echo "Total jobs: 9 (3 datasets × 3 seeds)"
echo "Estimated time: ~12 hours total"
echo "  - Baseline: ~1 hour/seed"
echo "  - TL: ~1.5 hours/seed"
echo "  - ATL7: ~4 hours/seed"
echo ""

# Check if data exists
DATA_DIR="../data"
MISSING=()
for d in "${DATASETS[@]}"; do
    if [ ! -d "$DATA_DIR/$d" ]; then
        MISSING+=("$d")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "WARNING: Missing datasets:"
    for d in "${MISSING[@]}"; do
        echo "  - $d"
    done
    echo ""
    echo "Generate with:"
    echo "  python data-uniform/generate_uniform_data.py --n_train 10240000 --datasets Baseline TL ATL7"
    echo ""
fi

read -p "Submit uniform training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "Submitting jobs..."
echo ""

# Submit all 9 jobs (they can run in parallel since each uses different dataset)
# Jobs for same dataset run sequentially (seed 0 -> 1 -> 2) via array dependencies

# Baseline (tasks 1-3)
JOB_ID_BASELINE=$(sbatch --parsable --array=1-3 train_job_uniform.sbatch)
echo "✓ Baseline_uniform submitted: Job ID $JOB_ID_BASELINE (tasks 1-3)"

# TL (tasks 4-6) - can start immediately, different dataset
JOB_ID_TL=$(sbatch --parsable --array=4-6 train_job_uniform.sbatch)
echo "✓ TL_uniform submitted: Job ID $JOB_ID_TL (tasks 4-6)"

# ATL7 (tasks 7-9) - can start immediately, different dataset
JOB_ID_ATL7=$(sbatch --parsable --array=7-9 train_job_uniform.sbatch)
echo "✓ ATL7_uniform submitted: Job ID $JOB_ID_ATL7 (tasks 7-9)"

echo ""
echo "========================================="
echo "✓ Uniform training submitted!"
echo "========================================="
echo ""
echo "Job IDs:"
echo "  Baseline: $JOB_ID_BASELINE"
echo "  TL:       $JOB_ID_TL"
echo "  ATL7:     $JOB_ID_ATL7"
echo ""
echo "Monitor: squeue -u $USER"
echo "Logs:    tail -f logs/uniform_job_*.out"
echo ""
