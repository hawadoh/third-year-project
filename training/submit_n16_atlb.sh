#!/bin/bash
# Submit 4 chained jobs for n=16 ATL-B training (25k iters each)
# Each job depends on the previous one completing successfully.

cd /dcs/23/u5514611/cs310/third-year-project

JOB1=$(sbatch --parsable --array=1 training/train_job_n16_atlb.sbatch)
echo "Batch 1 (0-25k):   job $JOB1"

JOB2=$(sbatch --parsable --array=2 --dependency=afterok:$JOB1 training/train_job_n16_atlb.sbatch)
echo "Batch 2 (25k-50k):  job $JOB2 (depends on $JOB1)"

JOB3=$(sbatch --parsable --array=3 --dependency=afterok:$JOB2 training/train_job_n16_atlb.sbatch)
echo "Batch 3 (50k-75k):  job $JOB3 (depends on $JOB2)"

JOB4=$(sbatch --parsable --array=4 --dependency=afterok:$JOB3 training/train_job_n16_atlb.sbatch)
echo "Batch 4 (75k-100k): job $JOB4 (depends on $JOB3)"

echo ""
echo "All 4 jobs submitted. Use 'squeue -u \$USER' to monitor."
