# Quick Start Guide

⚠️ **CRITICAL: Submit ONE dataset at a time!** Running multiple datasets simultaneously causes symlink conflicts and crashes.

## Submit Training Jobs

```bash
# 1. Connect to SLURM head node
ssh kudu-taught

# 2. Navigate to training directory
cd /dcs/23/u5514611/cs310/self-proving-models/training

# 3. Submit ONE dataset at a time
bash submit_baseline.sh   # Trains Baseline with seeds 0→1→2

# 4. Wait ~99 hours for completion, then submit next
bash submit_tl.sh         # Then TL with seeds 0→1→2

# DO NOT run: bash submit_baseline.sh && bash submit_tl.sh
```

## Monitor Progress

```bash
# Check running/queued jobs
squeue -u u5514611

# Watch training logs (replace JOB_ID and TASK_ID)
tail -f logs/job_<JOB_ID>_task_<TASK_ID>.err

# Check job status
seff <JOB_ID>_<TASK_ID>
```

## What Gets Trained

**8 datasets** × **3 seeds** = **24 models**:
- Baseline, TL, ATL2, ATL3, ATL4, ATL5, ATL6, ATL7
- Seeds: 0, 1, 2 (trained sequentially per dataset)
- Seed 0 saves checkpoints every epoch

## Available Scripts

| Script | Datasets | Tasks |
|--------|----------|-------|
| `submit_baseline.sh` | Baseline | 1, 2, 3 |
| `submit_tl.sh` | TL | 4, 5, 6 |
| `submit_atl2.sh` | ATL2 | 7, 8, 9 |
| `submit_atl3.sh` | ATL3 | 10, 11, 12 |
| `submit_atl4.sh` | ATL4 | 13, 14, 15 |
| `submit_atl5.sh` | ATL5 | 16, 17, 18 |
| `submit_atl6.sh` | ATL6 | 19, 20, 21 |
| `submit_atl7.sh` | ATL7 | 22, 23, 24 |

Each script submits 3 jobs (one per seed) with dependencies so they run sequentially.

## Timeline

- **Per seed**: ~33 hours
- **Per dataset** (3 seeds): ~99 hours (~4 days)
- **All 8 datasets**: ~792 hours (~33 days) **running sequentially**

**You MUST run datasets sequentially** due to workspace symlink conflicts. Submit one dataset, wait for completion, then submit the next.

## Cancel Jobs

```bash
scancel <JOB_ID>              # Cancel all tasks in array
scancel <JOB_ID>_<TASK_ID>    # Cancel specific task
```

## Troubleshooting

**Job failed?** Check logs:
```bash
cat logs/job_<JOB_ID>_task_<TASK_ID>.err
cat logs/job_<JOB_ID>_task_<TASK_ID>.out
```

**Stale FUSE mount?** Already handled automatically in `train_single.sh`.

**Jobs slow?** If multiple jobs run on same GPU, they share resources. Cancel extras and run sequentially.
