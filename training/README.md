# Training on DCS Batch Compute System

Scripts to train self-proving models using SLURM batch compute with fuse-zip for efficient data handling.

## Overview

- **Mounts** `data.zip` using fuse-zip in `$TMPDIR` (avoids quota limits)
- **Trains** 8 datasets × 3 seeds = 24 models
- **Uses** SLURM job dependencies for sequential execution per dataset
- **Runs** on GPU nodes (gecko/falcon partitions)

## Files

- `train_job.sbatch` - SLURM job array script
- `train_single.sh` - Mounts data and trains a single model
- `submit_*.sh` - 8 dataset-specific submission scripts
- `check_status.sh` - Monitor job progress
- `logs/` - Output and error logs

## Datasets

Each trained with 3 seeds (0, 1, 2):

1. **Baseline_1e4_m1e7_b210** - Table 1
2. **TL_1e4_m1e7_b210** - Table 1, Figure 2, 4
3. **ATL2_1e4_m1e7_b210** - Figure 2
4. **ATL3_1e4_m1e7_b210** - Figure 2, 4
5. **ATL4_1e4_m1e7_b210** - Figure 4
6. **ATL5_1e4_m1e7_b210** - Figure 2, 4
7. **ATL6_1e4_m1e7_b210** - Figure 4
8. **ATL7_1e4_m1e7_b210** - Table 1, Figure 2, 4

## Hyperparameters

```
Epochs: 10
Batch size: 1024
Learning rate: 0.0007
LR decay: 10%
Beta1: 0.733
Gradient clipping: 2.0
Model: n_embd=256, n_head=8, n_layer=8
Eval interval: 2000 iterations
Log interval: 100 iterations
Checkpoints: Every epoch for seed 0
```

## Usage

### Submit Jobs

⚠️ **CRITICAL: Run ONE dataset at a time!** Multiple jobs sharing the workspace data symlink will crash each other.

```bash
cd /dcs/23/u5514611/cs310/self-proving-models/training

# Submit ONE dataset
bash submit_baseline.sh

# Wait ~99 hours for all 3 seeds to complete, then submit next
bash submit_tl.sh

# Repeat for remaining datasets...
```

**DO NOT submit multiple datasets simultaneously!** Each dataset's 3 seeds run sequentially (seed 0→1→2) via job dependencies.

### Monitor Jobs

```bash
# View queue
squeue -u u5514611

# Check specific job
bash check_status.sh <JOB_ID>

# Watch logs live
tail -f logs/job_<JOB_ID>_task_<TASK_ID>.err
```

### Cancel Jobs

```bash
scancel <JOB_ID>              # Cancel all tasks
scancel <JOB_ID>_<TASK_ID>    # Cancel specific task
```

## Task ID Mapping

| Tasks | Dataset | Seeds |
|-------|---------|-------|
| 1-3 | Baseline | 0, 1, 2 |
| 4-6 | TL | 0, 1, 2 |
| 7-9 | ATL2 | 0, 1, 2 |
| 10-12 | ATL3 | 0, 1, 2 |
| 13-15 | ATL4 | 0, 1, 2 |
| 16-18 | ATL5 | 0, 1, 2 |
| 19-21 | ATL6 | 0, 1, 2 |
| 22-24 | ATL7 | 0, 1, 2 |

## Resource Allocation

Per job:
- **Partition**: gecko, falcon (prefers gecko A10 GPUs)
- **CPUs**: 12 threads
- **GPU**: 1 GPU
- **RAM**: 32GB
- **Time**: 48 hours max

## How It Works

1. **Data mounting**: fuse-zip mounts `data.zip` to unique `$TMPDIR` location
2. **Symlink**: Creates workspace `data/` → mounted data
3. **Training**: Runs `train.py` with hyperparameters from `annot_len.sh`
4. **Cleanup**: Unmounts and removes symlink on exit (via trap)

## fuse-zip Benefits

```bash
# Mount (no extraction, no disk quota used)
fuse-zip -r data.zip $TMPDIR/spm_data_$$

# Unmount when done
fusermount3 -u $TMPDIR/spm_data_$$
```

- ✅ No disk quota usage (data stays in ZIP)
- ✅ Fast (~100× faster than extraction)
- ✅ Read-only protection
- ✅ Unique mount per job (no conflicts)

## Expected Timeline

- **Per seed**: ~33 hours
- **Per dataset** (3 seeds): ~99 hours (~4 days)
- **All 8 datasets**: ~792 hours (~33 days) **MUST run sequentially**

## Troubleshooting

**Job fails immediately?**
```bash
cat logs/job_<JOB_ID>_task_<TASK_ID>.err
```

**Mount issues?**
- Handled automatically by cleanup code in `train_single.sh`
- Uses `ln -sfn` to force symlink replacement

**Multiple datasets running simultaneously?**
- ❌ **FATAL:** Causes symlink conflicts → `ConnectionAbortedError` crashes
- Cancel all jobs: `scancel -u u5514611`
- **ALWAYS submit only ONE dataset at a time**
- Wait for all 3 seeds to complete before submitting next dataset

**CUDA out of memory?**
- Should not happen with current settings (batch_size=1024, 32GB RAM, 24GB GPU)

## After Training

1. **Check results**: Model checkpoints in `ckpts/` (seed 0 only)
2. **Generate figures**: `python figs/annotation.py`
3. **Analyze logs**: Training curves in `logs/`

## Notes

- Seed 0 saves checkpoints every 10,000 iterations (~every epoch)
- Seeds 1-2 don't save checkpoints (for disk space)
- wandb logging disabled (batch job compatibility)
- Logs preserved in `training/logs/` for debugging
