# Tournament GCD Training (n=4)

Training script for tournament GCD models with n=4 inputs.

## Datasets

Three dataset types (all in `/dcs/large/u5514611/data/`):
- **Tournament_n4_Baseline_1e4_m1e7_b210** - Inputs → k only (GCD)
- **Tournament_n4_TL_1e4_m1e7_b210** - Inputs → k + Bézout coefficients
- **Tournament_n4_ATL_1e4_m1e7_b210** - Inputs → annotations → k + coefficients

Each dataset: 10.24M training samples, 1000 validation samples

## Training Configuration

**Job array:** 3 datasets × 3 seeds = 9 jobs total

**Hyperparameters** (from `train_single.sh`):
- Epochs: 10
- Batch size: 1024
- Learning rate: 0.0007
- Beta1: 0.733
- Model: n_embd=256, n_head=8, n_layer=8
- Gradient clip: 2.0
- LR decay: 10 epochs

**Resources per job:**
- 1 GPU (A10 on gecko, A6000 on falcon)
- 12 CPUs
- 32GB RAM
- Max runtime: 48 hours

## Usage

### Submit jobs

```bash
cd /dcs/23/u5514611/cs310/third-year-project/training

# Submit all 9 jobs (3 datasets × 3 seeds)
sbatch train_job_tournament_n4.sbatch
```

### Monitor jobs

```bash
# Check job status
squeue -u u5514611

# Check specific job array
squeue -u u5514611 -j <job_id>

# View live output (replace task number)
tail -f logs/tournament_n4_job_<job_id>_task_1.out

# Check errors
tail -f logs/tournament_n4_job_<job_id>_task_1.err
```

### Cancel jobs

```bash
# Cancel all jobs in array
scancel <job_id>

# Cancel specific task
scancel <job_id>_<task_number>
```

## Output Locations

**Models:** `/dcs/large/u5514611/models/<dataset>/seed_<seed>/`
- `ckpt_final.pt` - Final checkpoint (saved for seed 0 only)

**Logs:** `/dcs/23/u5514611/cs310/third-year-project/logs/`
- `<dataset>_seed_<seed>.log` - Training metrics (loss, accuracy, etc.)

**SLURM logs:** `/dcs/23/u5514611/cs310/third-year-project/training/logs/`
- `tournament_n4_job_<job_id>_task_<task_num>.out` - stdout
- `tournament_n4_job_<job_id>_task_<task_num>.err` - stderr

## Job Array Mapping

| Task ID | Dataset | Seed |
|---------|---------|------|
| 1 | Baseline | 0 |
| 2 | Baseline | 1 |
| 3 | Baseline | 2 |
| 4 | TL | 0 |
| 5 | TL | 1 |
| 6 | TL | 2 |
| 7 | ATL | 0 |
| 8 | ATL | 1 |
| 9 | ATL | 2 |