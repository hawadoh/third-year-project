# Self-Proving Models: Reproduction and Extension

**CS310 Third Year Project** - University of Warwick
Howard Cheung (5514611) | Supervised by Dr. Matthias C. Caro

## Overview

Self-proving models are neural networks that output verifiable proofs alongside their answers, transforming average-case correctness guarantees into per-input certificates. This project reproduces and extends the self-proving framework from [Amit et al. (2024)](https://arxiv.org/abs/2405.15722).

**Term 1 accomplishments:**
- Reproduced Figure 2 from the original paper with improved training stability (smaller error bars across seeds)
- Discovered a distribution mismatch: models trained on log-uniform inputs fail to generalise to uniformly sampled inputs

**Planned extensions (Term 2):**
- Tournament GCD: extend the framework to compute GCD of n > 2 inputs using a binary tree structure
- Investigate the distribution generalisation question
- Explore extensions to problems in NP

## Setup

1. Create a conda environment:
```bash
conda create -n spm python=3.12
conda activate spm
```

2. Install the package:
```bash
pip install -e .
```

## Data Generation

### Log-uniform distribution (original)

Generate training data with log-uniform sampling (as in the original paper):
```bash
python spm/data/generate_data.py
```

This populates `data/` with Transcripts and Annotated Transcripts. Dataset naming convention:
```
TL_{UPPER_BOUND}_m{NUM_SAMPLES}_b{BASE}      # Transcript Learning
ATL{DEPTH}_{UPPER_BOUND}_m{NUM_SAMPLES}_b{BASE}  # Annotated Transcript Learning
```

### Uniform distribution (new)

Generate evaluation data with uniform sampling for distribution comparison:
```bash
python data-uniform/generate_uniform_data.py
```

This creates `data-uniform/` with evaluation sets for all dataset types (Baseline, TL, ATL2-7).

## Training

### Local training
```bash
python spm/train.py --data TL_1e4_m1e7_b210 --device cuda
```

Useful arguments:
- `--device DEVICE`: `cpu`, `cuda`, or `mps` (Apple Silicon)
- `--epochs EPOCHS`: Number of training epochs
- `--seed SEED`: Random seed for reproducibility
- `--save_iters N [N ...]`: Save checkpoints at these iterations
- `--wandb`: Enable Weights & Biases logging

### DCS Batch Compute System

For training on Warwick's DCS cluster with SLURM:
```bash
cd training/
./submit_training.sh
```

See `training/README.md` and `dcs-docs/` for detailed instructions on batch job submission.

#### DCS Large Storage Setup

If you have a large storage allocation (e.g., `/dcs/large/u5514611/`), set up as follows:

```bash
# Create directory structure on large storage
mkdir -p /dcs/large/u5514611/data
mkdir -p /dcs/large/u5514611/models

# Copy training data to large storage
cp data.zip /dcs/large/u5514611/data/
```

The training scripts (`training/train_single.sh`) automatically:
- Read `data.zip` from `/dcs/large/u5514611/data/`
- Save models to `/dcs/large/u5514611/models/`
- Extract to `$TMPDIR` for fast I/O during training (35x faster than fuse-zip)
- Keep training logs in the repository directory

This setup avoids home directory quota issues while maintaining fast training performance.

### Reproduce Figure 2 (annotation length ablation)
```bash
./runs/annot_len.sh           # Train all annotation lengths
python figs/annotation.py     # Generate figure
```

### Reproduce Figure 3 (base of representation)
```bash
python spm/train_diff_bases.py --num_unique_primes NUM --seed SEED
python figs/diff_bases.py     # Generate figure
```

## Inference & Evaluation

### Batch inference
```bash
python tests/inference/inference.py
```

### Interactive testing
```bash
python tests/inference/interactive_inference.py
```

## Project Structure

```
spm/                    # Main package
  train.py              # Training entry point
  train_diff_bases.py   # Training with different bases
  utils.py              # Utilities (extended Euclidean algorithm)
  data/                 # Data generation and representation
    generate_data.py    # Dataset generation
    samplers.py         # GCD samplers (log-uniform, uniform)
    samples.py          # Sample, Transcript, AnnotatedTranscript classes
    str_repr.py         # String encoding (base conversion)
    tensor_repr.py      # Tensor representation for training
  gpt/                  # GPT model (adapted from nanoGPT)
    model.py            # Model architecture
    trainer.py          # Training loop
    config.py           # Configuration

data-uniform/           # Uniform distribution eval data
  generate_uniform_data.py

training/               # DCS batch compute scripts
  submit_*.sh           # SLURM submission scripts
  train_job.sbatch      # SLURM job template

tests/                  # Tests and inference
  inference/            # Inference scripts

figs/                   # Figure generation scripts
docs/                   # Documentation and reports
dcs-docs/               # DCS-specific guides
```

## Acknowledgements

This project is based on the [official repository](https://github.com/orrp/self-proving-models) by Amit et al. The GPT implementation is adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).

The authors acknowledge the use of the Batch Compute System in the Department of Computer Science at the University of Warwick, and associated support services, in the completion of this work.

## Citation

If you use this work, please cite the original paper:
```bibtex
@article{AGPR2024,
  author    = {Noga Amit and Shafi Goldwasser and Orr Paradise and Guy N. Rothblum},
  title     = {Models That Prove Their Own Correctness},
  journal   = {CoRR},
  volume    = {abs/2405.15722},
  year      = {2024},
  url       = {https://doi.org/10.48550/arXiv.2405.15722},
  doi       = {10.48550/ARXIV.2405.15722},
  eprinttype = {arXiv},
  eprint    = {2405.15722},
}
```
