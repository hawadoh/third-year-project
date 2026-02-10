"""Tournament GCD data generation script for n=4, 8, 16 inputs.

This script generates training and validation datasets for tournament GCD with variable n.
It produces three types of datasets (consistent with n=2 naming: Baseline, TL, ATL):

- Baseline: inputs → k only (no Bézout coefficients)
  → Analogous to n=2 Baseline

- TL (Transcript Learning): inputs → k + c1,...,cn (Bézout coefficients)
  → Analogous to n=2 TL

- ATL (Annotated Transcript Learning): inputs → [annotations] → k + c1,...,cn
  → Analogous to n=2 ATL (shows intermediate tournament rounds)

Usage:
    # Generate small test dataset for n=4
    python -m spm.data.tournament.generate_tournament_data --n_train 100 --n_val 10 --n_values 4

    # Generate full datasets for n=4, 8, 16 to /large folder
    SPM_DATA_DIR=/dcs/large/u5514611/data python -m spm.data.tournament.generate_tournament_data

Output structure (per n):
    Tournament_n{n}_Baseline_{ubound}_m{nsamples}_b{base}/
    Tournament_n{n}_TL_{ubound}_m{nsamples}_b{base}/
    Tournament_n{n}_ATL_{ubound}_m{nsamples}_b{base}/
    ├── train/
    │   ├── x.bin
    │   └── y.bin
    ├── eval/
    │   └── eval.npz
    └── meta.json

Compatibility notes:
- EncodedSamples hardcodes Labels.INPUTS for n=2, so we manually patch input_labels
- DisjointRejector expects .a and .b attributes, added as properties in TournamentSamples

Seed 67 is used for all tournament data generation (sixseven).
"""

import logging
import random
from argparse import ArgumentParser

import numpy as np

from spm.data.tournament.samplers import (
    LogUniformTournamentGCDSampler,
    TournamentTranscriptSampler,
    TournamentAnnotatedTranscriptSampler,
)
from spm.data.tournament.labels import TournamentLabels
from spm.data.samplers import DisjointRejector
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TensorRepr, TargetComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_seed(seed):
    """Set random seeds for reproducibility."""
    logging.info(f"Setting seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


def save_samples(train_samples, val_samples, name_prefix, args, n, dataset_type='ATL'):
    """Encode and save tournament samples with label compatibility fixes.

    Args:
        train_samples: TournamentSamples, TournamentTranscript, or TournamentAnnotatedTranscript
        val_samples: Same type as train_samples
        name_prefix: Prefix for dataset name (e.g., "Tournament_n4_TL")
        args: Command-line arguments (base, ubound, etc.)
        n: Number of inputs (4, 8, or 16)
        dataset_type: 'Baseline', 'TL', or 'ATL' - determines which evaluation variants to save
    """
    # Create EncodedSamples
    train_encoded = EncodedSamples(train_samples, args.base)
    val_encoded = EncodedSamples(val_samples, args.base)

    # FIX: Manually set input_labels since EncodedSamples hardcodes Labels.INPUTS
    # For tournament, input labels are [x1, x2, ..., xn]
    input_labels_list = TournamentLabels.input_labels(n)
    train_encoded.input_labels = input_labels_list
    train_encoded.target_labels = [k for k in train_encoded.all_labels if k not in input_labels_list]
    val_encoded.input_labels = input_labels_list
    val_encoded.target_labels = [k for k in val_encoded.all_labels if k not in input_labels_list]

    # Create name and save
    n_samples_str = f"{len(train_samples):.0e}".replace("e+0", "e")
    ubound_str = f"{args.ubound:.0e}".replace("e+0", "e")
    name = f"{name_prefix}_{ubound_str}_m{n_samples_str}_b{args.base}"

    logging.info(f"Saving dataset: {name}")
    tr = TensorRepr.from_samples(train_encoded, val_encoded, name)
    tr.save_train()

    # Save validation variants based on dataset type
    if dataset_type == 'Baseline':
        # Baseline: only output (just k)
        val_cols = [TargetComponent.OUTPUT]
    elif dataset_type == 'TL':
        # TL: transcript (k + coeffs) and output (just k)
        val_cols = [TargetComponent.TRANSCRIPT, TargetComponent.OUTPUT]
    else:  # ATL
        # ATL: annotated (everything), transcript (k + coeffs), output (just k)
        val_cols = [TargetComponent.ANNOTATED_TRANSCRIPT, TargetComponent.TRANSCRIPT, TargetComponent.OUTPUT]

    tr.save_val(val_cols)


def generate_baseline(args, n):
    """Generate Baseline data: inputs → k only (no Bézout coefficients).

    For n inputs (x1, ..., xn), generates:
    - Inputs: x1, ..., xn
    - Outputs: k (GCD only)

    Consistent with n=2 Baseline.

    Args:
        args: Command-line arguments
        n: Number of inputs (4, 8, or 16)
    """
    logging.info(f"Generating Baseline for n={n}")
    sampler = LogUniformTournamentGCDSampler(n, args.ubound)

    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)

    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(sampler, val_samples)(args.n_train)

    save_samples(train_samples, val_samples, f"Tournament_n{n}_Baseline", args, n, dataset_type='Baseline')


def generate_tournament_transcripts(args, n):
    """Generate TL (Transcript Learning) data: inputs → k + Bézout coefficients.

    For n inputs (x1, ..., xn), generates:
    - Inputs: x1, ..., xn
    - Outputs: k (GCD), c1, ..., cn (Bézout coefficients where sum(ci * xi) = k)

    Consistent with n=2 TL (Transcript Learning).

    Args:
        args: Command-line arguments
        n: Number of inputs (4, 8, or 16)
    """
    logging.info(f"Generating TL (Transcript Learning) for n={n}")
    sampler = TournamentTranscriptSampler(LogUniformTournamentGCDSampler(n, args.ubound))

    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)

    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(sampler, val_samples)(args.n_train)

    save_samples(train_samples, val_samples, f"Tournament_n{n}_TL", args, n, dataset_type='TL')


def generate_tournament_annotated(args, n):
    """Generate ATL (Annotated Transcript Learning) data with Option A annotations.

    For n inputs, generates:
    - Inputs: x1, ..., xn
    - Annotations: Intermediate (g, u, v) for each tournament round
    - Outputs: k (final GCD), c1, ..., cn (final Bézout coefficients)

    Consistent with n=2 ATL (Annotated Transcript Learning).

    Args:
        args: Command-line arguments
        n: Number of inputs (4, 8, or 16)
    """
    logging.info(f"Generating ATL (Annotated Transcript Learning) for n={n}")
    sampler = TournamentAnnotatedTranscriptSampler(LogUniformTournamentGCDSampler(n, args.ubound))

    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)

    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(sampler, val_samples)(args.n_train)

    save_samples(train_samples, val_samples, f"Tournament_n{n}_ATL", args, n, dataset_type='ATL')


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate tournament GCD training data for n=4, 8, 16")
    parser.add_argument("--n_train", type=int, default=1024 * 100 * 100, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--ubound", type=int, default=10**4, help="Upper bound for input integers")
    parser.add_argument("--base", type=int, default=210, help="Base for integer representation")
    parser.add_argument("--seed", type=int, default=67, help="Random seed (default: 67 for tournament)")
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[4, 8, 16],
        help="List of n values to generate data for (default: 4 8 16)",
    )
    args = parser.parse_args()

    logging.info(f"Starting tournament data generation for n={args.n_values}")
    logging.info(f"Parameters: n_train={args.n_train}, n_val={args.n_val}, ubound={args.ubound}, base={args.base}")

    for n in args.n_values:
        logging.info(f"=" * 60)
        logging.info(f"Processing n={n}")
        logging.info(f"=" * 60)

        generate_baseline(args, n)
        generate_tournament_transcripts(args, n)
        generate_tournament_annotated(args, n)

    logging.info("All datasets generated successfully!")
