"""Data generation script for Tournament GCD Option B annotations.

Option B shows the full Euclidean algorithm trace within each pairwise GCD computation
in the tournament tree, rather than just the final (g, u, v) result (Option A).

For each tournament node (round r, pair p), we record T quotient/Bézout steps,
then the final GCD and Bézout coefficients for that pair.

Generates a single dataset:
    Tournament_n{n}_ATL_B_T{T}_{ubound}_m{nsamples}_b{base}/

Usage (test run, n=4 only):
    python -m spm.data.tournament_optionb.generate_data --n_values 4

Usage (full run):
    SPM_DATA_DIR=/dcs/large/u5514611/data python -m spm.data.tournament_optionb.generate_data --n_values 4

Seed 67 is used for reproducibility (same as existing tournament data).
"""

import logging
import random
from argparse import ArgumentParser

import numpy as np

from spm.data.tournament_optionb.samplers import TournamentOptionBSampler
from spm.data.tournament_optionb.labels import TournamentOptionBLabels
from spm.data.samplers import DisjointRejector
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TensorRepr, TargetComponent

logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")


def set_seed(seed: int):
    logging.info(f"Setting seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


def save_samples(train_samples, val_samples, name: str, n: int):
    """Encode and save Option B samples."""
    train_encoded = EncodedSamples(train_samples, args.base)
    val_encoded = EncodedSamples(val_samples, args.base)

    # Fix: EncodedSamples defaults to Labels.INPUTS=[a,b,aq,bq] for input detection.
    # Manually override with tournament input labels.
    input_labels_list = TournamentOptionBLabels.input_labels(n)
    train_encoded.input_labels = input_labels_list
    train_encoded.target_labels = [k for k in train_encoded.all_labels if k not in input_labels_list]
    val_encoded.input_labels = input_labels_list
    val_encoded.target_labels = [k for k in val_encoded.all_labels if k not in input_labels_list]

    logging.info(f"Saving dataset: {name}")
    logging.info(f"  Target labels ({len(train_encoded.target_labels)}): {train_encoded.target_labels}")

    tr = TensorRepr.from_samples(train_encoded, val_encoded, name)
    tr.save_train()

    # ATL evaluation: all three variants
    val_cols = [TargetComponent.ANNOTATED_TRANSCRIPT, TargetComponent.TRANSCRIPT, TargetComponent.OUTPUT]
    tr.save_val(val_cols)


def generate_optionb(args, n: int):
    """Generate Option B ATL dataset for n inputs."""
    logging.info(f"Generating Option B ATL for n={n}, T={args.T}")
    sampler = TournamentOptionBSampler(n=n, ubound=args.ubound, T=args.T, base=args.base)

    logging.info("Generating validation samples")
    set_seed(args.seed)
    val_samples = sampler(args.n_val)

    logging.info("Generating training samples")
    set_seed(args.seed)
    train_samples = DisjointRejector(sampler, val_samples)(args.n_train)

    n_str = f"{len(train_samples):.0e}".replace("e+0", "e")
    ubound_str = f"{args.ubound:.0e}".replace("e+0", "e")
    name = f"Tournament_n{n}_ATL_B_T{args.T}_{ubound_str}_m{n_str}_b{args.base}"

    save_samples(train_samples, val_samples, name, n)


if __name__ == "__main__":
    parser = ArgumentParser(description="Generate tournament GCD Option B data")
    parser.add_argument("--n_train", type=int, default=1024 * 100 * 100, help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=1000, help="Number of validation samples")
    parser.add_argument("--ubound", type=int, default=10**4, help="Upper bound for input integers")
    parser.add_argument("--base", type=int, default=210, help="Base for integer representation")
    parser.add_argument("--T", type=int, default=6, help="Euclidean trace steps per pair (default 6)")
    parser.add_argument("--seed", type=int, default=67, help="Random seed")
    parser.add_argument(
        "--n_values",
        type=int,
        nargs="+",
        default=[4],
        help="List of n values to generate data for (default: 4)",
    )
    args = parser.parse_args()

    logging.info(f"Starting Option B data generation for n={args.n_values}")
    logging.info(f"Parameters: n_train={args.n_train}, n_val={args.n_val}, ubound={args.ubound}, T={args.T}, base={args.base}")

    for n in args.n_values:
        logging.info("=" * 60)
        logging.info(f"Processing n={n}")
        logging.info("=" * 60)
        generate_optionb(args, n)

    logging.info("Option B data generation complete!")
