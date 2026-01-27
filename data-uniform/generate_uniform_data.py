"""
Generate uniform distribution data for comparison with log-uniform trained models.

ORIGINAL VERSION (before Term 2 Week 3):
  - Only generated eval data (1000 samples) under data-uniform/
  - Used hardcoded N_VAL=1000, no training data generation
  - No command-line arguments

EXTENDED VERSION (Term 2 Week 3):
  - Added --n_train argument to generate training data
  - Added --datasets argument to select specific datasets
  - Training data saved to data/ directory with "_uniform" suffix
  - Backward compatible: running without arguments still generates eval-only

Two modes:
  1. Eval-only (default): Generates 1000 eval pairs under data-uniform/
  2. Training: Generates full training + eval data under data/

Usage:
  # Eval-only (for testing log-uniform models on uniform data)
  python data-uniform/generate_uniform_data.py

  # Training data (for training new models on uniform data)
  python data-uniform/generate_uniform_data.py --n_train 10240000

  # Training data for specific datasets only
  python data-uniform/generate_uniform_data.py --n_train 10240000 --datasets Baseline TL ATL7
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import spm
sys.path.insert(0, str(Path(__file__).parent.parent))

from spm.data.samplers import (
    AnnotatedTranscriptSampler,
    DisjointRejector,
    TranscriptSampler,
    UniformGCDSampler,
    UpperBoundRejector,
)
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TargetComponent, TensorRepr

# Directories
DATA_UNIFORM_DIR = Path(__file__).parent
DATA_DIR = Path(__file__).parent.parent / "data"


def set_seed(seed):
    print(f"Setting seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


def get_name(prefix, args, uniform_suffix=True):
    """Generate dataset name following original convention."""
    n_samples_str = f"{args.n_train:.0e}".replace("e+0", "e")
    ubound_str = f"{args.ubound:.0e}".replace("e+0", "e")
    if uniform_suffix:
        return f"{prefix}_uniform_{ubound_str}_m{n_samples_str}_b{args.base}"
    else:
        return f"{prefix}_{ubound_str}_m{n_samples_str}_b{args.base}"


def save_eval_only(val_samples, name, args, val_cols):
    """Save only eval.npz (no training data) under data-uniform/."""
    val_encoded = EncodedSamples(val_samples, args.base)

    out_dir = DATA_UNIFORM_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = TensorRepr.from_samples(val_encoded, val_encoded, name)

    inputs, masks, masked = tr.make_val(val_cols)
    val_data = {}
    for length in inputs:
        val_data[f"input_{length}"] = inputs[length]
        for col in masks[length].keys():
            val_data[f"masks_{length}_{col}"] = masks[length][col]
            val_data[f"masked_{length}_{col}"] = masked[length][col]

    eval_path = out_dir / "eval.npz"
    np.savez(eval_path, **val_data)
    print(f"Saved {eval_path} ({len(val_samples)} samples)")

    tr.m.root_path = out_dir
    tr.m.meta_path = out_dir / "meta.json"
    tr.m.save()


def save_train_and_eval(train_samples, val_samples, name, args, val_cols):
    """Save full training data (x.bin, y.bin) and eval.npz under data/."""
    train_encoded = EncodedSamples(train_samples, args.base)
    val_encoded = EncodedSamples(val_samples, args.base)

    tr = TensorRepr.from_samples(train_encoded, val_encoded, name)
    tr.save_train()
    tr.save_val(val_cols)

    print(f"Saved {DATA_DIR / name} ({len(train_samples)} train, {len(val_samples)} val)")


def generate_baseline(args, u_sampler, training_mode):
    name_prefix = "Baseline"
    print(f"\n=== Generating {name_prefix} (uniform) ===")

    if training_mode:
        name = get_name(name_prefix, args)
        set_seed(args.seed)
        val_samples = u_sampler(args.n_val)
        set_seed(args.seed)
        train_samples = DisjointRejector(u_sampler, val_samples)(args.n_train)
        save_train_and_eval(train_samples, val_samples, name, args,
                           [TargetComponent.ANNOTATED_TRANSCRIPT])
    else:
        # Eval-only uses original naming convention for compatibility
        name = f"{name_prefix}_{args.ubound:.0e}_m1e7_b{args.base}".replace("e+0", "e")
        set_seed(args.seed)
        val_samples = u_sampler(args.n_val)
        save_eval_only(val_samples, name, args, [TargetComponent.ANNOTATED_TRANSCRIPT])


def generate_transcripts(args, u_sampler, training_mode):
    name_prefix = "TL"
    print(f"\n=== Generating {name_prefix} (uniform) ===")

    sampler = UpperBoundRejector(TranscriptSampler(u_sampler), args.rejector_ubound)
    val_cols = [
        TargetComponent.ANNOTATED_TRANSCRIPT,
        TargetComponent.TRANSCRIPT,
        TargetComponent.OUTPUT,
    ]

    if training_mode:
        name = get_name(name_prefix, args)
        set_seed(args.seed)
        val_samples = sampler(args.n_val)
        set_seed(args.seed)
        train_samples = DisjointRejector(sampler, val_samples)(args.n_train)
        save_train_and_eval(train_samples, val_samples, name, args, val_cols)
    else:
        name = f"{name_prefix}_{args.ubound:.0e}_m1e7_b{args.base}".replace("e+0", "e")
        set_seed(args.seed)
        val_samples = sampler(args.n_val)
        save_eval_only(val_samples, name, args, val_cols)


def generate_annotated_transcripts(args, u_sampler, training_mode, target_depths=None):
    """Generate ATL datasets.

    Args:
        target_depths: List of annotation depths to generate (e.g., [7] for ATL7 only).
                      If None, generates all depths in annot_len_range.
    """
    start_len, stop_len = args.annot_len_range
    annot_len = stop_len - 1

    sampler = UpperBoundRejector(
        AnnotatedTranscriptSampler(u_sampler, annot_len=annot_len),
        args.rejector_ubound
    )

    val_cols = [
        TargetComponent.ANNOTATED_TRANSCRIPT,
        TargetComponent.TRANSCRIPT,
        TargetComponent.OUTPUT,
    ]

    set_seed(args.seed)
    val_samples = sampler(args.n_val)

    if training_mode:
        set_seed(args.seed)
        train_samples = DisjointRejector(sampler, val_samples)(args.n_train)

    while annot_len >= start_len - 1:
        name_prefix = f"ATL{annot_len}"

        # Skip if not in target_depths
        if target_depths is not None and annot_len not in target_depths:
            print(f"\n=== Skipping {name_prefix} (not in --datasets) ===")
            if training_mode:
                train_samples.shorten_annot()
            val_samples.shorten_annot()
            annot_len -= 1
            continue

        print(f"\n=== Generating {name_prefix} (uniform) ===")

        if training_mode:
            name = get_name(name_prefix, args)
            save_train_and_eval(train_samples, val_samples, name, args, val_cols)
            train_samples.shorten_annot()
        else:
            name = f"{name_prefix}_{args.ubound:.0e}_m1e7_b{args.base}".replace("e+0", "e")
            save_eval_only(val_samples, name, args, val_cols)

        val_samples.shorten_annot()
        annot_len -= 1


def parse_datasets(datasets_arg):
    """Parse dataset names into (generate_baseline, generate_tl, atl_depths)."""
    if datasets_arg is None:
        return True, True, None  # Generate all

    generate_baseline = False
    generate_tl = False
    atl_depths = []

    for d in datasets_arg:
        d_upper = d.upper()
        if d_upper == "BASELINE":
            generate_baseline = True
        elif d_upper == "TL":
            generate_tl = True
        elif d_upper.startswith("ATL"):
            depth = int(d_upper[3:])
            atl_depths.append(depth)

    return generate_baseline, generate_tl, atl_depths if atl_depths else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n_train", type=int, default=0,
                        help="Number of training samples. If 0, only generates eval data.")
    parser.add_argument("--n_val", type=int, default=1000,
                        help="Number of validation samples")
    parser.add_argument("--ubound", type=int, default=10**4,
                        help="Upper bound on integers")
    parser.add_argument("--rejector_ubound", type=int, default=210**3,
                        help="Upper bound on any integer in transcript")
    parser.add_argument("--base", type=int, default=210,
                        help="Base for integer representation")
    parser.add_argument("--annot_len_range", type=int, nargs=2, default=[3, 8],
                        help="Range of annotation lengths [start, stop)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to generate (e.g., Baseline TL ATL7). "
                             "If not specified, generates all.")

    args = parser.parse_args()

    training_mode = args.n_train > 0
    gen_baseline, gen_tl, atl_depths = parse_datasets(args.datasets)

    print("=" * 60)
    if training_mode:
        print(f"Generating UNIFORM distribution TRAINING data")
        print(f"  n_train={args.n_train}, n_val={args.n_val}")
        print(f"  Output: {DATA_DIR}")
    else:
        print(f"Generating UNIFORM distribution EVAL data only")
        print(f"  n_val={args.n_val}")
        print(f"  Output: {DATA_UNIFORM_DIR}")
    print(f"  ubound={args.ubound}, base={args.base}, seed={args.seed}")
    if args.datasets:
        print(f"  Datasets: {args.datasets}")
    print("=" * 60)

    u_sampler = UniformGCDSampler(args.ubound)

    if gen_baseline:
        generate_baseline(args, u_sampler, training_mode)

    if gen_tl:
        generate_transcripts(args, u_sampler, training_mode)

    if atl_depths is None or atl_depths:  # None means all, non-empty list means specific
        generate_annotated_transcripts(args, u_sampler, training_mode, atl_depths)

    print("\n" + "=" * 60)
    print("Done! Generated data for:")
    output_dir = DATA_DIR if training_mode else DATA_UNIFORM_DIR
    for d in sorted(output_dir.iterdir()):
        if d.is_dir():
            has_train = (d / "x.bin").exists()
            has_eval = (d / "eval.npz").exists()
            if "uniform" in d.name.lower() or not training_mode:
                if has_train or has_eval:
                    status = "train+eval" if has_train else "eval only"
                    print(f"  - {d.name} ({status})")
    print("=" * 60)


if __name__ == "__main__":
    main()
