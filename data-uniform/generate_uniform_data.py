"""
Generate uniform distribution eval data for comparison with log-uniform trained models.
Generates 1000 eval pairs for each dataset type (Baseline, TL, ATL2-7).
"""
import random
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path so we can import spm
sys.path.insert(0, str(Path(__file__).parent.parent))

from spm.data.samplers import (
    AnnotatedTranscriptSampler,
    TranscriptSampler,
    UniformGCDSampler,
    UpperBoundRejector,
)
from spm.data.str_repr import EncodedSamples
from spm.data.tensor_repr import TargetComponent, TensorRepr

# Match original generate_data.py defaults
N_VAL = 1000
UBOUND = 10**4
REJECTOR_UBOUND = 210**3
BASE = 210
ANNOT_LEN_RANGE = [3, 8]
SEED = 42

DATA_UNIFORM_DIR = Path(__file__).parent


def set_seed(seed):
    print(f"Setting seed to {seed}")
    np.random.seed(seed)
    random.seed(seed)


def save_eval_only(val_samples, name_prefix, val_cols):
    """Save only eval.npz (no training data) under data-uniform/."""
    val_encoded = EncodedSamples(val_samples, BASE)

    # Use same naming convention as original data
    name = f"{UBOUND:.0e}_m1e7_b{BASE}".replace("e+0", "e")
    name = f"{name_prefix}_{name}" if name_prefix else name

    # Create output directory under data-uniform/
    out_dir = DATA_UNIFORM_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create a minimal TensorRepr to generate eval data
    # We need a dummy train_samples (can be same as val for this purpose)
    tr = TensorRepr.from_samples(val_encoded, val_encoded, name)

    # Generate and save eval.npz manually to our custom directory
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

    # Also save meta.json for compatibility
    tr.m.root_path = out_dir
    tr.m.meta_path = out_dir / "meta.json"
    tr.m.save()


def generate_baseline(u_sampler):
    print("\n=== Generating Baseline (uniform) ===")
    set_seed(SEED)
    val_samples = u_sampler(N_VAL)
    save_eval_only(val_samples, "Baseline", [TargetComponent.ANNOTATED_TRANSCRIPT])


def generate_transcripts(u_sampler):
    print("\n=== Generating TL (uniform) ===")
    sampler = UpperBoundRejector(TranscriptSampler(u_sampler), REJECTOR_UBOUND)
    set_seed(SEED)
    val_samples = sampler(N_VAL)
    save_eval_only(val_samples, "TL", [
        TargetComponent.ANNOTATED_TRANSCRIPT,
        TargetComponent.TRANSCRIPT,
        TargetComponent.OUTPUT,
    ])


def generate_annotated_transcripts(u_sampler):
    start_len, stop_len = ANNOT_LEN_RANGE
    annot_len = stop_len - 1  # Start at max annotation length

    sampler = UpperBoundRejector(
        AnnotatedTranscriptSampler(u_sampler, annot_len=annot_len),
        REJECTOR_UBOUND
    )
    set_seed(SEED)
    val_samples = sampler(N_VAL)

    while annot_len >= start_len - 1:
        print(f"\n=== Generating ATL{annot_len} (uniform) ===")
        save_eval_only(val_samples, f"ATL{annot_len}", [
            TargetComponent.ANNOTATED_TRANSCRIPT,
            TargetComponent.TRANSCRIPT,
            TargetComponent.OUTPUT,
        ])
        # Trim annotations by one for next iteration
        val_samples.shorten_annot()
        annot_len -= 1


def main():
    print("=" * 60)
    print(f"Generating UNIFORM distribution eval data")
    print(f"  n_val={N_VAL}, ubound={UBOUND}, base={BASE}")
    print(f"  Output: {DATA_UNIFORM_DIR}")
    print("=" * 60)

    u_sampler = UniformGCDSampler(UBOUND)

    generate_baseline(u_sampler)
    generate_transcripts(u_sampler)
    generate_annotated_transcripts(u_sampler)

    print("\n" + "=" * 60)
    print("Done! Generated eval data for:")
    for d in sorted(DATA_UNIFORM_DIR.iterdir()):
        if d.is_dir() and (d / "eval.npz").exists():
            print(f"  - {d.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
