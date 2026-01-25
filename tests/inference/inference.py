"""
Batch inference for self-proving models on evaluation data.
Usage: python inference.py [--data-dir PATH]

Uses eval.npz (held-out data), not training data.
Reports 3 verification levels: output, transcript, annotated.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

import spm
import spm.data.tensor_repr as tensor_repr_module
from spm import MODELS_DIR
from spm.gpt.config import TrainerConfig
from spm.gpt.trainer import Trainer

# Configuration
NUM_EXAMPLES = 1000 # Number of eval examples to test per model
DEVICE = "mps" # "mps", "cuda", or "cpu"

# Token encoding constants
BASE = 210
PAD, PLUS, MINUS, DELIM_START = 210, 211, 212, 213


def parse_integers_from_tokens(tokens, vocab):
    """Extract integers from base-210 token sequence."""
    integers = []
    digits, negative = [], False

    def flush():
        nonlocal digits, negative
        if digits:
            num = sum(d * (BASE ** i) for i, d in enumerate(reversed(digits)))
            integers.append(-num if negative else num)
            digits, negative = [], False

    for tok in tokens:
        if tok == PAD or tok >= DELIM_START:
            flush()
        elif tok == PLUS:
            flush()
            negative = False
        elif tok == MINUS:
            flush()
            negative = True
        elif 0 <= tok < BASE:
            digits.append(tok)
    flush()
    return integers


def list_available_models():
    """Find model checkpoints in models/ directory."""
    if not MODELS_DIR.exists():
        return []

    models = []
    for f in sorted(MODELS_DIR.glob("*.pt")):
        name = f.stem
        if "Baseline" in name:
            mtype = "Baseline"
        elif match := re.search(r'ATL(\d+)', name):
            mtype = f"ATL{match.group(1)}"
        elif "-TL_" in name or name.startswith("TL_"):
            mtype = "TL"
        else:
            mtype = "Unknown"

        parts = name.split('-')
        dataset = parts[1].split('_iter')[0] if len(parts) >= 2 else None
        models.append({'name': name, 'type': mtype, 'path': f, 'dataset': dataset})
    return models


def select_model():
    """Interactive model selector. Returns list of models to test."""
    models = list_available_models()
    if not models:
        print("No models found in models/ directory.")
        sys.exit(1)

    print("\nAvailable models:")
    print(f"  0. [ALL] - Test all models")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m['type']:12s} - {m['name']}")

    while True:
        try:
            idx = int(input(f"\nSelect (0-{len(models)}): ").strip())
            if idx == 0:
                return models
            elif 1 <= idx <= len(models):
                return [models[idx - 1]]
            print(f"Enter 0-{len(models)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


def run_inference(model_info, quiet=False):
    """Run inference on a single model. quiet=True for batch mode."""
    name = model_info['name']
    dataset = model_info['dataset']
    is_baseline = model_info['type'] == 'Baseline'
    is_tl = model_info['type'] == 'TL'

    if not dataset:
        print(f"Could not parse dataset from {name}")
        return None

    if not quiet:
        print(f"\n{'='*70}")
        print(f"Model: {name}")
        print(f"Dataset: {dataset}, Device: {DEVICE}")
        print(f"{'='*70}")

    wandb.init(mode="disabled", reinit=True)

    config = {
        "load_ckpt": name, "data": dataset, "device": DEVICE,
        "epochs": 1, "n_layer": 8, "n_head": 8, "n_embd": 256,
    }

    try:
        trainer = Trainer(TrainerConfig.from_defaults(config))
        trainer.model.eval()
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None

    # Track per-example correctness for std calculation
    results_per_example = {"output": [], "transcript": [], "annotated": []}
    num_total = 0

    for _, inp, masks, masked in trainer.data.val_batches(device=trainer.cfg.device, batch_size=1):
        if num_total >= NUM_EXAMPLES:
            break

        with torch.no_grad():
            output = trainer.model.generate(
                inp.to(trainer.cfg.device),
                max_new_tokens=trainer.data.m.generation_size
            )

        output_generated = output[:, inp.shape[1]:]

        # Baseline only has "annotated" mask; others have all three
        this_correct = {}
        for level in ["output", "transcript", "annotated"]:
            if level not in masked:
                continue
            target = masked[level]
            target_hat = output_generated * masks[level]
            this_correct[level] = torch.all(torch.eq(target_hat, target), dim=1).item()
            results_per_example[level].append(1 if this_correct[level] else 0)

        num_total += 1

        if not quiet:
            inp_tokens = inp[0].cpu().numpy()
            all_ints = parse_integers_from_tokens(inp_tokens, trainer.data.m)
            a, b = all_ints[0], all_ints[1]
            if is_baseline:
                out_ok = "✓" if this_correct.get("annotated", False) else "✗"
                print(f"  {num_total}. GCD({a}, {b}) - output:{out_ok}")
            elif is_tl:
                out_ok = "✓" if this_correct.get("output", False) else "✗"
                trans_ok = "✓" if this_correct.get("transcript", False) else "✗"
                print(f"  {num_total}. GCD({a}, {b}) - output:{out_ok} transcript:{trans_ok}")
            else:
                out_ok = "✓" if this_correct.get("output", False) else "✗"
                trans_ok = "✓" if this_correct.get("transcript", False) else "✗"
                ann_ok = "✓" if this_correct.get("annotated", False) else "✗"
                print(f"  {num_total}. GCD({a}, {b}) - output:{out_ok} transcript:{trans_ok} annotated:{ann_ok}")

    # Calculate mean and std for each level
    def calc_stats(arr):
        if not arr:
            return None, None
        return 100 * np.mean(arr), 100 * np.std(arr)

    out_mean, out_std = calc_stats(results_per_example['annotated'] if is_baseline else results_per_example['output'])
    trans_mean, trans_std = (None, None) if is_baseline else calc_stats(results_per_example['transcript'])
    ann_mean, ann_std = (None, None) if (is_baseline or is_tl) else calc_stats(results_per_example['annotated'])

    results = {
        'model': model_info['type'],
        'name': name,
        'total': num_total,
        'output': (out_mean, out_std),
        'transcript': (trans_mean, trans_std) if trans_mean is not None else None,
        'annotated': (ann_mean, ann_std) if ann_mean is not None else None,
    }

    if not quiet:
        if is_baseline:
            print(f"\nResults: output={out_mean:.1f}±{out_std:.1f}% (Baseline: no transcript)")
        elif is_tl:
            print(f"\nResults: output={out_mean:.1f}±{out_std:.1f}%, "
                  f"transcript={trans_mean:.1f}±{trans_std:.1f}%")
        else:
            print(f"\nResults: output={out_mean:.1f}±{out_std:.1f}%, "
                  f"transcript={trans_mean:.1f}±{trans_std:.1f}%, "
                  f"annotated={ann_mean:.1f}±{ann_std:.1f}%")

    return results


def print_summary_table(all_results):
    """Print summary table with mean±std for all models."""
    print(f"\n{'='*78}")
    print("SUMMARY TABLE")
    print(f"{'='*78}")
    print(f"{'Model':<12} {'Output':>14} {'Transcript':>14} {'Annotated':>14}")
    print(f"{'-'*12} {'-'*14} {'-'*14} {'-'*14}")

    for r in all_results:
        if r is None:
            continue
        out_str = f"{r['output'][0]:.1f}±{r['output'][1]:.1f}%"
        trans_str = "N/A" if r['transcript'] is None else f"{r['transcript'][0]:.1f}±{r['transcript'][1]:.1f}%"
        ann_str = "N/A" if r['annotated'] is None else f"{r['annotated'][0]:.1f}±{r['annotated'][1]:.1f}%"
        print(f"{r['model']:<12} {out_str:>14} {trans_str:>14} {ann_str:>14}")

    print(f"{'='*78}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch inference for self-proving models")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Custom data directory (default: data/)")
    args = parser.parse_args()

    # Override DATA_DIR if custom path provided
    if args.data_dir:
        spm.DATA_DIR = args.data_dir
        tensor_repr_module.DATA_DIR = args.data_dir  # Patch the local import too
        print(f"Using custom data directory: {args.data_dir}")

    print("="*70)
    print(f"Self-Proving Model Inference (n={NUM_EXAMPLES}, device={DEVICE})")
    print("="*70)

    models_to_test = select_model()
    quiet = len(models_to_test) > 1

    all_results = []
    for model_info in models_to_test:
        result = run_inference(model_info, quiet=quiet)
        all_results.append(result)

        if quiet and result:
            out_str = f"{result['output'][0]:.1f}±{result['output'][1]:.1f}%"
            trans_str = "N/A" if result['transcript'] is None else f"{result['transcript'][0]:.1f}±{result['transcript'][1]:.1f}%"
            print(f"{result['model']:<12} output={out_str}, transcript={trans_str}")

    if len(models_to_test) > 1:
        print_summary_table(all_results)


if __name__ == "__main__":
    main()
