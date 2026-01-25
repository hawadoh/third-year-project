#!/usr/bin/env python3
"""
Interactive inference for self-proving GCD models.
Select a model, input two integers, see the GCD and proof verification.
"""
import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
import wandb

from spm.gpt.config import TrainerConfig
from spm.gpt.trainer import Trainer
from spm.utils import egcd, is_egcd

# Base-210 token encoding (Appendix F of the Amit et al. 2024 paper)
BASE = 210
PAD, PLUS, MINUS, DELIM_START = 210, 211, 212, 213


def encode_base210(num):
    """Convert integer to base-210 digits, most significant first."""
    if num == 0:
        return [0]
    digits = []
    num = abs(num)
    while num > 0:
        digits.append(num % BASE)
        num //= BASE
    return digits[::-1]


def encode_input_pair(a, b):
    """Encode (a, b) as tokens: [sign, digits..., delim, sign, digits..., delim]."""
    tokens = [PLUS if a >= 0 else MINUS] + encode_base210(abs(a)) + [213]
    tokens += [PLUS if b >= 0 else MINUS] + encode_base210(abs(b)) + [214]
    return tokens


def parse_integers_from_tokens(tokens):
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
    models_dir = Path("models")
    if not models_dir.exists():
        return []

    models = []
    for f in sorted(models_dir.glob("*.pt")):
        name = f.stem
        if "Baseline" in name:
            mtype = "Baseline"
        elif match := re.search(r'ATL(\d+)', name):
            mtype = f"ATL{match.group(1)}"
        elif "-TL_" in name or name.startswith("TL_"):
            mtype = "TL"
        else:
            mtype = "Unknown"
        models.append({'name': name, 'type': mtype, 'path': f})
    return models


def select_model():
    """Prompt user to choose a model."""
    models = list_available_models()
    if not models:
        print("No models found in models/ directory.")
        sys.exit(1)

    print("\nAvailable models:")
    for i, m in enumerate(models, 1):
        print(f"  {i}. {m['type']:12s} - {m['name']}")

    while True:
        try:
            choice = int(input(f"\nSelect (1-{len(models)}): ").strip()) - 1
            if 0 <= choice < len(models):
                return models[choice]
            print(f"Enter 1-{len(models)}")
        except (ValueError, KeyboardInterrupt):
            print("\nExiting.")
            sys.exit(0)


def get_user_integers():
    """Get two positive integers from user."""
    print("\nEnter two positive integers (model trained on 1-10000):")
    while True:
        try:
            a = int(input("  a: ").strip())
            b = int(input("  b: ").strip())
            if a <= 0 or b <= 0:
                print("Must be positive.")
                continue
            if a > 100000 or b > 100000:
                if input("Large numbers may not work well. Continue? (y/n): ").lower() != 'y':
                    continue
            return a, b
        except ValueError:
            print("Invalid integer.")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            sys.exit(0)


def load_model(model_info, device):
    """Load model checkpoint."""
    name = model_info['name']
    parts = name.split('-')
    if len(parts) < 2:
        print("Could not parse dataset from model filename.")
        sys.exit(1)

    dataset = parts[1].split('_iter')[0]
    print(f"\nLoading {model_info['type']} model ({dataset}) on {device}...")

    wandb.init(mode="disabled")
    config = {
        "load_ckpt": name, "data": dataset, "device": device,
        "epochs": 1, "n_layer": 8, "n_head": 8, "n_embd": 256,
    }

    try:
        trainer = Trainer(TrainerConfig.from_defaults(config))
        trainer.model.eval()
        print("Model loaded.")
        return trainer, dataset
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_proof(trainer, a, b):
    """Generate GCD proof for inputs (a, b)."""
    from spm.data.samples import Transcript
    from spm.data.str_repr import EncodedSamples

    # Encode using trainer's method to match training format
    k, u, v = egcd(a, b)
    samples = Transcript(
        a=np.array([a]), b=np.array([b]),
        k=np.array([k]), u=np.array([u]), v=np.array([v])
    )
    encoded = EncodedSamples(samples, trainer.data.m.base)
    input_arr = trainer.data.input_arr(encoded.get_inputs(0))
    inp = torch.tensor([input_arr], dtype=torch.long).to(trainer.cfg.device)

    if inp.shape[1] > trainer.data.m.block_size:
        inp = inp[:, :trainer.data.m.block_size]

    print("Generating...", end='', flush=True)
    with torch.no_grad():
        output = trainer.model.generate(inp, max_new_tokens=trainer.data.m.generation_size)
    print(" done.")

    return parse_integers_from_tokens(output[0].cpu().numpy())


def verify_and_display(a, b, integers, model_type):
    """Verify proof and show results."""
    true_gcd = math.gcd(a, b)
    _, true_u, true_v = egcd(a, b)

    print(f"\n{'='*60}")
    print(f"Input: GCD({a}, {b})")
    print(f"True:  GCD={true_gcd}, u={true_u}, v={true_v}")
    print(f"       Check: {true_u}*{a} + {true_v}*{b} = {true_u*a + true_v*b}")

    if len(integers) < 3:
        print(f"\nModel output too short: {integers}")
        return False

    # Parse: [a, b, k, ..., u, v] or [a, b, k] for baseline
    if integers[0] != a or integers[1] != b:
        print(f"\nWarning: echoed inputs {integers[:2]} don't match [{a}, {b}]")

    k_pred = integers[2]
    if model_type == "Baseline":
        u_pred, v_pred = None, None
    elif len(integers) >= 5:
        u_pred, v_pred = integers[-2], integers[-1]
    else:
        u_pred, v_pred = None, None

    print(f"\nModel: GCD={k_pred}", end='')
    if u_pred is not None:
        print(f", u={u_pred}, v={v_pred}")
    else:
        print()

    # Check GCD
    gcd_ok = (k_pred == true_gcd)
    print(f"GCD correct: {'yes' if gcd_ok else 'NO'}")

    if model_type == "Baseline":
        print("(Baseline model outputs GCD only, no proof)")
        return gcd_ok

    if u_pred is None:
        print("No Bezout coefficients in output.")
        return False

    # Verify Bezout identity
    bezout = u_pred * a + v_pred * b
    print(f"\nVerifying: {u_pred}*{a} + {v_pred}*{b} = {bezout}")
    valid = is_egcd(a, b, k_pred, u_pred, v_pred)

    if valid:
        print("PROOF VERIFIED")
    else:
        print("PROOF FAILED")
        if bezout != k_pred:
            print(f"  Linear combination gives {bezout}, not {k_pred}")
        if a % k_pred != 0:
            print(f"  {k_pred} does not divide {a}")
        if b % k_pred != 0:
            print(f"  {k_pred} does not divide {b}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="Interactive self-proving GCD inference")
    parser.add_argument("--device", default="mps", choices=["mps", "cuda", "cpu"])
    args = parser.parse_args()

    print("="*60)
    print("Self-Proving GCD Inference")
    print("The model computes GCD(a,b) and proves it via Bezout: k = u*a + v*b")
    print("="*60)

    model_info = select_model()
    trainer, _ = load_model(model_info, args.device)

    while True:
        try:
            a, b = get_user_integers()
            integers = generate_proof(trainer, a, b)
            verify_and_display(a, b, integers, model_info['type'])

            if input("\nAnother? (y/n): ").strip().lower() != 'y':
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            break

    print("\nGoodbye.")


if __name__ == "__main__":
    main()
