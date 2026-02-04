#!/bin/bash
# DONE ON MY LOCAL MACHINE USING MPS
# Test 1: Evaluate uniform-trained models on UNIFORM distribution
# Data: data/*_uniform_* folders (copied from data-uniform/ with correct names)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "TEST 1: Uniform models → Uniform eval"
echo "========================================"

# Create model symlinks in models/
echo "Creating model symlinks..."
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-Baseline_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-TL_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-ATL7_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"

# Data already exists in data/*_uniform_* folders (no symlinks needed)

# Run inference
echo "Running inference..."
cd "$PROJECT_ROOT/tests/inference"
PYTHONPATH="$PROJECT_ROOT" python inference.py

# Cleanup model symlinks only
echo "Cleaning up model symlinks..."
rm -f "$PROJECT_ROOT/models/8x8x256-Baseline_uniform_1e4_m1e7_b210_iter100000.pt"
rm -f "$PROJECT_ROOT/models/8x8x256-TL_uniform_1e4_m1e7_b210_iter100000.pt"
rm -f "$PROJECT_ROOT/models/8x8x256-ATL7_uniform_1e4_m1e7_b210_iter100000.pt"

echo "Done!"
