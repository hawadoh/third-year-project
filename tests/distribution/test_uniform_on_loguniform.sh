#!/bin/bash
# DONE ON MY LOCAL MACHINE USING MPS
# Test 2: Evaluate uniform-trained models on LOG-UNIFORM distribution
# Strategy: Temporarily replace uniform data folders with symlinks to log-uniform data
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "========================================"
echo "TEST 2: Uniform models → Log-uniform eval"
echo "========================================"

# Create model symlinks in models/
echo "Creating model symlinks..."
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-Baseline_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-TL_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"
ln -sf "$PROJECT_ROOT/uniform-models/8x8x256-ATL7_uniform_1e4_m1e7_b210_iter100000.pt" \
       "$PROJECT_ROOT/models/"

# Backup uniform data folders and replace with symlinks to log-uniform data
echo "Backing up uniform data and creating symlinks to log-uniform data..."
for ds in Baseline TL ATL7; do
    mv "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210" \
       "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210.bak"
    ln -s "$PROJECT_ROOT/data/${ds}_1e4_m1e7_b210" \
          "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210"
done

# Run inference
echo "Running inference..."
cd "$PROJECT_ROOT/tests/inference"
PYTHONPATH="$PROJECT_ROOT" python inference.py

# Cleanup: restore uniform data folders
echo "Restoring uniform data folders..."
for ds in Baseline TL ATL7; do
    rm -f "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210"
    mv "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210.bak" \
       "$PROJECT_ROOT/data/${ds}_uniform_1e4_m1e7_b210"
done

# Cleanup model symlinks
echo "Cleaning up model symlinks..."
rm -f "$PROJECT_ROOT/models/8x8x256-Baseline_uniform_1e4_m1e7_b210_iter100000.pt"
rm -f "$PROJECT_ROOT/models/8x8x256-TL_uniform_1e4_m1e7_b210_iter100000.pt"
rm -f "$PROJECT_ROOT/models/8x8x256-ATL7_uniform_1e4_m1e7_b210_iter100000.pt"

echo "Done!"
