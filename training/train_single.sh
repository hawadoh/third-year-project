#!/bin/bash
# Helper script to mount data.zip and train a single model
# Usage: ./train_single.sh <dataset> <seed>

set -euo pipefail

DATASET=$1
SEED=$2

# === PATH CONFIGURATION ===

# ORIGINAL (data.zip in home directory):
# WORKSPACE_DIR="/dcs/23/u5514611/cs310/self-proving-models"
# DATA_ZIP="$WORKSPACE_DIR/data.zip"
# MOUNT_POINT="$TMPDIR/spm_data_$$"
# SPM_DIR="$WORKSPACE_DIR/spm"

# NEW: Use /dcs/large/ for permanent storage (100GB allocation)
DATADIR="/dcs/large/u5514611"
WORKSPACE_DIR="/dcs/23/u5514611/cs310/self-proving-models"
DATA_ZIP="$DATADIR/data/data.zip"
MOUNT_POINT="$TMPDIR/spm_data_$$"
SPM_DIR="$WORKSPACE_DIR/spm"

# Export paths for Python (spm/__init__.py reads these)
export SPM_MODELS_DIR="$DATADIR/models"
export SPM_ANALYSIS_DIR="$WORKSPACE_DIR/logs"  # Keep logs in repo
export SPM_DATA_DIR="$WORKSPACE_DIR/data"      # Symlinks created here

# Ensure output directories exist
mkdir -p "$SPM_MODELS_DIR"
mkdir -p "$SPM_ANALYSIS_DIR"

echo "========================================="
echo "Training: $DATASET (seed $SEED)"
echo "Node: $(hostname)"
echo "TMPDIR: $TMPDIR"
echo "========================================="

# Clean up any stale data (disable exit on error for this section)
set +e
echo "Cleaning up any stale data..."
# ORIGINAL: Clean up stale fuse-zip mounts
# fusermount3 -u "$MOUNT_POINT" 2>/dev/null
# NEW: Just remove the directory (works for both extracted and mounted)
rm -rf "$MOUNT_POINT" 2>/dev/null
set -e

# Create mount point
mkdir -p "$MOUNT_POINT"

# ORIGINAL: Mount data.zip using fuse-zip (read-only) - SLOWER due to FUSE overhead
# echo "Mounting $DATA_ZIP to $MOUNT_POINT..."
# fuse-zip -r "$DATA_ZIP" "$MOUNT_POINT"
# Verify mount succeeded
# if [ ! -d "$MOUNT_POINT/data" ]; then
#     echo "ERROR: Mount failed - data directory not found"
#     fusermount3 -u "$MOUNT_POINT" 2>/dev/null || true
#     exit 1
# fi
# echo "Mount successful. Data available at: $MOUNT_POINT/data"

# NEW: Extract data.zip to TMPDIR for faster I/O (avoids FUSE overhead)
echo "Extracting $DATA_ZIP to $MOUNT_POINT..."
unzip -q "$DATA_ZIP" -d "$MOUNT_POINT"

# Verify extraction succeeded
if [ ! -d "$MOUNT_POINT/data" ]; then
    echo "ERROR: Extraction failed - data directory not found"
    exit 1
fi

echo "Extraction successful. Data available at: $MOUNT_POINT/data"
ls -lh "$MOUNT_POINT/data" | head -5

# Create data directory if it doesn't exist
DATA_DIR="$WORKSPACE_DIR/data"
mkdir -p "$DATA_DIR"

# ORIGINAL (simple symlink, assumes dataset always in data.zip):
# DATASET_SYMLINK="$DATA_DIR/$DATASET"
# if [ -L "$DATASET_SYMLINK" ]; then
#     echo "Removing existing dataset symlink: $DATASET_SYMLINK"
#     rm "$DATASET_SYMLINK" || unlink "$DATASET_SYMLINK" 2>/dev/null || true
# elif [ -d "$DATASET_SYMLINK" ]; then
#     echo "WARNING: $DATASET_SYMLINK exists as a directory. Removing it..."
#     rm -rf "$DATASET_SYMLINK"
# elif [ -e "$DATASET_SYMLINK" ]; then
#     echo "Removing existing file: $DATASET_SYMLINK"
#     rm -f "$DATASET_SYMLINK" || true
# fi
# echo "Creating dataset symlink: $DATASET_SYMLINK -> $MOUNT_POINT/data/$DATASET"
# ln -sfn "$MOUNT_POINT/data/$DATASET" "$DATASET_SYMLINK"

# NEW: Support datasets both in data.zip AND generated locally (e.g., uniform data)
DATASET_PATH="$DATA_DIR/$DATASET"

# Check if dataset exists in extracted zip
if [ -d "$MOUNT_POINT/data/$DATASET" ]; then
    echo "Dataset found in data.zip"
    USE_SYMLINK=true
    EXTRACTED_DATASET="$MOUNT_POINT/data/$DATASET"
elif [ -d "$DATASET_PATH" ] && [ ! -L "$DATASET_PATH" ]; then
    # Dataset exists as a real directory (e.g., uniform data generated locally)
    echo "Dataset found in workspace: $DATASET_PATH"
    echo "Using existing data directory (not from data.zip)"
    USE_SYMLINK=false
else
    echo "ERROR: Dataset '$DATASET' not found in data.zip or workspace"
    echo "  - Checked: $MOUNT_POINT/data/$DATASET"
    echo "  - Checked: $DATASET_PATH"
    echo ""
    echo "For uniform datasets, generate training data first:"
    echo "  python data-uniform/generate_uniform_data.py --n_train 10240000 --datasets $DATASET"
    exit 1
fi

# Create symlink only if using data from zip (not if data exists directly)
if [ "$USE_SYMLINK" = true ]; then
    # Remove existing symlink/file for this dataset if it exists
    if [ -L "$DATASET_PATH" ]; then
        echo "Removing existing dataset symlink: $DATASET_PATH"
        rm "$DATASET_PATH" || unlink "$DATASET_PATH" 2>/dev/null || true
    elif [ -d "$DATASET_PATH" ]; then
        echo "WARNING: $DATASET_PATH exists as a directory but we need symlink. Skipping removal."
        echo "Using existing directory instead of creating symlink."
        USE_SYMLINK=false
    elif [ -e "$DATASET_PATH" ]; then
        echo "Removing existing file: $DATASET_PATH"
        rm -f "$DATASET_PATH" || true
    fi

    if [ "$USE_SYMLINK" = true ]; then
        echo "Creating dataset symlink: $DATASET_PATH -> $EXTRACTED_DATASET"
        ln -sfn "$EXTRACTED_DATASET" "$DATASET_PATH"
    fi
fi

# Trap to ensure cleanup on exit (success or failure)
# ORIGINAL cleanup (always removes symlink):
# cleanup() {
#     echo "Cleaning up..."
#     if [ -L "$DATASET_SYMLINK" ]; then
#         echo "Removing dataset symlink: $DATASET_SYMLINK"
#         rm "$DATASET_SYMLINK"
#     fi
#     echo "Removing extracted data from $MOUNT_POINT..."
#     rm -rf "$MOUNT_POINT" 2>/dev/null || true
# }

# NEW cleanup (preserves real directories, only removes symlinks)
cleanup() {
    echo "Cleaning up..."
    # Only remove if it's a symlink (not a real directory with uniform data)
    if [ -L "$DATASET_PATH" ]; then
        echo "Removing dataset symlink: $DATASET_PATH"
        rm "$DATASET_PATH"
    else
        echo "Keeping dataset directory (not a symlink): $DATASET_PATH"
    fi

    # Remove extracted data from TMPDIR
    echo "Removing extracted data from $MOUNT_POINT..."
    rm -rf "$MOUNT_POINT" 2>/dev/null || true
}
trap cleanup EXIT

# Hyperparameters from annot_len.sh
EPOCHS=10
BETA1=0.733
LR=0.0007
BATCH_SIZE=1024
DECAY_LR=10
GRAD_CLIP=2
N_EMBD=256
N_HEAD=8
N_LAYER=8
WANDB_PROJ="self-proving-models"
EVAL_INTERVAL=10000  # Less frequent evaluation (was 2000)
LOG_INTERVAL=1000    # Less frequent logging (was 100)

# Build training command
cd "$SPM_DIR"

# Disable wandb to avoid authentication issues in batch jobs
TRAIN_CMD="python train.py \
    --device=cuda \
    --dropout=0 \
    --eval_batch_size=512 \
    --eval_interval=$EVAL_INTERVAL \
    --log_interval=$LOG_INTERVAL \
    --warmup_iters=0 \
    --epochs=$EPOCHS \
    --beta1=$BETA1 \
    --learning_rate=$LR \
    --batch_size=$BATCH_SIZE \
    --decay_lr=$DECAY_LR \
    --grad_clip=$GRAD_CLIP \
    --n_embd=$N_EMBD \
    --n_head=$N_HEAD \
    --n_layer=$N_LAYER \
    --seed=$SEED \
    --data=$DATASET"

# Save model checkpoints if seed is 0
# # OLD: Save every epochs (10000 iters) + final checkpoint
# With max_iters=100000 and 10 epochs, each epoch ≈ 10000 iters
if [ "$SEED" -eq 0 ]; then
    # TRAIN_CMD="$TRAIN_CMD --save_iters 10000 20000 30000 40000 50000 60000 70000 80000 90000 -1"
    TRAIN_CMD="$TRAIN_CMD --save_iters -1"
fi

echo "Starting training..."
echo "Command: $TRAIN_CMD"
echo ""

# Run training
eval $TRAIN_CMD

echo "Training complete for $DATASET (seed $SEED)"
