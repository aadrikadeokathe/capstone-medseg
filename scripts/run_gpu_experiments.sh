#!/usr/bin/env bash
# ==============================================================================
# scripts/run_gpu_experiments.sh
#
# Master GPU Execution Script for Episodic Meta-Learning & Zero-Shot Medical
# Image Segmentation Experiments.
# ==============================================================================

# Ensure script runs from the repository root directory regardless of where it is invoked
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Auto-detect python binary (prefer active python/python3)
if command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "ERROR: Neither 'python3' nor 'python' was found in your PATH!"
    echo "Please activate your virtual environment or conda environment first."
    exit 1
fi

echo "=================================================================="
echo " Starting Medical Segmentation GPU Training Pipeline"
echo " Repo Root: $REPO_ROOT"
echo " Python Binary: $($PY -c 'import sys; print(sys.executable)')"
echo " Timestamp: $(date)"
echo "=================================================================="

# Check GPU availability
echo ""
echo "[1/6] Checking GPU Environment..."
$PY -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available: {cuda_avail}')
if cuda_avail:
    print(f'Device Count: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name(0)}')
    print(f'VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')
else:
    print('WARNING: CUDA is NOT available! PyTorch will run on CPU.')
"

mkdir -p models/checkpoints
mkdir -p logs

# ------------------------------------------------------------------------------
# EXPERIMENT 1: Joint Episodic Meta-Learning Training (All 4 Organs)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[2/6] Running Joint Episodic Meta-Learning Training (30 Epochs)..."
echo "=================================================================="
$PY -u src/train_episodic.py \
    --epochs 30 \
    --episodes_per_epoch 500 \
    --num_layers 3 \
    --lr 1e-3 \
    --num_support 2 \
    --seed 42 \
    --checkpoint models/checkpoints/best_fusion_episodic.pt \
    --results_file logs/episodic_zeroshot_results.txt

# ------------------------------------------------------------------------------
# EXPERIMENT 2: Leave-One-Dataset-Out (LODO) Zero-Shot Matrix
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[3/6] Running Leave-One-Dataset-Out Zero-Shot Verification..."
echo "=================================================================="

echo "  -> (A) Held-Out Target: SPLEEN (Train on Liver, Heart, BrainTumour)"
$PY -u src/train_episodic.py \
    --train_datasets liver heart braintumour \
    --test_dataset spleen \
    --epochs 25 \
    --episodes_per_epoch 400 \
    --num_layers 3 \
    --lr 1e-3 \
    --num_support 2 \
    --checkpoint models/checkpoints/best_fusion_episodic_heldout_spleen.pt \
    --results_file logs/episodic_lodo_spleen.txt

echo "  -> (B) Held-Out Target: LIVER (Train on Spleen, Heart, BrainTumour)"
$PY -u src/train_episodic.py \
    --train_datasets spleen heart braintumour \
    --test_dataset liver \
    --epochs 25 \
    --episodes_per_epoch 400 \
    --num_layers 3 \
    --lr 1e-3 \
    --num_support 2 \
    --checkpoint models/checkpoints/best_fusion_episodic_heldout_liver.pt \
    --results_file logs/episodic_lodo_liver.txt

echo "  -> (C) Held-Out Target: HEART (Train on Spleen, Liver, BrainTumour)"
$PY -u src/train_episodic.py \
    --train_datasets spleen liver braintumour \
    --test_dataset heart \
    --epochs 25 \
    --episodes_per_epoch 400 \
    --num_layers 3 \
    --lr 1e-3 \
    --num_support 2 \
    --checkpoint models/checkpoints/best_fusion_episodic_heldout_heart.pt \
    --results_file logs/episodic_lodo_heart.txt

echo "  -> (D) Held-Out Target: BRAINTUMOUR (Train on Spleen, Liver, Heart)"
$PY -u src/train_episodic.py \
    --train_datasets spleen liver heart \
    --test_dataset braintumour \
    --epochs 25 \
    --episodes_per_epoch 400 \
    --num_layers 3 \
    --lr 1e-3 \
    --num_support 2 \
    --checkpoint models/checkpoints/best_fusion_episodic_heldout_braintumour.pt \
    --results_file logs/episodic_lodo_braintumour.txt

# ------------------------------------------------------------------------------
# EXPERIMENT 3: Few-Shot Support Context Ablation (K in {1, 2, 4, 8})
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[4/6] Running Shot-Count Ablation on Joint Episodic Checkpoint..."
echo "=================================================================="
$PY -u src/evaluate_shot_ablation.py \
    --checkpoint models/checkpoints/best_fusion_episodic.pt \
    --shots 1 2 4 8 \
    --output_txt logs/shot_ablation_results.txt \
    --output_png logs/shot_ablation_curve.png

# ------------------------------------------------------------------------------
# EXPERIMENT 4: Multi-Seed Robustness Runs (Seeds 123, 456 for Error Bars)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[5/6] Running Multi-Seed Robustness Runs (Seeds 123, 456)..."
echo "=================================================================="
if [ "$1" != "--skip-seeds" ]; then
    echo "  -> Running Seed 123..."
    $PY -u src/train_episodic.py \
        --epochs 30 \
        --episodes_per_epoch 500 \
        --num_layers 3 \
        --lr 1e-3 \
        --num_support 2 \
        --seed 123 \
        --checkpoint models/checkpoints/best_fusion_episodic_seed123.pt \
        --results_file logs/episodic_zeroshot_results_seed123.txt

    echo "  -> Running Seed 456..."
    $PY -u src/train_episodic.py \
        --epochs 30 \
        --episodes_per_epoch 500 \
        --num_layers 3 \
        --lr 1e-3 \
        --num_support 2 \
        --seed 456 \
        --checkpoint models/checkpoints/best_fusion_episodic_seed456.pt \
        --results_file logs/episodic_zeroshot_results_seed456.txt
else
    echo "  -> Skipping multi-seed runs (--skip-seeds passed)."
fi

echo ""
echo "=================================================================="
echo "[6/6] All GPU Experiments Completed Successfully!"
echo " Timestamp: $(date)"
echo " Results Summary:"
echo "   - Main Episodic Log (Seed 42):  logs/episodic_training.txt"
echo "   - Main Zero-Shot Table:         logs/episodic_zeroshot_results.txt"
echo "   - LODO Matrix Files:            logs/episodic_lodo_*.txt"
echo "   - Shot Ablation Table:          logs/shot_ablation_results.txt"
echo "   - Multi-Seed Results:           logs/episodic_zeroshot_results_seed*.txt"
echo "=================================================================="
