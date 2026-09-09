#!/usr/bin/env bash
# ==============================================================================
# scripts/run_gpu_experiments.sh
#
# Master GPU Execution Script for Episodic Meta-Learning & Zero-Shot Medical
# Image Segmentation Experiments.
# ==============================================================================

set -e

echo "=================================================================="
echo " Starting Medical Segmentation GPU Training Pipeline"
echo " Timestamp: $(date)"
echo "=================================================================="

# Check GPU availability
echo "[1/5] Checking GPU Environment..."
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

mkdir -p models/checkpoints
mkdir -p logs

# ------------------------------------------------------------------------------
# EXPERIMENT 1: Joint Episodic Meta-Learning Training (All 4 Organs)
# ------------------------------------------------------------------------------
echo ""
echo "=================================================================="
echo "[2/5] Running Joint Episodic Meta-Learning Training (30 Epochs)..."
echo "=================================================================="
python src/train_episodic.py \
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
echo "[3/5] Running Leave-One-Dataset-Out Zero-Shot Verification..."
echo "=================================================================="

echo "  -> (A) Held-Out Target: SPLEEN (Train on Liver, Heart, BrainTumour)"
python src/train_episodic.py \
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
python src/train_episodic.py \
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
python src/train_episodic.py \
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
python src/train_episodic.py \
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
echo "[4/5] Running Shot-Count Ablation on Joint Episodic Checkpoint..."
echo "=================================================================="
python src/evaluate_shot_ablation.py \
    --checkpoint models/checkpoints/best_fusion_episodic.pt \
    --shots 1 2 4 8 \
    --output_txt logs/shot_ablation_results.txt \
    --output_png logs/shot_ablation_curve.png

echo ""
echo "=================================================================="
echo "[5/5] All GPU Experiments Completed Successfully!"
echo " Timestamp: $(date)"
echo " Results Summary:"
echo "   - Main Episodic Log:    logs/episodic_training.txt"
echo "   - Main Zero-Shot Table: logs/episodic_zeroshot_results.txt"
echo "   - LODO Matrix Files:    logs/episodic_lodo_*.txt"
echo "   - Shot Ablation Table:  logs/shot_ablation_results.txt"
echo "=================================================================="
