"""
run_gpu_experiments.py

Cross-platform Master GPU Execution Runner for Medical Segmentation Experiments.
Runs on Linux, Windows, and Mac without needing bash.
"""

import os
import sys
import subprocess
import time
import argparse
import torch

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_command(cmd_list, description):
    print("\n" + "=" * 70)
    print(f" >>> {description}")
    print(" >>> Command: " + " ".join(cmd_list))
    print("=" * 70 + "\n")
    start = time.time()
    result = subprocess.run(cmd_list, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"\n[ERROR] Command failed with exit code {result.returncode} after {elapsed:.1f}s!")
        sys.exit(result.returncode)
    print(f"\n[SUCCESS] Completed in {elapsed:.1f}s.")


def main():
    parser = argparse.ArgumentParser(description="Master GPU Experiment Runner")
    parser.add_argument("--skip_seeds", action="store_true", help="Skip multi-seed runs (Seeds 123, 456)")
    parser.add_argument("--dry_run", action="store_true", help="Run in dry run mode for testing")
    parser.add_argument("--only_joint", action="store_true", help="Run only the main joint episodic training")
    parser.add_argument("--only_lodo", action="store_true", help="Run only the Leave-One-Out experiments")
    parser.add_argument("--only_ablation", action="store_true", help="Run only the shot ablation")
    args = parser.parse_args()

    py_exe = sys.executable

    print("==================================================================")
    print(" Starting Medical Segmentation GPU Training Pipeline")
    print(f" Working Directory: {PROJECT_ROOT}")
    print(f" Python Binary:     {py_exe}")
    print(f" PyTorch Version:   {torch.__version__}")
    cuda_avail = torch.cuda.is_available()
    print(f" CUDA Available:    {cuda_avail}")
    if cuda_avail:
        print(f" GPU Device:        {torch.cuda.get_device_name(0)}")
        print(f" GPU Memory:        {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print(" [WARNING] CUDA is NOT available! Running on CPU.")
    print("==================================================================")

    os.makedirs(os.path.join(PROJECT_ROOT, "models", "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

    dry_flag = ["--dry_run"] if args.dry_run else []

    # 1. Joint Episodic Training
    if not args.only_lodo and not args.only_ablation:
        run_command(
            [
                py_exe, "-u", "src/train_episodic.py",
                "--epochs", "40" if not args.dry_run else "2",
                "--episodes_per_epoch", "500" if not args.dry_run else "50",
                "--num_layers", "3",
                "--lr", "5e-4",
                "--num_support", "2",
                "--seed", "42",
                "--checkpoint", "models/checkpoints/best_fusion_episodic.pt",
                "--results_file", "logs/episodic_zeroshot_results.txt"
            ] + dry_flag,
            "EXPERIMENT 1: Joint Episodic Meta-Learning Training (All 4 Organs - 40 Epochs)"
        )

    # 2. Leave-One-Dataset-Out (LODO) Matrix
    if not args.only_joint and not args.only_ablation:
        lodo_configs = [
            ("spleen", ["liver", "heart", "braintumour"]),
            ("liver", ["spleen", "heart", "braintumour"]),
            ("heart", ["spleen", "liver", "braintumour"]),
            ("braintumour", ["spleen", "liver", "heart"]),
        ]
        for heldout, train_ds in lodo_configs:
            run_command(
                [
                    py_exe, "-u", "src/train_episodic.py",
                    "--train_datasets", *train_ds,
                    "--test_dataset", heldout,
                    "--epochs", "30" if not args.dry_run else "2",
                    "--episodes_per_epoch", "400" if not args.dry_run else "50",
                    "--num_layers", "3",
                    "--lr", "5e-4",
                    "--num_support", "2",
                    "--checkpoint", f"models/checkpoints/best_fusion_episodic_heldout_{heldout}.pt",
                    "--results_file", f"logs/episodic_lodo_{heldout}.txt"
                ] + dry_flag,
                f"EXPERIMENT 2: Leave-One-Out Zero-Shot (Held-Out: {heldout.upper()})"
            )

    # 3. Few-Shot Context Ablation (K in {1, 2, 4, 8})
    if not args.only_joint and not args.only_lodo:
        run_command(
            [
                py_exe, "-u", "src/evaluate_shot_ablation.py",
                "--checkpoint", "models/checkpoints/best_fusion_episodic.pt",
                "--shots", "1", "2", "4", "8",
                "--output_txt", "logs/shot_ablation_results.txt",
                "--output_png", "logs/shot_ablation_curve.png"
            ] + dry_flag,
            "EXPERIMENT 3: Few-Shot Support Context Ablation (K in {1, 2, 4, 8})"
        )

    # 4. Multi-Seed Robustness
    if not args.skip_seeds and not args.only_joint and not args.only_lodo and not args.only_ablation and not args.dry_run:
        for seed in [123, 456]:
            run_command(
                [
                    py_exe, "-u", "src/train_episodic.py",
                    "--epochs", "30",
                    "--episodes_per_epoch", "500",
                    "--num_layers", "3",
                    "--lr", "1e-3",
                    "--num_support", "2",
                    "--seed", str(seed),
                    "--checkpoint", f"models/checkpoints/best_fusion_episodic_seed{seed}.pt",
                    "--results_file", f"logs/episodic_zeroshot_results_seed{seed}.txt"
                ],
                f"EXPERIMENT 4: Multi-Seed Robustness Run (Seed {seed})"
            )

    print("\n" + "=" * 70)
    print(" ALL GPU EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
