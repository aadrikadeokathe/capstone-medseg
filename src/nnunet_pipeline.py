"""
src/nnunet_pipeline.py

Master pipeline script for nnU-Net v2 benchmark training on MSD datasets:
- Task02 Heart (Dataset002_Heart)
- Task03 Liver (Dataset003_Liver)
- Task01 BrainTumour (Dataset001_BrainTumour)
- Task09 Spleen (Dataset009_Spleen)

Converts raw NIfTI scans into nnU-Net v2 directory structure and generates dataset.json.
Provides execution commands for plan & preprocess, training (2D, 250 epochs), and evaluation.
"""

import os
import sys
import glob
import json
import shutil
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_BASE_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
NNUNET_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "nnunet_raw")
NNUNET_PREPROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "nnunet_preprocessed")
NNUNET_RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "nnunet_results")

DATASET_CONFIGS = {
    "heart": {
        "id": 2,
        "name": "Dataset002_Heart",
        "raw_dir_name": "Task02_Heart",
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "left_atrium": 1},
        "file_ending": ".nii.gz",
        "max_cases": 20,
    },
    "liver": {
        "id": 3,
        "name": "Dataset003_Liver",
        "raw_dir_name": "Task03_Liver",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 1},
        "file_ending": ".nii.gz",
        "max_cases": 20,
    },
    "braintumour": {
        "id": 1,
        "name": "Dataset001_BrainTumour",
        "raw_dir_name": "Task01_BrainTumour",
        "channel_names": {"0": "FLAIR"},
        "labels": {"background": 0, "tumour": 1},
        "file_ending": ".nii.gz",
        "max_cases": 15,
    },
    "spleen": {
        "id": 9,
        "name": "Dataset009_Spleen",
        "raw_dir_name": "Task09_Spleen",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "spleen": 1},
        "file_ending": ".nii.gz",
        "max_cases": 10,
    },
}


def convert_dataset_to_nnunet(dataset_key: str):
    cfg = DATASET_CONFIGS[dataset_key]
    raw_src_dir = os.path.join(RAW_BASE_DIR, cfg["raw_dir_name"])
    target_dir = os.path.join(NNUNET_RAW_DIR, cfg["name"])

    images_tr_src = os.path.join(raw_src_dir, "imagesTr")
    labels_tr_src = os.path.join(raw_src_dir, "labelsTr")

    if not os.path.exists(images_tr_src) or not os.path.exists(labels_tr_src):
        print(f"[WARN] Raw data directories for {dataset_key} not found at {raw_src_dir}. Skipping.")
        return False

    images_tr_dst = os.path.join(target_dir, "imagesTr")
    labels_tr_dst = os.path.join(target_dir, "labelsTr")

    os.makedirs(images_tr_dst, exist_ok=True)
    os.makedirs(labels_tr_dst, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_tr_src, "*.nii.gz")))
    max_cases = cfg.get("max_cases", len(image_files))
    image_files = image_files[:max_cases]

    print(f"\n[INFO] Converting {len(image_files)} cases for {cfg['name']}...")

    num_converted = 0
    for img_p in image_files:
        base = os.path.basename(img_p)
        case_id = base.replace(".nii.gz", "")

        lbl_p = os.path.join(labels_tr_src, base)
        if not os.path.exists(lbl_p):
            continue

        dst_img = os.path.join(images_tr_dst, f"{case_id}_0000.nii.gz")
        dst_lbl = os.path.join(labels_tr_dst, f"{case_id}.nii.gz")

        if not os.path.exists(dst_img):
            shutil.copyfile(img_p, dst_img)
        if not os.path.exists(dst_lbl):
            shutil.copyfile(lbl_p, dst_lbl)

        num_converted += 1

    print(f"[INFO] Successfully prepared {num_converted} volumes in {target_dir}")

    # Generate dataset.json
    d_json = {
        "channel_names": cfg["channel_names"],
        "labels": cfg["labels"],
        "numTraining": num_converted,
        "file_ending": cfg["file_ending"],
    }
    json_path = os.path.join(target_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(d_json, f, indent=4)
    print(f"[INFO] Generated dataset.json at {json_path}")
    return True


def print_environment_setup():
    print("\n" + "=" * 70)
    print(" NNU-NET ENVIRONMENT VARIABLES (REQUIRED ON WINDOWS GPU PC)")
    print("=" * 70)
    print("In PowerShell, run:")
    print('  $env:nnUNet_raw = "' + NNUNET_RAW_DIR + '"')
    print('  $env:nnUNet_preprocessed = "' + NNUNET_PREPROCESSED_DIR + '"')
    print('  $env:nnUNet_results = "' + NNUNET_RESULTS_DIR + '"')
    print("\nIn Command Prompt (CMD), run:")
    print(f'  set nnUNet_raw={NNUNET_RAW_DIR}')
    print(f'  set nnUNet_preprocessed={NNUNET_PREPROCESSED_DIR}')
    print(f'  set nnUNet_results={NNUNET_RESULTS_DIR}')
    print("=" * 70)


def print_training_commands():
    print("\n" + "=" * 70)
    print(" NNU-NET TRAINING COMMANDS (2D CONFIGURATION, 250 EPOCHS)")
    print("=" * 70)
    for ds_key in ["heart", "liver", "braintumour"]:
        cfg = DATASET_CONFIGS[ds_key]
        d_id = cfg["id"]
        print(f"\n--- {cfg['name']} (Dataset ID: {d_id}) ---")
        print(f"1. Plan & Preprocess:")
        print(f"   nnUNetv2_plan_and_preprocess -d {d_id} --verify_dataset_integrity")
        print(f"2. Train 2D Fold 0:")
        print(f"   nnUNetv2_train {d_id} 2d 0 -tr nnUNetTrainer_250epochs")
        print(f"3. Find Validation Metrics in:")
        print(f"   data\\nnunet_results\\{cfg['name']}\\nnUNetTrainer_250epochs__nnUNetPlans__2d\\fold_0\\summary.json")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="nnU-Net v2 Dataset Converter & Pipeline Helper")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["heart", "liver", "braintumour", "spleen", "all"],
        help="Dataset to convert to nnU-Net v2 format",
    )
    args = parser.parse_args()

    os.makedirs(NNUNET_RAW_DIR, exist_ok=True)
    os.makedirs(NNUNET_PREPROCESSED_DIR, exist_ok=True)
    os.makedirs(NNUNET_RESULTS_DIR, exist_ok=True)

    print_environment_setup()

    datasets_to_run = ["heart", "liver", "braintumour", "spleen"] if args.dataset == "all" else [args.dataset]
    for d in datasets_to_run:
        convert_dataset_to_nnunet(d)

    print_training_commands()


if __name__ == "__main__":
    main()
