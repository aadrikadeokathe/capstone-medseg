# nnunet_convert.py
"""
Script to convert MSD Task09_Spleen raw dataset into nnU-Net v2 raw dataset format.
Target dataset directory: data/nnunet_raw/Dataset009_Spleen
"""

import os
import glob
import json
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_SPLEEN_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "Task09_Spleen")
NNUNET_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "nnunet_raw")
DATASET_NAME = "Dataset009_Spleen"
TARGET_DIR = os.path.join(NNUNET_RAW_DIR, DATASET_NAME)


def convert_spleen_to_nnunet():
    """
    Converts raw NIfTI files from Task09_Spleen into nnU-Net v2 dataset format.
    """
    images_tr_src = os.path.join(RAW_SPLEEN_DIR, "imagesTr")
    labels_tr_src = os.path.join(RAW_SPLEEN_DIR, "labelsTr")

    if not os.path.exists(images_tr_src) or not os.path.exists(labels_tr_src):
        raise FileNotFoundError(
            f"Source directories missing. Ensure Task09_Spleen is downloaded in {RAW_SPLEEN_DIR}"
        )

    images_tr_dst = os.path.join(TARGET_DIR, "imagesTr")
    labels_tr_dst = os.path.join(TARGET_DIR, "labelsTr")

    os.makedirs(images_tr_dst, exist_ok=True)
    os.makedirs(labels_tr_dst, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(images_tr_src, "*.nii.gz")))
    print(f"[INFO] Found {len(image_files)} raw training image volumes.")

    num_cases = 0
    for img_path in image_files:
        base_name = os.path.basename(img_path) # e.g. spleen_2.nii.gz
        case_id = base_name.replace(".nii.gz", "") # e.g. spleen_2

        label_path = os.path.join(labels_tr_src, base_name)
        if not os.path.isfile(label_path):
            print(f"[WARN] Label file for {case_id} not found at {label_path}, skipping.")
            continue

        # nnU-Net v2 image naming: <case_identifier>_0000.nii.gz
        dst_img_name = f"{case_id}_0000.nii.gz"
        dst_img_path = os.path.join(images_tr_dst, dst_img_name)

        # nnU-Net v2 label naming: <case_identifier>.nii.gz
        dst_lbl_name = f"{case_id}.nii.gz"
        dst_lbl_path = os.path.join(labels_tr_dst, dst_lbl_name)

        # Copy files (or symlink/hardlink to save disk space if available)
        if not os.path.exists(dst_img_path):
            shutil.copyfile(img_path, dst_img_path)
        if not os.path.exists(dst_lbl_path):
            shutil.copyfile(label_path, dst_lbl_path)

        num_cases += 1

    print(f"[INFO] Converted {num_cases} volumes to nnU-Net format at {TARGET_DIR}")

    # Generate nnU-Net v2 dataset.json
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0,
            "spleen": 1
        },
        "numTraining": num_cases,
        "file_ending": ".nii.gz"
    }

    json_path = os.path.join(TARGET_DIR, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"[INFO] Created dataset.json at {json_path}")


if __name__ == "__main__":
    convert_spleen_to_nnunet()
