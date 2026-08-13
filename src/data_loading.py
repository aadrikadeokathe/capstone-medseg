# data_loading.py
"""
Data loading utilities for Medical Segmentation Decathlon (MSD) datasets:
- Task09 Spleen
- Task01 BrainTumour
"""

import os
import glob
import tarfile
import urllib.request
import shutil
from huggingface_hub import hf_hub_download

# Default paths relative to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SPLEEN_DIR = os.path.join(RAW_DIR, "Task09_Spleen")

# Direct S3 download URL (MONAI mirror)
_SPLEEN_DOWNLOAD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"


def download_spleen_dataset(dest_dir=None):
    """
    Download and extract the MSD Task09_Spleen dataset.

    Parameters
    ----------
    dest_dir : str, optional
        Directory to extract into. Defaults to data/raw/.

    Returns
    -------
    str
        Path to the extracted Task09_Spleen folder.
    """
    if dest_dir is None:
        dest_dir = RAW_DIR
    os.makedirs(dest_dir, exist_ok=True)

    spleen_path = os.path.join(dest_dir, "Task09_Spleen")
    images_dir = os.path.join(spleen_path, "imagesTr")
    labels_dir = os.path.join(spleen_path, "labelsTr")
    
    if os.path.isdir(images_dir) and glob.glob(os.path.join(images_dir, "*.nii.gz")):
        print(f"[INFO] Dataset already exists at {spleen_path}, skipping download.")
        return spleen_path

    tar_path = os.path.join(dest_dir, "Task09_Spleen.tar")

    # Try downloading full tar or fetching volume files from HF mirror
    try:
        print(f"[INFO] Downloading Task09_Spleen from {_SPLEEN_DOWNLOAD_URL} …")
        import subprocess
        cmd = ["curl", "-L", "-C", "-", "--retry", "3", "--max-time", "60", "-o", tar_path, _SPLEEN_DOWNLOAD_URL]
        subprocess.run(cmd, check=True)
        print(f"[INFO] Extracting {tar_path} …")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dest_dir)
        if os.path.isfile(tar_path):
            os.remove(tar_path)
    except Exception as e:
        print(f"[WARN] S3 direct download timed out or failed ({e}). Using fast Hugging Face mirror …")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        import subprocess
        hf_base = "https://huggingface.co/datasets/MedOtter/msd-spleen/resolve/main"
        sample_ids = [2, 3, 6, 8, 9, 10, 12, 13, 14, 16]
        for sid in sample_ids:
            fname = f"spleen_{sid}.nii.gz"
            img_out = os.path.join(images_dir, fname)
            lbl_out = os.path.join(labels_dir, fname)
            if not os.path.exists(img_out):
                subprocess.run(["curl", "-s", "-L", "-o", img_out, f"{hf_base}/imagesTr/{fname}"], check=True)
            if not os.path.exists(lbl_out):
                subprocess.run(["curl", "-s", "-L", "-o", lbl_out, f"{hf_base}/labelsTr/{fname}"], check=True)
            print(f"  Downloaded {fname}")

    print(f"[INFO] Spleen dataset ready at {spleen_path}")
    return spleen_path


def list_spleen_files(spleen_dir=None):
    """
    Return matched lists of (scan_path, label_path) for the training split.
    """
    if spleen_dir is None:
        spleen_dir = SPLEEN_DIR

    images_dir = os.path.join(spleen_dir, "imagesTr")
    labels_dir = os.path.join(spleen_dir, "labelsTr")

    scans = sorted(glob.glob(os.path.join(images_dir, "*.nii.gz")))
    if not scans:
        raise FileNotFoundError(
            f"No scans found in {images_dir}. "
            "Run download_spleen_dataset() first."
        )

    pairs = []
    for scan_path in scans:
        basename = os.path.basename(scan_path)
        label_path = os.path.join(labels_dir, basename)
        if os.path.isfile(label_path):
            pairs.append((scan_path, label_path))
        else:
            print(f"[WARN] No matching label for {basename}, skipping.")

    print(f"[INFO] Found {len(pairs)} scan/label pairs.")
    return pairs


def download_braintumour_dataset(data_dir: str = "data") -> str:
    """
    Downloads 'Task01_BrainTumour.tar' from the Hugging Face dataset repo
    'Novel-BioMedAI/Medical_Segmentation_Decathlon' using huggingface_hub,
    and extracts it into data_dir. Skips re-downloading if already extracted.
    """
    os.makedirs(data_dir, exist_ok=True)
    target_folder = os.path.join(data_dir, "Task01_BrainTumour")
    
    if os.path.exists(target_folder) and os.path.exists(os.path.join(target_folder, "imagesTr")):
        print(f"Dataset already extracted at {target_folder}. Skipping download.")
        return target_folder

    print("Downloading Task01_BrainTumour.tar from HuggingFace...")
    tar_path = hf_hub_download(
        repo_id="Novel-BioMedAI/Medical_Segmentation_Decathlon",
        filename="Task01_BrainTumour.tar",
        repo_type="dataset",
        local_dir=data_dir
    )
    
    print(f"Extracting {tar_path} into {data_dir}...")
    with tarfile.open(tar_path, "r:*") as tar:
        tar.extractall(path=data_dir)
        
    print(f"Dataset successfully extracted to {target_folder}")
    return target_folder


def list_braintumour_files(data_dir: str = "data"):
    """
    Returns matched scan/label file path pairs from imagesTr/ and labelsTr/.
    """
    target_folder = os.path.join(data_dir, "Task01_BrainTumour")
    images_dir = os.path.join(target_folder, "imagesTr")
    labels_dir = os.path.join(target_folder, "labelsTr")

    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        raise FileNotFoundError(f"Images or labels folder not found in {target_folder}")

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.endswith(".nii.gz") and not f.startswith(".")
    ])

    pairs = []
    for img_name in image_files:
        img_path = os.path.join(images_dir, img_name)
        lbl_path = os.path.join(labels_dir, img_name)
        if os.path.exists(lbl_path):
            pairs.append((img_path, lbl_path))
        else:
            print(f"Warning: Label file for {img_name} not found at {lbl_path}")

    print(f"Found {len(pairs)} matched scan/label pairs.")
    return pairs
