import os
import glob
import subprocess
import numpy as np
import nibabel as nib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
LIVER_DIR = os.path.join(RAW_DIR, "Task03_Liver")
IMAGES_DIR = os.path.join(LIVER_DIR, "imagesTr")
LABELS_DIR = os.path.join(LIVER_DIR, "labelsTr")

def download_liver_samples():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(LABELS_DIR, exist_ok=True)

    # Hugging Face MedOtter mirror for MSD liver sample volumes
    hf_base = "https://huggingface.co/datasets/MedOtter/msd-liver/resolve/main"
    sample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    for sid in sample_ids:
        fname = f"liver_{sid}.nii.gz"
        img_out = os.path.join(IMAGES_DIR, fname)
        lbl_out = os.path.join(LABELS_DIR, fname)
        if not os.path.exists(img_out):
            print(f"Downloading {fname} image...")
            subprocess.run(["curl", "-s", "-L", "-o", img_out, f"{hf_base}/imagesTr/{fname}"], check=True)
        if not os.path.exists(lbl_out):
            print(f"Downloading {fname} label...")
            subprocess.run(["curl", "-s", "-L", "-o", lbl_out, f"{hf_base}/labelsTr/{fname}"], check=True)

    print("Downloaded Liver scan/label pairs:")
    scans = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.nii.gz")))
    print(f"Total scans: {len(scans)}")

if __name__ == "__main__":
    download_liver_samples()
