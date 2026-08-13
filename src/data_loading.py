import os
import tarfile
from huggingface_hub import hf_hub_download


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
