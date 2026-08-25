import os
import urllib.request
from huggingface_hub import hf_hub_download

def test_sources():
    s3_url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task03_Liver.tar"
    print(f"Checking S3 URL: {s3_url}")
    try:
        req = urllib.request.Request(s3_url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            print("S3 Status Code:", resp.status)
            print("Content Length:", resp.headers.get("Content-Length"))
    except Exception as e:
        print("S3 Error:", e)

    print("\nChecking Hugging Face repo Novel-BioMedAI/Medical_Segmentation_Decathlon for Task03_Liver.tar...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id="Novel-BioMedAI/Medical_Segmentation_Decathlon", repo_type="dataset")
        liver_files = [f for f in files if "liver" in f.lower() or "task03" in f.lower()]
        print("Matching files in HF repo:", liver_files)
    except Exception as e:
        print("HF Error:", e)

if __name__ == "__main__":
    test_sources()
