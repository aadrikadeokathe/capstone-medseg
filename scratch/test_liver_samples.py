import urllib.request
import os

def check_samples():
    sample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    available = []
    for sid in sample_ids:
        fname = f"liver_{sid}.nii.gz"
        img_url = f"https://huggingface.co/datasets/MedOtter/msd-liver/resolve/main/imagesTr/{fname}"
        lbl_url = f"https://huggingface.co/datasets/MedOtter/msd-liver/resolve/main/labelsTr/{fname}"
        try:
            req1 = urllib.request.Request(img_url, method='HEAD')
            req2 = urllib.request.Request(lbl_url, method='HEAD')
            with urllib.request.urlopen(req1) as r1, urllib.request.urlopen(req2) as r2:
                if r1.status == 200 and r2.status == 200:
                    available.append(fname)
                    print(f"Sample {fname} available.")
        except Exception as e:
            print(f"Sample {fname} failed: {e}")
    print("Available liver samples:", available)

if __name__ == "__main__":
    check_samples()
