import urllib.request

def check_hf_medotter():
    url = "https://huggingface.co/datasets/MedOtter/msd-liver/resolve/main/imagesTr/liver_0.nii.gz"
    print("Checking MedOtter msd-liver URL:", url)
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            print("Status:", resp.status)
            print("Content-Length:", resp.headers.get("Content-Length"))
    except Exception as e:
        print("Error:", e)

    # Let's also test another sample id if liver_0 doesn't exist
    url2 = "https://huggingface.co/datasets/MedOtter/msd-liver/resolve/main/imagesTr/liver_1.nii.gz"
    print("Checking liver_1:", url2)
    try:
        req = urllib.request.Request(url2, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            print("Status:", resp.status)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    check_hf_medotter()
