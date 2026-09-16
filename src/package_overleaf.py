"""
src/package_overleaf.py

Packages all paper assets (main.tex, references.bib, and all 300-DPI figures)
into a single ready-to-upload ZIP file for Overleaf: paper_overleaf_bundle.zip.
"""

import os
import zipfile

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
OUTPUT_ZIP = os.path.join(PROJECT_ROOT, "paper_overleaf_bundle.zip")


def package_paper():
    print("=" * 60)
    print(" Packaging Overleaf Bundle for PAUL 2026 Submission")
    print("=" * 60)

    files_to_zip = []

    # Main tex and bib
    for f in ["main.tex", "references.bib"]:
        fp = os.path.join(PAPER_DIR, f)
        if os.path.exists(fp):
            files_to_zip.append((fp, f))
            print(f"  + Added: {f}")

    # Figures
    fig_dir = os.path.join(PAPER_DIR, "figures")
    if os.path.exists(fig_dir):
        for fig_name in os.listdir(fig_dir):
            if fig_name.endswith((".png", ".jpg", ".pdf")):
                fp = os.path.join(fig_dir, fig_name)
                arc_name = os.path.join("figures", fig_name)
                files_to_zip.append((fp, arc_name))
                print(f"  + Added figure: {arc_name}")

    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path, arc_name in files_to_zip:
            zf.write(file_path, arc_name)

    print("------------------------------------------------------------")
    print(f"[SUCCESS] Created Overleaf Zip Package at:")
    print(f"  -> {OUTPUT_ZIP}")
    print(f"  -> Size: {os.path.getsize(OUTPUT_ZIP) / (1024**2):.2f} MB")
    print("=" * 60)


if __name__ == "__main__":
    package_paper()
