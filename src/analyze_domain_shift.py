"""
src/analyze_domain_shift.py

Quantifies and plots the cross-modality domain shift between CT (Spleen, Liver)
and MRI (Heart, Brain Tumour). Generates Figure 1 (Domain Shift Characterization)
for the PAUL 2026 research paper.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_split import (
    get_spleen_splits,
    get_liver_splits,
    get_heart_splits,
    get_braintumour_splits,
)


def generate_domain_shift_plot(output_png: str = "reports/domain_shift_analysis.png"):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), dpi=300)
    plt.subplots_adjust(wspace=0.25, hspace=0.35)

    datasets = [
        ("spleen", "Spleen (CT)", "#2b5c8f", get_spleen_splits),
        ("liver", "Liver (CT)", "#388e3c", get_liver_splits),
        ("heart", "Heart (Cine MRI)", "#d32f2f", get_heart_splits),
        ("braintumour", "Brain Tumour (FLAIR MRI)", "#7b1fa2", lambda **kw: get_braintumour_splits(channel_idx=0, **kw)),
    ]

    for col_idx, (ds_name, display_name, color, split_fn) in enumerate(datasets):
        try:
            _, _, test_ds, _ = split_fn(num_support=2, target_size=(128, 128))
        except Exception as e:
            print(f"[WARNING] Could not load {ds_name}: {e}")
            continue

        # Find sample with clear foreground
        chosen_idx = 0
        for idx in range(len(test_ds)):
            q_img, q_mask, _, _ = test_ds[idx]
            if q_mask.sum().item() > 200:
                chosen_idx = idx
                break

        q_img, q_mask, _, _ = test_ds[chosen_idx]
        img_np = q_img.squeeze().numpy()
        mask_np = q_mask.squeeze().numpy() > 0

        # Row 1: Slice preview + Mask overlay
        axes[0, col_idx].imshow(img_np, cmap="gray")
        axes[0, col_idx].contour(mask_np, levels=[0.5], colors=[color], linewidths=2.0)
        axes[0, col_idx].set_title(display_name, fontsize=12, fontweight="bold", pad=8)
        axes[0, col_idx].set_xticks([])
        axes[0, col_idx].set_yticks([])

        # Collect pixel intensities across 20 sample slices for distribution curve
        intensities_fg = []
        intensities_bg = []
        for i in range(min(20, len(test_ds))):
            s_img, s_msk, _, _ = test_ds[i]
            s_img_arr = s_img.squeeze().numpy()
            s_msk_arr = s_msk.squeeze().numpy() > 0
            if np.count_nonzero(s_msk_arr) > 0:
                intensities_fg.extend(s_img_arr[s_msk_arr].tolist())
                intensities_bg.extend(s_img_arr[~s_msk_arr].tolist())

        # Row 2: Density / Histogram Comparison
        axes[1, col_idx].hist(
            intensities_bg, bins=40, density=True, alpha=0.45, color="gray", label="Background"
        )
        axes[1, col_idx].hist(
            intensities_fg, bins=40, density=True, alpha=0.75, color=color, label="Organ / Lesion"
        )
        axes[1, col_idx].set_xlabel("Normalized Intensity", fontsize=10)
        axes[1, col_idx].set_ylabel("Probability Density", fontsize=10)
        axes[1, col_idx].grid(True, linestyle="--", alpha=0.4)
        axes[1, col_idx].legend(loc="upper right", fontsize=8)
        axes[1, col_idx].set_xlim([0.0, 1.0])

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)

    # Save to logs and paper/figures
    logs_png = os.path.join(PROJECT_ROOT, "logs", "domain_shift_analysis.png")
    paper_png = os.path.join(PROJECT_ROOT, "paper", "figures", "domain_shift_analysis.png")
    os.makedirs(os.path.dirname(paper_png), exist_ok=True)
    plt.savefig(logs_png, bbox_inches="tight", dpi=300)
    plt.savefig(paper_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved Domain Shift Characterization plot to:")
    print(f"  -> {output_png}")
    print(f"  -> {logs_png}")
    print(f"  -> {paper_png}")


if __name__ == "__main__":
    generate_domain_shift_plot()
