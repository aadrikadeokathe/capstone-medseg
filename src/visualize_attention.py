"""
src/visualize_attention.py

Extracts and visualizes cross-attention affinity maps from the In-Context FusionModule.
Generates Figure 3 (Mechanism Analysis) for the PAUL 2026 research paper.
Demonstrates that query foreground tokens dynamically attend to support mask tokens.
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
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
from src.models.in_context_model import InContextSegmentationModel


def generate_attention_figure(
    checkpoint_path: str = "models/checkpoints/best_fusion_episodic.pt",
    output_png: str = "reports/attention_heatmaps.png",
    device: torch.device = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = InContextSegmentationModel(in_channels=1, num_layers=3).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[INFO] Loaded trained checkpoint from {checkpoint_path}")
    else:
        print(f"[WARNING] Checkpoint not found at {checkpoint_path}. Using initialization.")

    model.eval()

    datasets = [
        ("spleen", "Spleen (CT)", get_spleen_splits),
        ("liver", "Liver (CT)", get_liver_splits),
        ("heart", "Heart (MRI)", get_heart_splits),
        ("braintumour", "Brain Tumour (MRI)", lambda **kw: get_braintumour_splits(channel_idx=0, **kw)),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(14, 12), dpi=300)
    plt.subplots_adjust(wspace=0.1, hspace=0.2)

    col_titles = [
        "Query Image (Target)",
        "Support Image (Reference)",
        "Query Cross-Attention Focus",
        "Support Feature Alignment"
    ]

    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=12, fontweight="bold", pad=10)

    for row_idx, (ds_name, display_name, split_fn) in enumerate(datasets):
        try:
            _, _, test_ds, _ = split_fn(num_support=2, target_size=(128, 128))
        except Exception as e:
            print(f"[WARNING] Could not load {ds_name}: {e}")
            continue

        # Pick a slice with good foreground
        chosen_idx = 0
        for idx in range(len(test_ds)):
            q_img, q_mask, s_imgs, s_masks = test_ds[idx]
            if q_mask.sum().item() > 200:
                chosen_idx = idx
                break

        q_img, q_mask, s_imgs, s_masks = test_ds[chosen_idx]
        q_img_t = q_img.unsqueeze(0).to(device)
        s_imgs_t = s_imgs.unsqueeze(0).to(device)
        s_masks_t = s_masks.unsqueeze(0).to(device)

        with torch.no_grad():
            logits, attns = model(q_img_t, s_imgs_t, s_masks_t, return_attention=True)

        # attns is list of 3 layers: each [1, 1024, 2048]
        # Layer 3 (deepest semantic layer) attention map
        last_attn = attns[-1].squeeze(0).cpu().numpy()  # [1024, 2048]

        # Downsampled query mask to 32x32 to identify foreground query tokens
        q_mask_32 = F.interpolate(q_mask.unsqueeze(0), size=(32, 32), mode="nearest").squeeze().numpy() > 0
        fg_indices = np.where(q_mask_32.flatten())[0]

        if len(fg_indices) > 0:
            # Mean attention from foreground query tokens to support tokens
            query_to_supp = last_attn[fg_indices, :].mean(axis=0)  # [2048]
            # Mean attention received by query tokens from support
            supp_to_query = last_attn[:, :1024].mean(axis=1)       # [1024]
        else:
            query_to_supp = last_attn.mean(axis=0)
            supp_to_query = last_attn.mean(axis=1)

        # Reshape support attention (Slice 0: first 1024 tokens)
        supp_attn_map = query_to_supp[:1024].reshape(32, 32)
        supp_attn_128 = F.interpolate(
            torch.tensor(supp_attn_map).unsqueeze(0).unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # Reshape query attention map
        query_attn_map = supp_to_query.reshape(32, 32)
        query_attn_128 = F.interpolate(
            torch.tensor(query_attn_map).unsqueeze(0).unsqueeze(0),
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        # Normalize heatmaps
        supp_attn_128 = (supp_attn_128 - supp_attn_128.min()) / (supp_attn_128.max() - supp_attn_128.min() + 1e-8)
        query_attn_128 = (query_attn_128 - query_attn_128.min()) / (query_attn_128.max() - query_attn_128.min() + 1e-8)

        raw_q = q_img.squeeze().numpy()
        raw_s = s_imgs[0].squeeze().numpy()
        gt_q = q_mask.squeeze().numpy() > 0
        gt_s = s_masks[0].squeeze().numpy() > 0

        # Col 0: Query Image + GT Contour
        axes[row_idx, 0].imshow(raw_q, cmap="gray")
        axes[row_idx, 0].contour(gt_q, levels=[0.5], colors=["#00FF66"], linewidths=1.5)
        axes[row_idx, 0].set_ylabel(display_name, fontsize=11, fontweight="bold", labelpad=8)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])

        # Col 1: Support Image + Support Mask Contour
        axes[row_idx, 1].imshow(raw_s, cmap="gray")
        axes[row_idx, 1].contour(gt_s, levels=[0.5], colors=["#FFCC00"], linewidths=1.5)
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])

        # Col 2: Query Cross-Attention Focus
        axes[row_idx, 2].imshow(raw_q, cmap="gray")
        axes[row_idx, 2].imshow(query_attn_128, cmap="jet", alpha=0.55)
        axes[row_idx, 2].contour(gt_q, levels=[0.5], colors=["white"], linewidths=1.0, linestyles="dashed")
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])

        # Col 3: Support Feature Alignment Heatmap
        axes[row_idx, 3].imshow(raw_s, cmap="gray")
        axes[row_idx, 3].imshow(supp_attn_128, cmap="plasma", alpha=0.55)
        axes[row_idx, 3].contour(gt_s, levels=[0.5], colors=["white"], linewidths=1.0, linestyles="dashed")
        axes[row_idx, 3].set_xticks([])
        axes[row_idx, 3].set_yticks([])

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)

    # Copy to logs and paper/figures
    logs_png = os.path.join(PROJECT_ROOT, "logs", "attention_heatmaps.png")
    paper_png = os.path.join(PROJECT_ROOT, "paper", "figures", "attention_heatmaps.png")
    os.makedirs(os.path.dirname(paper_png), exist_ok=True)
    plt.savefig(logs_png, bbox_inches="tight", dpi=300)
    plt.savefig(paper_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved Mechanism Analysis attention heatmaps to:")
    print(f"  -> {output_png}")
    print(f"  -> {logs_png}")
    print(f"  -> {paper_png}")


if __name__ == "__main__":
    generate_attention_figure()
