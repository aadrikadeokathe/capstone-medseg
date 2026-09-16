"""
src/generate_architecture_diagram.py

Generates a publication-grade System Architecture Diagram (Figure 1)
for the PAUL 2026 research paper.
Visually distinguishes the Frozen Foundation Backbone (76.9%) from the
Trainable Cross-Attention Fusion Adapter and Skip Decoder (23.1%).
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def generate_architecture_diagram(output_png: str = "paper/figures/architecture_diagram.png"):
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    # Color Palette (Publication Grade)
    c_frozen_bg = "#EBF3FA"
    c_frozen_border = "#1E6091"
    c_trainable_bg = "#FDECEF"
    c_trainable_border = "#C9184A"
    c_decoder_bg = "#E8F5E9"
    c_decoder_border = "#2E7D32"
    c_arrow = "#2B2D42"

    # Title & Legend
    ax.text(
        0.5, 8.1,
        "Figure 1: In-Context Medical Segmentation Pipeline with Stacked Cross-Attention Fusion",
        fontsize=14, fontweight="bold", color="#1D3557"
    )

    # Legend Badges
    p1 = patches.FancyBboxPatch((9.5, 7.85), 2.3, 0.45, boxstyle="round,pad=0.08", facecolor=c_frozen_bg, edgecolor=c_frozen_border, linewidth=1.5)
    ax.add_patch(p1)
    ax.text(10.65, 8.05, "Frozen Backbone (76.9%)", ha="center", va="center", fontsize=9, fontweight="bold", color=c_frozen_border)

    p2 = patches.FancyBboxPatch((12.2, 7.85), 2.5, 0.45, boxstyle="round,pad=0.08", facecolor=c_trainable_bg, edgecolor=c_trainable_border, linewidth=1.5)
    ax.add_patch(p2)
    ax.text(13.45, 8.05, "Trainable Adapter (23.1%)", ha="center", va="center", fontsize=9, fontweight="bold", color=c_trainable_border)

    # ==========================================
    # 1. INPUT STAGE
    # ==========================================
    # Query Image Box
    b_q = patches.FancyBboxPatch((0.5, 5.2), 2.2, 1.6, boxstyle="round,pad=0.1", facecolor="#F8F9FA", edgecolor="#495057", linewidth=1.5)
    ax.add_patch(b_q)
    ax.text(1.6, 6.2, "Query Image (Target)", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(1.6, 5.7, "Scan: [B, 1, 128, 128]\n(CT / MRI)", ha="center", va="center", fontsize=8.5, color="#6C757D")

    # Support Image & Mask Box
    b_s = patches.FancyBboxPatch((0.5, 1.2), 2.2, 2.4, boxstyle="round,pad=0.1", facecolor="#F8F9FA", edgecolor="#495057", linewidth=1.5)
    ax.add_patch(b_s)
    ax.text(1.6, 2.9, "Support Set (Context)", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(1.6, 2.2, "Images: [B, S, 1, 128, 128]\nMasks: [B, S, 1, 128, 128]\n(K = 1, 2, 4, 8 Shots)", ha="center", va="center", fontsize=8.5, color="#6C757D")

    # ==========================================
    # 2. FROZEN BACKBONE (UNIVERSEG)
    # ==========================================
    b_backbone = patches.FancyBboxPatch((3.5, 4.2), 2.5, 3.2, boxstyle="round,pad=0.15", facecolor=c_frozen_bg, edgecolor=c_frozen_border, linewidth=2)
    ax.add_patch(b_backbone)
    ax.text(4.75, 7.0, "Frozen Backbone\n(UniverSeg Encoder)", ha="center", va="center", fontsize=10, fontweight="bold", color=c_frozen_border)
    ax.text(4.75, 6.2, "Stage 1: [B, 64, 128, 128]\nStage 2: [B, 64, 64, 64]\nBottleneck: [B, 64, 32, 32]", ha="center", va="center", fontsize=8, color="#333333")
    ax.text(4.75, 4.7, "1,182,785 Parameters\n(requires_grad=False)", ha="center", va="center", fontsize=8, fontweight="bold", color=c_frozen_border)

    # Support Encoder (Shared Weights)
    b_supp_enc = patches.FancyBboxPatch((3.5, 1.2), 2.5, 2.0, boxstyle="round,pad=0.15", facecolor=c_frozen_bg, edgecolor=c_frozen_border, linewidth=1.5, linestyle="--")
    ax.add_patch(b_supp_enc)
    ax.text(4.75, 2.4, "Shared Frozen Backbone\nFeature Extraction", ha="center", va="center", fontsize=9, fontweight="bold", color=c_frozen_border)
    ax.text(4.75, 1.7, "Bottleneck: [B, S, 64, 32, 32]\nDownsampled Masks: [32, 32]", ha="center", va="center", fontsize=7.5, color="#555555")

    # ==========================================
    # 3. STACKED CROSS-ATTENTION FUSION MODULE
    # ==========================================
    b_fusion = patches.FancyBboxPatch((6.8, 1.2), 3.4, 4.4, boxstyle="round,pad=0.2", facecolor=c_trainable_bg, edgecolor=c_trainable_border, linewidth=2.5)
    ax.add_patch(b_fusion)
    ax.text(8.5, 5.2, "Stacked In-Context FusionModule\n(3-Layer Cross-Attention)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_trainable_border)
    ax.text(8.5, 4.5, "100,800 Trainable Parameters", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#7209B7")

    # Subcomponents inside FusionModule
    # Mask Projection
    p_mask = patches.FancyBboxPatch((7.1, 3.4), 2.8, 0.7, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="#E63946", linewidth=1.2)
    ax.add_patch(p_mask)
    ax.text(8.5, 3.75, "1. Mask Projection (1x1 Conv) -> Key/Val Gating", ha="center", va="center", fontsize=8, fontweight="bold")

    # Multihead Attention
    p_mha = patches.FancyBboxPatch((7.1, 2.3), 2.8, 0.85, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="#E63946", linewidth=1.2)
    ax.add_patch(p_mha)
    ax.text(8.5, 2.85, "2. Multi-Head Cross-Attention (4 Heads)", ha="center", va="center", fontsize=8, fontweight="bold")
    ax.text(8.5, 2.5, "Q: Query Tokens | K, V: Conditioned Support", ha="center", va="center", fontsize=7.5, color="#555555")

    # FFN & LayerNorm
    p_ffn = patches.FancyBboxPatch((7.1, 1.4), 2.8, 0.65, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor="#E63946", linewidth=1.2)
    ax.add_patch(p_ffn)
    ax.text(8.5, 1.72, "3. LayerNorm + Residual FFN (64 -> 128 -> 64)", ha="center", va="center", fontsize=8, fontweight="bold")

    # ==========================================
    # 4. TRAINABLE UNET SKIP DECODER
    # ==========================================
    b_decoder = patches.FancyBboxPatch((10.9, 2.2), 2.3, 4.6, boxstyle="round,pad=0.15", facecolor=c_decoder_bg, edgecolor=c_decoder_border, linewidth=2)
    ax.add_patch(b_decoder)
    ax.text(12.05, 6.3, "Trainable UNet Skip Decoder", ha="center", va="center", fontsize=10, fontweight="bold", color=c_decoder_border)
    ax.text(12.05, 5.8, "254,657 Parameters", ha="center", va="center", fontsize=8.5, color="#1B5E20")

    # Decoder stages
    p_up2 = patches.FancyBboxPatch((11.1, 4.7), 1.9, 0.7, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor=c_decoder_border, linewidth=1.2)
    ax.add_patch(p_up2)
    ax.text(12.05, 5.05, "Up2 (32x32 -> 64x64)\n+ Skip from q_f2", ha="center", va="center", fontsize=7.5, fontweight="bold")

    p_up1 = patches.FancyBboxPatch((11.1, 3.6), 1.9, 0.7, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor=c_decoder_border, linewidth=1.2)
    ax.add_patch(p_up1)
    ax.text(12.05, 3.95, "Up1 (64x64 -> 128x128)\n+ Skip from q_f1", ha="center", va="center", fontsize=7.5, fontweight="bold")

    p_head = patches.FancyBboxPatch((11.1, 2.5), 1.9, 0.7, boxstyle="round,pad=0.05", facecolor="#FFFFFF", edgecolor=c_decoder_border, linewidth=1.2)
    ax.add_patch(p_head)
    ax.text(12.05, 2.85, "1x1 Conv Head\nLogits Output", ha="center", va="center", fontsize=7.5, fontweight="bold")

    # ==========================================
    # 5. FINAL OUTPUT
    # ==========================================
    b_out = patches.FancyBboxPatch((13.7, 3.3), 1.1, 2.0, boxstyle="round,pad=0.1", facecolor="#F8F9FA", edgecolor="#495057", linewidth=1.5)
    ax.add_patch(b_out)
    ax.text(14.25, 4.5, "Binary Mask\n[1, 128, 128]", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(14.25, 3.8, "Dice Score\nEvaluation", ha="center", va="center", fontsize=8, color="#555555")

    # ==========================================
    # CONNECTING ARROWS & DATA FLOW
    # ==========================================
    arrow_style = dict(facecolor=c_arrow, edgecolor=c_arrow, width=1.5, headwidth=6, headlength=7)

    # Query -> Backbone
    ax.annotate("", xy=(3.5, 6.0), xytext=(2.7, 6.0), arrowprops=arrow_style)

    # Support -> Support Backbone
    ax.annotate("", xy=(3.5, 2.2), xytext=(2.7, 2.2), arrowprops=arrow_style)

    # Backbone Bottleneck -> FusionModule
    ax.annotate("", xy=(6.8, 5.0), xytext=(6.0, 5.2), arrowprops=arrow_style)
    ax.text(6.4, 5.3, "q_f3 [32x32]", fontsize=7.5, fontweight="bold", ha="center")

    # Support Bottleneck -> FusionModule
    ax.annotate("", xy=(6.8, 2.4), xytext=(6.0, 2.2), arrowprops=arrow_style)
    ax.text(6.4, 2.5, "s_f3 [32x32]", fontsize=7.5, fontweight="bold", ha="center")

    # FusionModule -> Decoder Up2
    ax.annotate("", xy=(10.9, 5.2), xytext=(10.2, 4.0), arrowprops=arrow_style)
    ax.text(10.5, 4.8, "Fused Bottleneck", fontsize=7.5, fontweight="bold", ha="center")

    # Decoder -> Output
    ax.annotate("", xy=(13.7, 4.3), xytext=(13.2, 4.3), arrowprops=arrow_style)

    # Skip Connections from Backbone to Decoder
    # Skip 2 (64x64)
    ax.annotate("", xy=(11.1, 5.2), xytext=(6.0, 6.2),
                arrowprops=dict(arrowstyle="->", color=c_frozen_border, lw=1.5, linestyle="--", connectionstyle="arc3,rad=-0.15"))
    ax.text(8.5, 6.6, "Frozen Skip q_f2 [64x64]", fontsize=7.5, color=c_frozen_border, fontweight="bold", ha="center")

    # Skip 1 (128x128)
    ax.annotate("", xy=(11.1, 4.1), xytext=(6.0, 7.0),
                arrowprops=dict(arrowstyle="->", color=c_frozen_border, lw=1.5, linestyle="--", connectionstyle="arc3,rad=-0.25"))
    ax.text(8.5, 7.3, "Frozen Skip q_f1 [128x128]", fontsize=7.5, color=c_frozen_border, fontweight="bold", ha="center")

    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)

    # Also save to reports
    reports_png = os.path.join(PROJECT_ROOT, "reports", "architecture_diagram.png")
    os.makedirs(os.path.dirname(reports_png), exist_ok=True)
    plt.savefig(reports_png, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SUCCESS] Saved Publication Architecture Diagram to:")
    print(f"  -> {output_png}")
    print(f"  -> {reports_png}")


if __name__ == "__main__":
    generate_architecture_diagram()
