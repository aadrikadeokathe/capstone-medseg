"""
src/plot_zeroshot_transfer.py
==============================
Generates publication-quality comparison bar chart for:
"Cross-Domain Zero-Shot Transfer: Standard vs. Episodic Meta-Learning"
Fixes legend overlap, badge clipping, and formatting for Springer LNCS/CCIS papers.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def generate_zeroshot_plot(output_paths=None):
    if output_paths is None:
        output_paths = [
            "paper/figures/zeroshot_transfer_comparison.png",
            "logs/zeroshot_transfer_comparison.png",
        ]

    # Data from logs/episodic_zeroshot_results.txt
    datasets = ["Spleen", "Liver", "Heart", "Brain Tumour", "Average"]
    std_zero_shot = [1.47, 1.04, 10.68, 2.78, 3.99]
    episodic_zero_shot = [15.65, 62.48, 42.29, 39.52, 39.98]
    absolute_gains = [14.18, 61.44, 31.61, 36.74, 35.99]

    # Plot styling
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    x = np.arange(len(datasets))
    width = 0.35

    # Color palette matching manuscript
    color_std = "#d9534f"       # Coral red for naive standard
    color_epi = "#208b8b"       # Deep teal for episodic meta-learning
    badge_bg = "#e8f8f5"        # Soft mint for badge fill
    badge_border = "#1abc9c"    # Green border for badge

    # Draw bars
    rects1 = ax.bar(x - width/2, std_zero_shot, width, label="Standard Per-Dataset Training (Zero-Shot)",
                    color=color_std, edgecolor="none", alpha=0.92, zorder=3)
    rects2 = ax.bar(x + width/2, episodic_zero_shot, width, label="Episodic Meta-Learning (Zero-Shot)",
                    color=color_epi, edgecolor="none", alpha=0.95, zorder=3)

    # Grid & Limits
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", alpha=0.4, color="#cccccc", zorder=0)
    ax.set_ylim(0, 78)  # Generous headroom to prevent badge/legend collision
    ax.set_ylabel("Zero-Shot Dice Similarity Coefficient (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title("Cross-Domain Zero-Shot Transfer: Standard vs. Episodic Meta-Learning",
                 fontsize=12.5, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10.5, fontweight="bold")

    # Annotate Red Bars (Standard)
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#a93226")

    # Annotate Teal Bars (Episodic) and Gains
    for idx, rect in enumerate(rects2):
        h = rect.get_height()
        gain = absolute_gains[idx]

        # Value label on top of bar
        ax.annotate(f"{h:.2f}%",
                    xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom",
                    fontsize=8.5, fontweight="bold", color="#114b4b")

        # Absolute Gain Pill / Badge above bar
        badge_y = h + 6.5
        ax.text(rect.get_x() + rect.get_width()/2, badge_y,
                f"+{gain:.2f} pp",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#0e6251",
                bbox=dict(boxstyle="round,pad=0.28", fc=badge_bg, ec=badge_border, lw=0.9, alpha=0.95))

    # Move legend to upper right to cleanly avoid Liver bar and badge
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc",
              framealpha=0.95, fontsize=9.5, borderpad=0.6)

    # Spine aesthetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    plt.tight_layout()

    # Save to all requested targets
    for p in output_paths:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"[SUCCESS] Saved figure to: {p}")

    plt.close(fig)

if __name__ == "__main__":
    generate_zeroshot_plot()
