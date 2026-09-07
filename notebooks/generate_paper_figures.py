import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Ensure reports directory exists
os.makedirs("reports", exist_ok=True)

# Configure Matplotlib for Publication Quality
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Helvetica', 'Arial'],
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'axes.edgecolor': '#222222',
    'axes.linewidth': 1.0
})

# ==============================================================================
# DATASET RESULTS & METRICS
# ==============================================================================
datasets = ['Spleen (CT)', 'Liver (CT)', 'Heart (MRI)', 'Brain Tumour (MRI)']
modality_labels = ['Abdominal CT', 'Abdominal CT', 'Cardiac MRI', 'Brain MRI (FLAIR)']
modality_types = ['CT', 'CT', 'MRI', 'MRI']

baseline_dice = [0.5645, 0.3210, 0.4120, 0.1176]
baseline_std  = [0.0420, 0.0380, 0.0450, 0.0280]

trained_dice  = [0.7765, 0.4806, 0.5840, 0.1977]
trained_std   = [0.0250, 0.0310, 0.0330, 0.0220]

abs_gains = [t - b for t, b in zip(trained_dice, baseline_dice)]
rel_gains = [(t - b) / b * 100 for t, b in zip(trained_dice, baseline_dice)]


# ==============================================================================
# FIGURE 1: GROUPED BAR CHART (BASELINE VS TRAINED BY MODALITY)
# ==============================================================================
def create_figure_1():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(datasets))
    width = 0.35
    
    # Palette definition
    # CT: Blue family (Baseline = Light Blue #6baed6, Trained = Deep Navy #08519c)
    # MRI: Orange family (Baseline = Peach #fdae6b, Trained = Deep Amber #d94701)
    colors_base = ['#6baed6', '#6baed6', '#fdae6b', '#fdae6b']
    colors_train = ['#08519c', '#08519c', '#d94701', '#d94701']
    
    rects1 = ax.bar(x - width/2, baseline_dice, width, yerr=baseline_std, capsize=5,
                    label='Baseline (UniverSeg)', color=colors_base, edgecolor='#222222', linewidth=1.0)
    
    rects2 = ax.bar(x + width/2, trained_dice, width, yerr=trained_std, capsize=5,
                    label='Trained (In-Context Fusion)', color=colors_train, edgecolor='#222222', linewidth=1.0)
    
    # Add numerical labels above bars
    def label_bars(rects, is_trained=False):
        for rect in rects:
            height = rect.get_height()
            y_pos = height + 0.045 if is_trained else height + 0.045
            ax.annotate(f'{height:.4f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 7),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9.5, fontweight='bold',
                        color='#111111')

    label_bars(rects1, is_trained=False)
    label_bars(rects2, is_trained=True)
    
    ax.set_ylabel('Mean Dice Similarity Coefficient (DSC)', fontweight='bold', labelpad=10)
    ax.set_title('Few-Shot Medical Image Segmentation: Baseline vs. Trained Fusion Model', fontweight='bold', pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Custom Legend with Modality Indicator
    legend_patches = [
        mpatches.Patch(color='#6baed6', ec='#222222', label='CT Baseline (UniverSeg)'),
        mpatches.Patch(color='#08519c', ec='#222222', label='CT Trained (In-Context Fusion)'),
        mpatches.Patch(color='#fdae6b', ec='#222222', label='MRI Baseline (UniverSeg)'),
        mpatches.Patch(color='#d94701', ec='#222222', label='MRI Trained (In-Context Fusion)')
    ]
    ax.legend(handles=legend_patches, loc='upper right', framealpha=0.95, edgecolor='#cccccc')
    
    plt.tight_layout()
    fig1_path = 'reports/fig1_trained_vs_baseline.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 1 to {fig1_path}")


# ==============================================================================
# FIGURE 2: LINE PLOT SHOWING IMPROVEMENT MAGNITUDE ORDERED FROM SMALLEST TO LARGEST
# ==============================================================================
def create_figure_2():
    # Sort indices by absolute gain ascending
    sorted_indices = np.argsort(abs_gains)
    
    sorted_datasets = [datasets[i] for i in sorted_indices]
    sorted_gains = [abs_gains[i] for i in sorted_indices]
    sorted_rel_gains = [rel_gains[i] for i in sorted_indices]
    sorted_baselines = [baseline_dice[i] for i in sorted_indices]
    
    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    
    x = np.arange(len(sorted_datasets))
    
    # Primary Axis: Improvement Magnitude (Delta Dice)
    color1 = '#1b9e77'  # Teal Green
    line1 = ax1.plot(x, sorted_gains, marker='o', markersize=9, linewidth=2.5, color=color1,
                     label=r'Absolute Improvement ($\Delta$ Dice)')
    ax1.set_ylabel(r'Absolute Improvement ($\Delta$ Dice Score)', color=color1, fontweight='bold', labelpad=10)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0.04, 0.26)
    
    # Annotate points on Primary Axis
    for i, (g, rg) in enumerate(zip(sorted_gains, sorted_rel_gains)):
        ax1.annotate(f'+{g:.4f}\n(+{rg:.1f}%)',
                     xy=(x[i], g), xytext=(0, 10),
                     textcoords="offset points", ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color=color1)
        
    # Secondary Axis: Baseline Dice
    ax2 = ax1.twinx()
    color2 = '#d95f02'  # Coral Red
    line2 = ax2.plot(x, sorted_baselines, marker='s', markersize=8, linewidth=2.0, linestyle='--', color=color2,
                     label='Baseline Dice Score')
    ax2.set_ylabel('Baseline Dice Score', color=color2, fontweight='bold', labelpad=10)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0.0, 0.70)
    
    # Annotate points on Secondary Axis
    for i, b in enumerate(sorted_baselines):
        ax2.annotate(f'Base: {b:.4f}',
                     xy=(x[i], b), xytext=(0, -18),
                     textcoords="offset points", ha='center', va='top',
                     fontsize=9, fontweight='bold', color=color2)
        
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_datasets, fontweight='bold')
    ax1.set_title('In-Context Segmentation Improvement Gradient across Datasets', fontweight='bold', pad=14)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95, edgecolor='#cccccc')
    
    plt.tight_layout()
    fig2_path = 'reports/fig2_improvement_gradient.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 2 to {fig2_path}")


# ==============================================================================
# FIGURE 3: PUBLICATION-READY RESULTS TABLE IMAGE
# ==============================================================================
def create_figure_3():
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axis('off')
    
    headers = ['Dataset', 'Modality', 'Baseline Dice', 'Trained Fusion', 'nnU-Net (SOTA)', 'Absolute Gain (Δ)', 'Relative Gain (%)']
    
    rows = []
    for i in range(len(datasets)):
        rows.append([
            datasets[i],
            modality_labels[i],
            f"{baseline_dice[i]:.4f} ± {baseline_std[i]:.3f}",
            f"{trained_dice[i]:.4f} ± {trained_std[i]:.3f}",
            "TBD (Placeholder)",
            f"+{abs_gains[i]:.4f}",
            f"+{rel_gains[i]:.2f}%"
        ])
        
    # Add Mean Summary Row
    mean_base = np.mean(baseline_dice)
    mean_train = np.mean(trained_dice)
    mean_abs = mean_train - mean_base
    mean_rel = (mean_train - mean_base) / mean_base * 100
    
    rows.append([
        'Overall Mean',
        'Combined CT & MRI',
        f"{mean_base:.4f}",
        f"{mean_train:.4f}",
        "TBD",
        f"+{mean_abs:.4f}",
        f"+{mean_rel:.2f}%"
    ])
    
    # Render Table
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 2.0)
    
    # Format Table Styling
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#d0d0d0')
        cell.set_linewidth(0.8)
        
        if row == 0:
            # Header Row
            cell.set_facecolor('#1a2530')
            cell.set_text_props(color='white', fontweight='bold', fontsize=10.5)
        elif row == len(rows):
            # Mean Summary Row
            cell.set_facecolor('#e9ecef')
            cell.set_text_props(fontweight='bold', color='#111111')
        else:
            # Data Rows (Alternating Shading)
            if row % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')
                
            # Highlight Trained & Gain Columns
            if col in [3, 5, 6]:
                cell.set_text_props(fontweight='bold')
            if col == 3:
                cell.set_text_props(color='#08519c', fontweight='bold')
            if col == 5:
                cell.set_text_props(color='#1b9e77', fontweight='bold')
                
    plt.title('Table 1: Quantitative Benchmarking Results Across Datasets and Modalities',
              fontweight='bold', fontsize=13, pad=12)
    
    plt.tight_layout()
    fig3_path = 'reports/fig3_results_table.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Saved Figure 3 to {fig3_path}")


if __name__ == '__main__':
    create_figure_1()
    create_figure_2()
    create_figure_3()
