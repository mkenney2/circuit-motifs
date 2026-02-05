"""Figure 4: Task-type cosine similarity heatmap with hierarchical clustering.

Shows pairwise cosine similarity between task categories' SP-normalized profiles.
Clustering reorders rows/columns to reveal structure: reasoning-safety cluster,
code-uncategorized cluster, creative as most distant.

Usage:
    python figures/generate_fig4_task_heatmap.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from scipy.spatial.distance import squareform

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "analysis_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig4_task_heatmap.png"

# Display names
TASK_DISPLAY = {
    "arithmetic": "Arithmetic",
    "code": "Code",
    "creative": "Creative",
    "factual_recall": "Factual Recall",
    "multihop": "Multi-hop",
    "multilingual": "Multilingual",
    "reasoning": "Reasoning",
    "safety": "Safety",
    "uncategorized": "Uncategorized",
}

# Category counts (for labels)
TASK_COLORS = {
    "factual_recall": "#2166ac",
    "multihop": "#4393c3",
    "arithmetic": "#d6604d",
    "creative": "#f4a582",
    "multilingual": "#92c5de",
    "safety": "#b2182b",
    "reasoning": "#762a83",
    "code": "#1b7837",
    "uncategorized": "#878787",
}


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_figure():
    data = load_data()
    sim_data = data["similarity_matrix"]
    task_names = sim_data["task_names"]
    sim_matrix = np.array(sim_data["matrix"])

    # Count graphs per category
    cat_counts = {}
    for g in data["graphs"]:
        cat_counts[g["category"]] = cat_counts.get(g["category"], 0) + 1

    # Hierarchical clustering on cosine distance
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0)  # ensure diagonal is exactly 0
    dist_condensed = squareform(dist_matrix)
    Z = linkage(dist_condensed, method="average")
    order = leaves_list(Z)

    # Reorder matrix and labels
    reordered = sim_matrix[np.ix_(order, order)]
    reordered_names = [task_names[i] for i in order]

    n = len(reordered_names)

    # --- Figure: heatmap + dendrogram ---
    fig = plt.figure(figsize=(11, 10), facecolor="white")

    # Layout: dendrogram on top, heatmap below — extra space at top for subtitle
    ax_dendro = fig.add_axes([0.15, 0.74, 0.68, 0.12])   # top (shorter)
    ax_heat = fig.add_axes([0.15, 0.08, 0.68, 0.63])     # main (shifted down)
    ax_cbar = fig.add_axes([0.86, 0.08, 0.025, 0.63])    # colorbar

    # --- Dendrogram ---
    dendro = dendrogram(
        Z, ax=ax_dendro, labels=[TASK_DISPLAY.get(task_names[i], task_names[i])
                                  for i in range(len(task_names))],
        color_threshold=0.05,
        above_threshold_color="#888888",
        leaf_rotation=0, leaf_font_size=0,  # hide labels (shown on heatmap)
    )
    ax_dendro.set_ylabel("Cosine distance", fontsize=9)
    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)
    ax_dendro.spines["bottom"].set_visible(False)
    ax_dendro.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_dendro.tick_params(axis="y", labelsize=8)

    # --- Heatmap ---
    # Custom colormap: white at 0.90, deep warm at 1.0
    cmap = plt.cm.YlOrRd
    norm = mcolors.Normalize(vmin=0.89, vmax=1.0)

    im = ax_heat.imshow(reordered, cmap=cmap, norm=norm, aspect="equal")

    # Annotations
    for i in range(n):
        for j in range(n):
            val = reordered[i, j]
            if i == j:
                text = "1.00"
                color = "white"
                weight = "bold"
            else:
                text = f"{val:.3f}"
                color = "white" if val > 0.975 else "#333333"
                weight = "bold" if val > 0.995 else "normal"
            ax_heat.text(j, i, text, ha="center", va="center",
                         fontsize=9, color=color, fontweight=weight)

    # Labels with graph counts
    display_labels = []
    for name in reordered_names:
        display = TASK_DISPLAY.get(name, name)
        count = cat_counts.get(name, 0)
        display_labels.append(f"{display} (n={count})")

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(display_labels, fontsize=9.5, rotation=45,
                             ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(display_labels, fontsize=9.5)

    # Color the tick labels by task category
    for idx, name in enumerate(reordered_names):
        color = TASK_COLORS.get(name, "#333333")
        ax_heat.get_xticklabels()[idx].set_color(color)
        ax_heat.get_yticklabels()[idx].set_color(color)
        ax_heat.get_xticklabels()[idx].set_fontweight("bold")
        ax_heat.get_yticklabels()[idx].set_fontweight("bold")

    # Highlight key clusters with rectangles
    # Find reasoning and safety indices in reordered list
    rs_indices = [i for i, name in enumerate(reordered_names)
                  if name in ("reasoning", "safety")]
    if len(rs_indices) == 2 and abs(rs_indices[0] - rs_indices[1]) == 1:
        r_min = min(rs_indices)
        rect = plt.Rectangle((r_min - 0.5, r_min - 0.5), 2, 2,
                              fill=False, edgecolor="#e74c3c",
                              linewidth=2.5, linestyle="-", zorder=5)
        ax_heat.add_patch(rect)
        ax_heat.annotate(
            "cos = 0.998",
            xy=(r_min + 1.5, r_min),
            xytext=(r_min - 1.5, r_min - 1.2),
            fontsize=8.5, fontweight="bold", color="#e74c3c",
            arrowprops=dict(arrowstyle="-|>", color="#e74c3c", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#e74c3c", alpha=0.9),
        )

    # Find code and uncategorized indices
    cu_indices = [i for i, name in enumerate(reordered_names)
                  if name in ("code", "uncategorized")]
    if len(cu_indices) == 2 and abs(cu_indices[0] - cu_indices[1]) == 1:
        c_min = min(cu_indices)
        rect = plt.Rectangle((c_min - 0.5, c_min - 0.5), 2, 2,
                              fill=False, edgecolor="#27ae60",
                              linewidth=2.5, linestyle="-", zorder=5)
        ax_heat.add_patch(rect)
        ax_heat.annotate(
            "cos = 0.998",
            xy=(c_min + 0.5, c_min + 1.5),
            xytext=(c_min + 3.0, c_min + 2.5),
            fontsize=8.5, fontweight="bold", color="#27ae60",
            arrowprops=dict(arrowstyle="-|>", color="#27ae60", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#27ae60", alpha=0.9),
        )

    # Find creative index — annotate it as most distant
    creative_idx = next((i for i, name in enumerate(reordered_names)
                         if name == "creative"), None)
    if creative_idx is not None:
        # Find its minimum similarity
        creative_sims = [reordered[creative_idx, j] for j in range(n)
                         if j != creative_idx]
        min_sim = min(creative_sims)
        ax_heat.annotate(
            f"Most distant\n(min = {min_sim:.3f})",
            xy=(creative_idx, creative_idx),
            xytext=(creative_idx - 3.0, creative_idx + 1.5),
            fontsize=8.5, fontweight="bold", color="#e67e22",
            arrowprops=dict(arrowstyle="-|>", color="#e67e22", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="#e67e22", alpha=0.9),
        )

    # Colorbar
    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.set_label("Cosine similarity (SP vectors)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Title
    fig.suptitle(
        "Task-type similarity of motif significance profiles",
        fontsize=14, fontweight="bold", y=0.97,
    )
    fig.text(
        0.15, 0.90,
        "Pairwise cosine similarity of mean SP vectors across 9 task categories. "
        "High similarity (>0.97) across all pairs reflects the universal\n"
        "FFL-dominant motif profile; fine-grained differences reveal "
        "reasoning-safety and code-uncategorized clusters.",
        fontsize=9, color="#555555", va="top",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
