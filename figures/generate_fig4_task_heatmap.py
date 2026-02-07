"""Figure 4: Task-type cosine similarity heatmap with hierarchical clustering.

Shows pairwise cosine similarity between task categories' SP-normalized profiles.
Permutation testing (10,000 permutations, max-statistic FWER correction) shows
that no pairwise similarity is significantly higher than expected from random
label assignment — the universal FFL-dominant profile dominates over task-specific
variation.

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
from scipy.cluster.hierarchy import linkage, leaves_list
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
    "cross_model": "Cross-model",
    "factual_recall": "Factual Recall",
    "multihop": "Multi-hop",
    "multilingual": "Multilingual",
    "reasoning": "Reasoning",
    "safety": "Safety",
    "uncategorized": "Uncategorized",
}

# Category colors
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
    "cross_model": "#e6ab02",
}


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_figure():
    data = load_data()
    sim_data = data["similarity_matrix"]
    task_names = sim_data["task_names"]
    sim_matrix = np.array(sim_data["matrix"])

    # Exclude cross_model from the analysis
    exclude = {"cross_model"}
    keep_idx = [i for i, t in enumerate(task_names) if t not in exclude]
    task_names = [task_names[i] for i in keep_idx]
    sim_matrix = sim_matrix[np.ix_(keep_idx, keep_idx)]

    # Count graphs per category (excluding cross_model)
    cat_counts = {}
    for g in data["graphs"]:
        if g["category"] not in exclude:
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

    # --- Figure: heatmap only ---
    fig, ax_heat = plt.subplots(figsize=(10, 9), facecolor="white")
    fig.subplots_adjust(left=0.22, right=0.88, top=0.88, bottom=0.18)

    # --- Heatmap ---
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
                weight = "normal"
            ax_heat.text(j, i, text, ha="center", va="center",
                         fontsize=8.5, color=color, fontweight=weight)

    # Labels with graph counts
    display_labels = []
    for name in reordered_names:
        display = TASK_DISPLAY.get(name, name)
        count = cat_counts.get(name, 0)
        display_labels.append(f"{display} (n={count})")

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(display_labels, fontsize=9, rotation=45, ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(display_labels, fontsize=9)

    # Color the tick labels by task category
    for idx, name in enumerate(reordered_names):
        color = TASK_COLORS.get(name, "#333333")
        ax_heat.get_xticklabels()[idx].set_color(color)
        ax_heat.get_yticklabels()[idx].set_color(color)
        ax_heat.get_xticklabels()[idx].set_fontweight("bold")
        ax_heat.get_yticklabels()[idx].set_fontweight("bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity (SP vectors)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    # Title
    n_graphs = sum(1 for g in data["graphs"] if g["category"] not in exclude)
    n_tasks = len(task_names)
    fig.suptitle(
        "Task-type similarity of motif significance profiles\n",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.text(
        0.55, 0.93,
        f"Pairwise cosine similarity of mean SP vectors across {n_tasks} task "
        f"categories ({n_graphs} graphs, 1,000 null models each).\n"
        "All pairs show high similarity (0.90\u20130.998), reflecting the universal "
        "FFL-dominant motif profile.",
        fontsize=9, color="#555555", va="top", ha="center",
    )

    # Permutation test note below heatmap
    fig.text(
        0.55, 0.01,
        "Permutation test (10,000 perms, max-statistic FWER correction): "
        "no pairwise similarity reaches significance. "
        "Null mean max cosine = 0.999.",
        fontsize=8.5, color="#888888", fontstyle="italic",
        ha="center", va="bottom",
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
