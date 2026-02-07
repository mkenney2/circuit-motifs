"""Figure 2: Aggregate motif significance profile across attribution graphs.

Shows mean SP values per motif class with error bars. Mutual-edge motifs
(111D through 300) are averaged into a single "Mutual-edge" bar for clarity.

Usage:
    python figures/generate_fig2_motif_profile.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "analysis_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig2_motif_profile.png"

# Individual motifs to show as separate bars
INDIVIDUAL_MOTIFS = ["021D", "021U", "021C", "030T", "030C"]

# Mutual-edge motifs to average into one bar
MUTUAL_MOTIFS = ["111D", "111U", "120D", "120U", "120C", "201", "210", "300"]

# Display labels
BAR_LABELS = [
    "021D\n(fan-out)",
    "021U\n(fan-in)",
    "021C\n(chain)",
    "030T\n(FFL)",
    "030C\n(cycle)",
    "Mutual-edge\n(8 classes avg.)",
]


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_figure():
    data = load_data()

    # Exclude cross_model graphs
    graphs = [g for g in data["graphs"] if g["category"] != "cross_model"]
    n_graphs = len(graphs)
    n_tasks = len(set(g["category"] for g in graphs))

    # Collect per-graph SP values for individual motifs
    individual_vals = {m: [] for m in INDIVIDUAL_MOTIFS}
    mutual_vals = []  # each entry is the mean of mutual motifs for one graph

    for g in graphs:
        sp = g["significance_profile"]
        for m in INDIVIDUAL_MOTIFS:
            individual_vals[m].append(sp.get(m, 0))
        graph_mutual = [sp.get(m, 0) for m in MUTUAL_MOTIFS]
        mutual_vals.append(np.mean(graph_mutual))

    # Compute bar heights and error bars
    all_vals = [individual_vals[m] for m in INDIVIDUAL_MOTIFS] + [mutual_vals]
    means = [np.mean(v) for v in all_vals]
    sems = [np.std(v) / np.sqrt(n_graphs) for v in all_vals]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    fig.subplots_adjust(left=0.10, right=0.95, top=0.87, bottom=0.15)

    x = np.arange(len(means))

    # Bar colors
    bar_colors = []
    for mean in means:
        if mean > 0.05:
            bar_colors.append("#c0392b")
        elif mean < -0.05:
            bar_colors.append("#2980b9")
        else:
            bar_colors.append("#95a5a6")

    # Bars
    ax.bar(x, means, width=0.6, color=bar_colors, alpha=0.85,
           edgecolor="white", linewidth=0.8, zorder=3)

    # Error bars (SEM)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#333333", capsize=4,
                capthick=1.2, elinewidth=1.2, zorder=4)

    # Zero line
    ax.axhline(y=0, color="#333333", linewidth=0.8, zorder=1)

    # Annotate FFL and Chain
    # FFL annotation (right side, below title)
    ffl_idx = 3
    ax.annotate(
        "Enriched in 100%\nof graphs",
        xy=(ffl_idx, means[ffl_idx] + sems[ffl_idx]),
        xytext=(ffl_idx + 1.3, means[ffl_idx] - 0.10),
        fontsize=9, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                  edgecolor="#ffc107", alpha=0.9, linewidth=1.0),
        ha="center", zorder=6,
    )
    # Chain annotation (left side)
    chain_idx = 2
    ax.annotate(
        "Enriched in 97%\nof graphs",
        xy=(chain_idx, means[chain_idx] + sems[chain_idx]),
        xytext=(chain_idx - 1.5, means[chain_idx] - 0.05),
        fontsize=9, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2,
                        connectionstyle="arc3,rad=0.2"),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                  edgecolor="#ffc107", alpha=0.9, linewidth=1.0),
        ha="center", zorder=6,
    )

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels(BAR_LABELS, fontsize=9.5, ha="center")
    ax.set_ylabel("Mean Significance Profile (SP)", fontsize=12)
    ax.set_xlim(-0.6, len(means) - 0.4)

    # Y-axis formatting
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title
    ax.set_title(
        f"Motif significance profile across {n_graphs} LLM attribution graphs",
        fontsize=13, fontweight="bold", loc="left", pad=10,
    )

    # Stats annotation (bottom-left to avoid collision with annotations)
    stats_text = (
        f"n = {n_graphs} graphs, {n_tasks} task categories\n"
        "1,000 null models per graph (degree-preserving)\n"
        "Error bars = SEM"
    )
    ax.text(0.42, 0.03, stats_text, transform=ax.transAxes,
            fontsize=8, va="bottom", ha="center", color="#888888",
            fontstyle="italic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
