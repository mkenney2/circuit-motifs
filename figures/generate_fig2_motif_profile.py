"""Figure 2: Aggregate motif significance profile across all 99 attribution graphs.

The single most important result figure. Shows mean SP values per triad class
with individual graph data points overlaid, demonstrating the universal
FFL/chain enrichment and systematic depletion of mutual-edge motifs.

Usage:
    python figures/generate_fig2_motif_profile.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "analysis_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig2_motif_profile.png"

# Connected triads in a logical order: simple → complex, grouped by edge count
# 2-edge triads, then 3-edge, then 4+
MOTIF_ORDER = [
    "012",   # single edge (trivial, always ~0)
    "021D",  # fan-out
    "021U",  # fan-in
    "021C",  # chain
    "030T",  # FFL ← the star
    "030C",  # cycle
    "102",   # mutual
    "111D",  # mutual + out
    "111U",  # mutual + in
    "120D",  # mutual + FFL
    "120U",  # mutual + fan-in
    "120C",  # regulated mutual
    "201",   # double mutual
    "210",   # dense partial
    "300",   # complete
]

# Display names
MOTIF_NAMES = {
    "012": "012\n(edge)",
    "021D": "021D\n(fan-out)",
    "021U": "021U\n(fan-in)",
    "021C": "021C\n(chain)",
    "030T": "030T\n(FFL)",
    "030C": "030C\n(cycle)",
    "102": "102\n(mutual)",
    "111D": "111D",
    "111U": "111U",
    "120D": "120D",
    "120U": "120U",
    "120C": "120C",
    "201": "201",
    "210": "210",
    "300": "300\n(complete)",
}

# Task category colors
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

TASK_SHORT = {
    "factual_recall": "Factual",
    "multihop": "Multihop",
    "arithmetic": "Arith.",
    "creative": "Creative",
    "multilingual": "Multilng.",
    "safety": "Safety",
    "reasoning": "Reason.",
    "code": "Code",
    "uncategorized": "Uncat.",
}


def load_data():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def generate_figure():
    data = load_data()
    graphs = data["graphs"]
    n_graphs = len(graphs)

    # Skip 012 and 102 (always zero) for cleaner figure
    motifs = [m for m in MOTIF_ORDER if m not in ("012", "102")]

    # Collect per-graph SP values
    sp_by_motif = {m: [] for m in motifs}
    cat_by_graph = []
    for g in graphs:
        cat_by_graph.append(g["category"])
        for m in motifs:
            sp_by_motif[m].append(g["significance_profile"].get(m, 0))

    means = [np.mean(sp_by_motif[m]) for m in motifs]
    sems = [np.std(sp_by_motif[m]) / np.sqrt(n_graphs) for m in motifs]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(15, 7), facecolor="white")
    fig.subplots_adjust(left=0.07, right=0.88, top=0.90, bottom=0.16)

    x = np.arange(len(motifs))

    # Bar colors: enriched (warm red), depleted (steel blue), near-zero (gray)
    bar_colors = []
    for m, mean in zip(motifs, means):
        if mean > 0.05:
            bar_colors.append("#c0392b")  # enriched red
        elif mean < -0.05:
            bar_colors.append("#2980b9")  # depleted blue
        else:
            bar_colors.append("#95a5a6")  # neutral gray

    # Bars
    bars = ax.bar(x, means, width=0.55, color=bar_colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Error bars (SEM)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#333333", capsize=3,
                capthick=1.2, elinewidth=1.2, zorder=4)

    # Individual graph dots (strip plot), jittered and colored by category
    rng = np.random.RandomState(42)
    for i, m in enumerate(motifs):
        vals = sp_by_motif[m]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        for j, (val, cat) in enumerate(zip(vals, cat_by_graph)):
            color = TASK_COLORS.get(cat, "#888888")
            ax.scatter(i + jitter[j], val, s=8, color=color, alpha=0.4,
                       edgecolors="none", zorder=2)

    # Zero line
    ax.axhline(y=0, color="#333333", linewidth=0.8, zorder=1)

    # Highlight FFL and Chain
    for highlight_m, label_text, y_offset in [
        ("030T", "FFL: enriched\nin 100% of graphs", 0.07),
        ("021C", "Chain: enriched\nin 96% of graphs", 0.07),
    ]:
        idx = motifs.index(highlight_m)
        ax.annotate(
            label_text,
            xy=(idx, means[idx] + sems[idx]),
            xytext=(idx + 1.8, means[idx] + y_offset),
            fontsize=9, fontweight="bold", color="#333333",
            arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.2,
                            connectionstyle="arc3,rad=-0.15"),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                      edgecolor="#ffc107", alpha=0.9, linewidth=1.0),
            zorder=6,
        )

    # Bracket for "mutual-edge motifs" (111D through 300)
    # Use axis coordinates to draw bracket below x-axis labels
    mutual_start = motifs.index("111D")
    mutual_end = motifs.index("300")
    mid_x = (mutual_start + mutual_end) / 2

    # Label below the bracket using blended transform (data x, axes y)
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(mid_x, -0.10, "Mutual-edge motifs (systematically depleted)",
            transform=blend, fontsize=8.5, color="#666666", fontstyle="italic",
            ha="center", va="top", zorder=5)
    # Horizontal bracket line under the mutual-edge bars
    ax.plot([mutual_start - 0.4, mutual_end + 0.4], [-0.07, -0.07],
            color="#888888", linewidth=1.5, clip_on=False, zorder=5,
            transform=blend)
    # Tick marks at ends
    for bx in [mutual_start - 0.4, mutual_end + 0.4]:
        ax.plot([bx, bx], [-0.07, -0.05], color="#888888", linewidth=1.5,
                clip_on=False, zorder=5, transform=blend)

    # X-axis
    ax.set_xticks(x)
    ax.set_xticklabels([MOTIF_NAMES.get(m, m) for m in motifs],
                       fontsize=8.5, ha="center")
    ax.set_ylabel("Mean Significance Profile (SP)", fontsize=12)
    ax.set_xlim(-0.6, len(motifs) - 0.4)

    # Y-axis formatting
    ax.tick_params(axis="y", labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Category legend on the right side
    legend_handles = []
    cat_order = ["factual_recall", "multihop", "arithmetic", "reasoning",
                 "safety", "code", "multilingual", "creative", "uncategorized"]
    for cat in cat_order:
        n = sum(1 for c in cat_by_graph if c == cat)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=TASK_COLORS[cat], markersize=6,
                       label=f"{TASK_SHORT[cat]} ({n})")
        )
    ax.legend(handles=legend_handles, loc="center left",
              bbox_to_anchor=(1.01, 0.5), fontsize=8.5,
              title="Task category", title_fontsize=9,
              framealpha=0.9, edgecolor="#cccccc")

    # Title
    ax.set_title(
        f"Aggregate motif significance profile across {n_graphs} attribution graphs\n"
        "Feedforward loops and chains are universally enriched; "
        "mutual-edge motifs are universally depleted",
        fontsize=12, fontweight="bold", loc="left", pad=10,
    )

    # Stats annotation
    stats_text = (
        f"n = {n_graphs} graphs, 9 task categories\n"
        "100 null models per graph (degree-preserving)\n"
        "Dots = individual graphs, bars = mean SP"
    )
    ax.text(0.99, 0.97, stats_text, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="#888888",
            fontstyle="italic")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
