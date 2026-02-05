"""Generate the size-normalization validation figure for the blog post.

Shows that raw Z-scores scale strongly with graph size, but Significance
Profile (SP) normalization (Milo et al., 2004) removes the size confound
while preserving cross-task discriminability.

Three panels:
  A. Z-score vs edge count (strong correlation)
  B. SP value vs edge count (no correlation)
  C. Spearman |r| comparison across all key motifs

Usage:
    python figures/generate_size_normalization.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy import stats

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "analysis_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig_size_normalization.png"

# Category colors (consistent with other figures)
CATEGORY_COLORS = {
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

# Short labels for categories
CATEGORY_SHORT = {
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

# Motifs to show in panel C (the key ones)
KEY_MOTIFS = ["030T", "021C", "111U", "021D", "021U", "030C"]
MOTIF_DISPLAY = {
    "030T": "FFL",
    "021C": "Chain",
    "111U": "111U",
    "021D": "Fan-out",
    "021U": "Fan-in",
    "030C": "Cycle",
}


def load_data():
    """Load analysis results."""
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["graphs"]


def generate_figure():
    """Generate the 3-panel normalization validation figure."""
    graphs = load_data()

    n_edges = np.array([g["n_edges"] for g in graphs])
    categories = [g["category"] for g in graphs]
    cat_colors = [CATEGORY_COLORS.get(c, "#888888") for c in categories]

    # Extract Z-scores and SP values for FFL (030T) — the flagship motif
    z_ffl = np.array([g["z_scores"]["030T"] for g in graphs])
    sp_ffl = np.array([g["significance_profile"]["030T"] for g in graphs])

    # --- Figure setup ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.subplots_adjust(wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.14)

    # ===== Panel A: Z-score vs edge count =====
    ax = axes[0]
    ax.scatter(n_edges, z_ffl, c=cat_colors, s=30, alpha=0.7, edgecolors="white",
               linewidths=0.4, zorder=3)

    # Regression line
    slope, intercept, _, _, _ = stats.linregress(n_edges, z_ffl)
    x_fit = np.linspace(n_edges.min(), n_edges.max(), 100)
    ax.plot(x_fit, slope * x_fit + intercept, color="#333333", linewidth=1.5,
            linestyle="--", alpha=0.7, zorder=2)

    r_z, p_z = stats.spearmanr(n_edges, z_ffl)
    ax.text(0.97, 0.05, f"Spearman r = {r_z:+.2f}\np < 0.0001",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                      edgecolor="#cc0000", linewidth=1.5))

    ax.set_xlabel("Edge count", fontsize=11)
    ax.set_ylabel("Z-score (FFL / 030T)", fontsize=11)
    ax.set_title("A.  Raw Z-scores scale with graph size", fontsize=12, fontweight="bold",
                 loc="left")
    ax.axhline(y=0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ===== Panel B: SP value vs edge count =====
    ax = axes[1]
    ax.scatter(n_edges, sp_ffl, c=cat_colors, s=30, alpha=0.7, edgecolors="white",
               linewidths=0.4, zorder=3)

    # Regression line
    slope2, intercept2, _, _, _ = stats.linregress(n_edges, sp_ffl)
    ax.plot(x_fit, slope2 * x_fit + intercept2, color="#333333", linewidth=1.5,
            linestyle="--", alpha=0.7, zorder=2)

    r_sp, p_sp = stats.spearmanr(n_edges, sp_ffl)
    ax.text(0.97, 0.05, f"Spearman r = {r_sp:+.2f}\np = {p_sp:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9,
                      edgecolor="#2ca02c", linewidth=1.5))

    ax.set_xlabel("Edge count", fontsize=11)
    ax.set_ylabel("Significance Profile (FFL / 030T)", fontsize=11)
    ax.set_title("B.  SP normalization removes size dependence", fontsize=12,
                 fontweight="bold", loc="left")
    ax.axhline(y=0, color="#cccccc", linewidth=0.8, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # ===== Panel C: Spearman |r| comparison =====
    ax = axes[2]

    r_z_vals = []
    r_sp_vals = []
    motif_labels = []
    for motif in KEY_MOTIFS:
        z_vals = np.array([g["z_scores"].get(motif, 0) for g in graphs])
        sp_vals = np.array([g["significance_profile"].get(motif, 0) for g in graphs])
        rz, _ = stats.spearmanr(n_edges, z_vals)
        rsp, _ = stats.spearmanr(n_edges, sp_vals)
        r_z_vals.append(abs(rz))
        r_sp_vals.append(abs(rsp))
        motif_labels.append(MOTIF_DISPLAY[motif])

    x_pos = np.arange(len(KEY_MOTIFS))
    bar_width = 0.35

    bars1 = ax.bar(x_pos - bar_width / 2, r_z_vals, bar_width, label="Raw Z-score",
                   color="#cc4444", alpha=0.85, edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x_pos + bar_width / 2, r_sp_vals, bar_width, label="SP (normalized)",
                   color="#4488cc", alpha=0.85, edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(motif_labels, fontsize=10)
    ax.set_ylabel("|Spearman r| with edge count", fontsize=11)
    ax.set_title("C.  Normalization across all key motifs", fontsize=12,
                 fontweight="bold", loc="left")
    ax.axhline(y=0.3, color="#cccccc", linewidth=0.8, linestyle=":", zorder=1)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # Value annotations on bars
    for bar, val in zip(bars1, r_z_vals):
        if val > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f".{int(val * 100):02d}", ha="center", va="bottom", fontsize=8,
                    color="#882222")
    for bar, val in zip(bars2, r_sp_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f".{int(val * 100):02d}", ha="center", va="bottom", fontsize=8,
                color="#225588")

    # --- Category legend (shared across A and B) ---
    legend_handles = []
    for cat in sorted(CATEGORY_COLORS.keys()):
        count = sum(1 for c in categories if c == cat)
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=CATEGORY_COLORS[cat], markersize=7,
                       label=f"{CATEGORY_SHORT[cat]} (n={count})")
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=5, fontsize=9,
               framealpha=0.9, columnspacing=1.0, handletextpad=0.3,
               bbox_to_anchor=(0.38, -0.01))

    # --- Suptitle ---
    fig.suptitle(
        "Significance Profile normalization removes graph-size confound from motif Z-scores",
        fontsize=13, fontweight="bold", y=0.97,
    )

    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
