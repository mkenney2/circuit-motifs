"""Generate CLT vs PLT motif profile comparison figure.

Compares Significance Profile (SP) vectors between cross-layer transcoder (CLT)
and per-layer transcoder (PLT) attribution graphs for the same prompts on
Claude Haiku. Shows that the overall motif profile shape is preserved (cosine
similarity 0.97) but the dominant motif shifts: chains dominate CLT, FFLs
dominate PLT.

Usage:
    python figures/generate_clt_vs_plt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "analysis_summary.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig_clt_vs_plt.png"

# Matched CLT/PLT pairs (same prompt, same model, different transcoder)
PAIRS = [
    ("capital-analogy-clt-clean", "capital-analogy-plt-clean"),
    ("common-colors-clt-clean", "common-colors-plt-clean"),
    ("count-by-sevens-clt-clean", "count-by-sevens-plt-clean"),
    ("currency-analogy-clt-clean", "currency-analogy-plt-clean"),
    ("five-plus-three-clt-clean", "five-plus-three-plt-clean"),
    ("iasg-clt-clean", "iasg-plt-clean"),
    ("michael-clt-clean", "michael-plt-clean"),
    ("michael-fr-clt-clean", "michael-fr-plt-clean"),
    ("ndag-clt-clean", "ndag-plt-clean"),
    ("opposite-hot-clt-clean", "opposite-hot-plt-clean"),
    ("opposite-of-small-clt-clean", "opposite-of-small-plt-clean"),
    ("pandas-group-clt-clean", "pandas-group-plt-clean"),
    ("sally-school-clt-clean", "sally-school-plt-clean"),
    ("season-after-spring-fr-clt-clean", "season-after-spring-fr-plt-clean"),
    ("str-indexing-pos-0-clt-clean", "str-indexing-pos-0-plt-clean"),
    ("uspto-telephone-clt-clean", "uspto-telephone-plt-clean"),
]

# Connected triads only (skip 003 and 012 which are trivial)
MOTIFS = ["021U", "102", "021C", "111U", "021D", "030T",
          "120U", "111D", "201", "030C", "120C", "120D", "210", "300"]

MOTIF_DISPLAY = {
    "021U": "Fan-in\n021U", "102": "Mutual\n102",
    "021C": "Chain\n021C", "111U": "111U",
    "021D": "Fan-out\n021D", "030T": "FFL\n030T",
    "120U": "120U", "111D": "111D", "201": "201",
    "030C": "Cycle\n030C", "120C": "120C", "120D": "120D",
    "210": "210", "300": "Complete\n300",
}

CLT_COLOR = "#e66101"
PLT_COLOR = "#5e3c99"


def load_data():
    """Load analysis results and extract paired SP vectors."""
    with open(RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)

    by_name = {g["name"]: g for g in data["graphs"]}

    clt_sp = {m: [] for m in MOTIFS}
    plt_sp = {m: [] for m in MOTIFS}

    for clt_name, plt_name in PAIRS:
        clt = by_name[clt_name]
        plt_g = by_name[plt_name]
        for m in MOTIFS:
            clt_sp[m].append(clt["significance_profile"].get(m, 0))
            plt_sp[m].append(plt_g["significance_profile"].get(m, 0))

    return clt_sp, plt_sp


def generate_figure():
    """Generate the CLT vs PLT comparison figure."""
    clt_sp, plt_sp = load_data()
    n_pairs = len(PAIRS)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 2])
    fig.subplots_adjust(hspace=0.35, left=0.08, right=0.95, top=0.90, bottom=0.08)

    # ===== Panel A: Mean SP profiles side by side =====
    ax = axes[0]

    clt_means = [np.mean(clt_sp[m]) for m in MOTIFS]
    plt_means = [np.mean(plt_sp[m]) for m in MOTIFS]
    clt_sems = [np.std(clt_sp[m]) / np.sqrt(n_pairs) for m in MOTIFS]
    plt_sems = [np.std(plt_sp[m]) / np.sqrt(n_pairs) for m in MOTIFS]

    x = np.arange(len(MOTIFS))
    w = 0.35

    bars_clt = ax.bar(x - w / 2, clt_means, w, yerr=clt_sems, capsize=3,
                       color=CLT_COLOR, alpha=0.85, edgecolor="white",
                       linewidth=0.5, label="CLT (cross-layer)", error_kw=dict(lw=1))
    bars_plt = ax.bar(x + w / 2, plt_means, w, yerr=plt_sems, capsize=3,
                       color=PLT_COLOR, alpha=0.85, edgecolor="white",
                       linewidth=0.5, label="PLT (per-layer)", error_kw=dict(lw=1))

    ax.set_xticks(x)
    ax.set_xticklabels([MOTIF_DISPLAY[m] for m in MOTIFS], fontsize=8)
    ax.set_ylabel("Mean Significance Profile", fontsize=11)
    ax.axhline(y=0, color="#cccccc", linewidth=0.8)
    ax.legend(fontsize=10, loc="upper right", framealpha=0.9)

    # Highlight the key motifs (FFL and Chain) with annotations
    ffl_idx = MOTIFS.index("030T")
    chain_idx = MOTIFS.index("021C")

    # Significance stars
    for i, m in enumerate(MOTIFS):
        try:
            _, p = stats.wilcoxon(clt_sp[m], plt_sp[m])
            if p < 0.001:
                star = "***"
            elif p < 0.01:
                star = "**"
            elif p < 0.05:
                star = "*"
            else:
                continue
            max_val = max(abs(clt_means[i]) + clt_sems[i],
                          abs(plt_means[i]) + plt_sems[i])
            y_star = max(clt_means[i] + clt_sems[i],
                         plt_means[i] + plt_sems[i]) + 0.02
            if clt_means[i] < 0 and plt_means[i] < 0:
                y_star = min(clt_means[i] - clt_sems[i],
                             plt_means[i] - plt_sems[i]) - 0.04
            ax.text(i, y_star, star, ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#333333")
        except Exception:
            pass

    # Annotate the key finding — position below bars to avoid title overlap
    ax.annotate(
        "CLT: chains dominate",
        xy=(chain_idx - w / 2, clt_means[chain_idx] + clt_sems[chain_idx]),
        xytext=(chain_idx + 2.5, 0.35),
        fontsize=9, fontweight="bold", color=CLT_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=CLT_COLOR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=CLT_COLOR, alpha=0.9),
    )
    ax.annotate(
        "PLT: FFLs dominate",
        xy=(ffl_idx + w / 2, plt_means[ffl_idx] + plt_sems[ffl_idx]),
        xytext=(ffl_idx + 2.5, 0.35),
        fontsize=9, fontweight="bold", color=PLT_COLOR,
        arrowprops=dict(arrowstyle="-|>", color=PLT_COLOR, lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=PLT_COLOR, alpha=0.9),
    )

    # Cosine similarity annotation
    clt_vec = np.array(clt_means)
    plt_vec = np.array(plt_means)
    cos_sim = np.dot(clt_vec, plt_vec) / (np.linalg.norm(clt_vec) * np.linalg.norm(plt_vec))
    ax.text(0.02, 0.97, f"Cosine similarity: {cos_sim:.3f}\n({n_pairs} matched prompt pairs)",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_title(
        "A.  Motif significance profiles: CLT vs PLT (same model, same prompts)",
        fontsize=12, fontweight="bold", loc="left",
    )

    # ===== Panel B: Per-pair scatter for FFL and Chain =====
    ax = axes[1]

    # FFL scatter
    ax.scatter(
        clt_sp["030T"], plt_sp["030T"],
        c=PLT_COLOR, s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
        label="FFL (030T)", marker="o", zorder=3,
    )
    # Chain scatter
    ax.scatter(
        clt_sp["021C"], plt_sp["021C"],
        c=CLT_COLOR, s=50, alpha=0.7, edgecolors="white", linewidths=0.5,
        label="Chain (021C)", marker="s", zorder=3,
    )

    # Identity line
    lim_min = min(min(clt_sp["030T"]), min(plt_sp["030T"]),
                  min(clt_sp["021C"]), min(plt_sp["021C"])) - 0.05
    lim_max = max(max(clt_sp["030T"]), max(plt_sp["030T"]),
                  max(clt_sp["021C"]), max(plt_sp["021C"])) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="#bbbbbb",
            linewidth=1, zorder=1)

    ax.set_xlabel("CLT Significance Profile", fontsize=11)
    ax.set_ylabel("PLT Significance Profile", fontsize=11)
    ax.legend(fontsize=10, loc="lower right", framealpha=0.9)

    # Annotate correlations
    r_ffl, p_ffl = stats.spearmanr(clt_sp["030T"], plt_sp["030T"])
    r_chain, p_chain = stats.spearmanr(clt_sp["021C"], plt_sp["021C"])
    ax.text(0.02, 0.97,
            f"FFL: Spearman r = {r_ffl:.2f} (p={p_ffl:.3f})\n"
            f"Chain: Spearman r = {r_chain:.2f} (p={p_chain:.3f})",
            transform=ax.transAxes, fontsize=9, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    # Show that FFL points are mostly above the diagonal (PLT > CLT)
    # and Chain points are mostly below (CLT > PLT)
    n_ffl_above = sum(1 for c, p in zip(clt_sp["030T"], plt_sp["030T"]) if p > c)
    n_chain_below = sum(1 for c, p in zip(clt_sp["021C"], plt_sp["021C"]) if c > p)
    ax.text(0.98, 0.03,
            f"FFL: PLT > CLT in {n_ffl_above}/{n_pairs} pairs\n"
            f"Chain: CLT > PLT in {n_chain_below}/{n_pairs} pairs",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0",
                      edgecolor="#cccccc", alpha=0.9))

    ax.set_title(
        "B.  Per-prompt pair comparison: FFL and Chain SP values",
        fontsize=12, fontweight="bold", loc="left",
    )

    # --- Suptitle ---
    fig.suptitle(
        "Transcoder architecture shapes motif profiles:\n"
        "cross-layer (CLT) vs per-layer (PLT) on Claude Haiku",
        fontsize=13, fontweight="bold", y=0.98,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
