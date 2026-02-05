"""Generate cross-model motif comparison figure.

Compares SP vectors for the Dallas multihop prompt across three models:
Claude Haiku (CLT), Gemma-2-2B (CLT), Qwen3-4B (CLT). Shows FFL dominance
is universal across model architectures and scales.

Usage:
    python figures/generate_cross_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "cross_model_results.json"
OUTPUT_PATH = PROJECT_ROOT / "figures" / "fig_cross_model.png"

# Use one representative threshold per model
MODELS = {
    "Haiku-CLT (original)": {"color": "#2166ac", "short": "Claude Haiku\n(CLT, 83n)"},
    "Gemma-2-2B (t=10)": {"color": "#d6604d", "short": "Gemma-2-2B\n(CLT, 432e)"},
    "Qwen3-4B (t=3)": {"color": "#1b7837", "short": "Qwen3-4B\n(CLT, 599e)"},
}

# Show connected triads only
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


def generate_figure():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8.5), height_ratios=[3, 1.3])
    fig.subplots_adjust(hspace=0.40, left=0.08, right=0.95, top=0.88, bottom=0.08)

    # ===== Panel A: SP bar chart =====
    ax = axes[0]
    n_models = len(MODELS)
    x = np.arange(len(MOTIFS))
    w = 0.8 / n_models

    for i, (model_key, info) in enumerate(MODELS.items()):
        sp = results[model_key]["significance_profile"]
        vals = [sp.get(m, 0) for m in MOTIFS]
        offset = (i - (n_models - 1) / 2) * w
        ax.bar(x + offset, vals, w, color=info["color"], alpha=0.85,
               edgecolor="white", linewidth=0.5, label=info["short"].replace("\n", " "))

    ax.set_xticks(x)
    ax.set_xticklabels([MOTIF_DISPLAY[m] for m in MOTIFS], fontsize=8)
    ax.set_ylabel("Significance Profile", fontsize=11)
    ax.axhline(y=0, color="#cccccc", linewidth=0.8)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9, ncol=1)

    # Highlight FFL
    ffl_idx = MOTIFS.index("030T")
    ax.axvspan(ffl_idx - 0.5, ffl_idx + 0.5, color="#fff3cd", alpha=0.4, zorder=0)
    ax.annotate(
        "FFL enriched in\nall 3 models",
        xy=(ffl_idx, 0.75), xytext=(ffl_idx + 3, 0.72),
        fontsize=9, fontweight="bold", color="#333333",
        arrowprops=dict(arrowstyle="-|>", color="#666666", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff3cd",
                  edgecolor="#ffc107", alpha=0.9),
    )

    ax.set_title(
        'A.  Motif profiles for "capital of the state containing Dallas" across 3 models',
        fontsize=12, fontweight="bold", loc="left",
    )

    # ===== Panel B: Cosine similarity + threshold robustness =====
    ax = axes[1]

    # Build comparison data: all configs
    all_labels = list(results.keys())
    # Group by model
    haiku = [k for k in all_labels if "Haiku" in k]
    gemma = [k for k in all_labels if "Gemma" in k]
    qwen = [k for k in all_labels if "Qwen" in k]

    motifs_all = [m for m in MOTIFS]

    def cos_sim(k1, k2):
        v1 = np.array([results[k1]["significance_profile"].get(m, 0) for m in motifs_all])
        v2 = np.array([results[k2]["significance_profile"].get(m, 0) for m in motifs_all])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0
        return np.dot(v1, v2) / (n1 * n2)

    # Build comparison table
    comparisons = []

    # Within-model (threshold robustness)
    if len(gemma) >= 2:
        comparisons.append(("Gemma-2-2B\n(t=10 vs t=15)", cos_sim(gemma[0], gemma[1]), "#d6604d", "Within"))
    if len(qwen) >= 2:
        comparisons.append(("Qwen3-4B\n(t=3 vs t=5)", cos_sim(qwen[0], qwen[1]), "#1b7837", "Within"))

    # Cross-model (using representative thresholds)
    h_key = "Haiku-CLT (original)"
    g_key = "Gemma-2-2B (t=10)"
    q_key = "Qwen3-4B (t=3)"

    comparisons.append(("Haiku vs\nGemma-2-2B", cos_sim(h_key, g_key), "#7570b3", "Cross"))
    comparisons.append(("Haiku vs\nQwen3-4B", cos_sim(h_key, q_key), "#7570b3", "Cross"))
    comparisons.append(("Gemma vs\nQwen3-4B", cos_sim(g_key, q_key), "#7570b3", "Cross"))

    labels = [c[0] for c in comparisons]
    values = [c[1] for c in comparisons]
    colors = [c[2] for c in comparisons]
    types = [c[3] for c in comparisons]

    bars = ax.barh(range(len(comparisons)), values, color=colors, alpha=0.85,
                   edgecolor="white", linewidth=0.5, height=0.6)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0.75, 1.02)
    ax.set_xlabel("Cosine similarity of SP vectors", fontsize=10)
    ax.invert_yaxis()

    # Separator between within-model and cross-model
    n_within = sum(1 for t in types if t == "Within")
    if n_within > 0:
        ax.axhline(y=n_within - 0.5, color="#cccccc", linewidth=1, linestyle="--")
        ax.text(0.76, n_within / 2 - 0.5, "Threshold\nrobustness",
                fontsize=8, va="center", ha="left", fontstyle="italic", color="#888888")
        ax.text(0.76, n_within + (len(comparisons) - n_within) / 2 - 0.5,
                "Cross-model",
                fontsize=8, va="center", ha="left", fontstyle="italic", color="#888888")

    ax.set_title(
        "B.  Robustness: within-model threshold sensitivity and cross-model similarity",
        fontsize=12, fontweight="bold", loc="left",
    )

    fig.suptitle(
        "FFL dominance is universal across transformer architectures",
        fontsize=13, fontweight="bold", y=0.96,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    generate_figure()
