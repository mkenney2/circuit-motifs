"""Generate publication-ready figures for unrolled motif analysis.

Reads unrolled_analysis.json and pilot_results.json, produces:
  1. Universal enrichment heatmap (all null models x motifs)
  2. Z-score heatmap (tasks x motifs) for LP-ER and LPC-shuf
  3. Task similarity matrix + dendrogram for LPC-shuf
  4. Sign effect comparison (LPC-shuf vs LPC-sign)
  5. SP profile comparison across tasks for LPC-shuf

Usage:
    python scripts/unrolled_figures.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cosine

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ── Constants ─────────────────────────────────────────────────────────

RESULTS_DIR = _REPO / "data" / "results" / "unrolled_null_pilot"
FIGURES_DIR = _REPO / "figures"

TEMPLATE_NAMES = [
    "cross_chain_inhibition",
    "feedforward_damping",
    "feedforward_amplification",
    "residual_self_loop_positive",
    "residual_self_loop_negative",
    "coherent_ffl",
    "incoherent_ffl",
    "cross_chain_toggle",
]

SHORT_NAMES = {
    "cross_chain_inhibition": "Cross-chain\ninhibition",
    "feedforward_damping": "FF\ndamping",
    "feedforward_amplification": "FF\namplification",
    "residual_self_loop_positive": "Self-loop\n(+)",
    "residual_self_loop_negative": "Self-loop\n(-)",
    "coherent_ffl": "Coherent\nFFL",
    "incoherent_ffl": "Incoherent\nFFL",
    "cross_chain_toggle": "Cross-chain\ntoggle",
}

SHORT_NAMES_ONELINE = {
    "cross_chain_inhibition": "XChainInhib",
    "feedforward_damping": "FF Damp",
    "feedforward_amplification": "FF Amp",
    "residual_self_loop_positive": "SelfLoop+",
    "residual_self_loop_negative": "SelfLoop-",
    "coherent_ffl": "Coher. FFL",
    "incoherent_ffl": "Incoher. FFL",
    "cross_chain_toggle": "XChainToggle",
}

NULL_TYPES = [
    "configuration",
    "erdos_renyi",
    "layer_preserving_er",
    "layer_pair_config",
    "layer_pair_config_signs",
]

NULL_LABELS = {
    "configuration": "Configuration",
    "erdos_renyi": "Erdos-Renyi",
    "layer_preserving_er": "Layer-pres. ER",
    "layer_pair_config": "LPC (shuf. signs)",
    "layer_pair_config_signs": "LPC (pres. signs)",
}

TASK_COLORS = {
    "factual_recall": "#1f77b4",
    "multihop": "#ff7f0e",
    "arithmetic": "#2ca02c",
    "creative": "#d62728",
    "multilingual": "#9467bd",
    "safety": "#8c564b",
    "reasoning": "#e377c2",
    "code": "#17becf",
    "uncategorized": "#7f7f7f",
}

TASK_ORDER = [
    "factual_recall", "multihop", "arithmetic", "creative",
    "multilingual", "safety", "reasoning", "code", "uncategorized",
]


# ── Data loading ──────────────────────────────────────────────────────

def load_data():
    """Load pilot results and analysis JSON."""
    with open(RESULTS_DIR / "pilot_results.json") as f:
        pilot = json.load(f)
    with open(RESULTS_DIR / "unrolled_analysis.json") as f:
        analysis = json.load(f)
    return pilot, analysis


def compute_sp(z_dict: dict[str, float]) -> np.ndarray:
    """Z-score dict -> SP vector."""
    z = np.array([z_dict[t] for t in TEMPLATE_NAMES])
    norm = np.sqrt(np.sum(z ** 2))
    return z / norm if norm > 1e-10 else np.zeros_like(z)


# ── Figure 1: Universal enrichment heatmap ────────────────────────────

def fig_universal_enrichment(pilot: dict):
    """Heatmap: mean Z-score per motif per null model, all 99 graphs."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), width_ratios=[1, 1, 1])
    fig.suptitle("Unrolled motif enrichment across 99 attribution graphs",
                 fontsize=14, fontweight="bold", y=1.02)

    motif_labels = [SHORT_NAMES_ONELINE[t] for t in TEMPLATE_NAMES]
    null_labels = [NULL_LABELS[nt] for nt in NULL_TYPES]

    # Panel A: Mean Z-score heatmap
    z_matrix = np.zeros((len(NULL_TYPES), len(TEMPLATE_NAMES)))
    for ni, nt in enumerate(NULL_TYPES):
        for mi, mname in enumerate(TEMPLATE_NAMES):
            zs = [pilot[g][nt]["z_scores"][mname] for g in pilot if nt in pilot[g]]
            z_matrix[ni, mi] = np.mean(zs)

    ax = axes[0]
    vmax = max(abs(z_matrix.min()), abs(z_matrix.max()))
    # Cap for readability
    z_display = np.clip(z_matrix, -50, 50)
    im = ax.imshow(z_display, cmap="RdBu_r", aspect="auto",
                   vmin=-50, vmax=50)
    ax.set_xticks(range(len(TEMPLATE_NAMES)))
    ax.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(NULL_TYPES)))
    ax.set_yticklabels(null_labels, fontsize=9)
    ax.set_title("Mean Z-score", fontsize=11)
    # Annotate
    for ni in range(len(NULL_TYPES)):
        for mi in range(len(TEMPLATE_NAMES)):
            val = z_matrix[ni, mi]
            txt = f"{val:+.0f}" if abs(val) < 1000 else f"{val:+.0e}"
            color = "white" if abs(z_display[ni, mi]) > 25 else "black"
            ax.text(mi, ni, txt, ha="center", va="center", fontsize=7, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Z-score (clipped)")

    # Panel B: % enriched
    enrich_matrix = np.zeros((len(NULL_TYPES), len(TEMPLATE_NAMES)))
    for ni, nt in enumerate(NULL_TYPES):
        for mi, mname in enumerate(TEMPLATE_NAMES):
            zs = [pilot[g][nt]["z_scores"][mname] for g in pilot if nt in pilot[g]]
            enrich_matrix[ni, mi] = sum(1 for z in zs if z > 2.0) / len(zs) * 100

    ax = axes[1]
    im2 = ax.imshow(enrich_matrix, cmap="Reds", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(TEMPLATE_NAMES)))
    ax.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(NULL_TYPES)))
    ax.set_yticklabels(null_labels, fontsize=9)
    ax.set_title("% graphs enriched (Z > 2)", fontsize=11)
    for ni in range(len(NULL_TYPES)):
        for mi in range(len(TEMPLATE_NAMES)):
            val = enrich_matrix[ni, mi]
            color = "white" if val > 50 else "black"
            ax.text(mi, ni, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im2, ax=ax, shrink=0.8, label="% enriched")

    # Panel C: % depleted
    deplete_matrix = np.zeros((len(NULL_TYPES), len(TEMPLATE_NAMES)))
    for ni, nt in enumerate(NULL_TYPES):
        for mi, mname in enumerate(TEMPLATE_NAMES):
            zs = [pilot[g][nt]["z_scores"][mname] for g in pilot if nt in pilot[g]]
            deplete_matrix[ni, mi] = sum(1 for z in zs if z < -2.0) / len(zs) * 100

    ax = axes[2]
    im3 = ax.imshow(deplete_matrix, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(TEMPLATE_NAMES)))
    ax.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(NULL_TYPES)))
    ax.set_yticklabels(null_labels, fontsize=9)
    ax.set_title("% graphs depleted (Z < -2)", fontsize=11)
    for ni in range(len(NULL_TYPES)):
        for mi in range(len(TEMPLATE_NAMES)):
            val = deplete_matrix[ni, mi]
            color = "white" if val > 50 else "black"
            ax.text(mi, ni, f"{val:.0f}%", ha="center", va="center",
                    fontsize=7, color=color)
    plt.colorbar(im3, ax=ax, shrink=0.8, label="% depleted")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_universal_enrichment.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 2: Z-score heatmaps (tasks x motifs) ──────────────────────

def fig_zscore_heatmaps(pilot: dict):
    """Side-by-side Z-score heatmaps for LP-ER and LPC-shuf."""
    null_types_to_show = ["layer_preserving_er", "layer_pair_config", "layer_pair_config_signs"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mean Z-scores by task category (layer-aware null models)",
                 fontsize=14, fontweight="bold", y=1.02)

    motif_labels = [SHORT_NAMES_ONELINE[t] for t in TEMPLATE_NAMES]
    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]

    for idx, nt in enumerate(null_types_to_show):
        z_matrix = np.zeros((len(tasks), len(TEMPLATE_NAMES)))
        for ti, task in enumerate(tasks):
            for mi, mname in enumerate(TEMPLATE_NAMES):
                zs = [pilot[g][nt]["z_scores"][mname]
                      for g in pilot if g.startswith(task + "/") and nt in pilot[g]]
                z_matrix[ti, mi] = np.mean(zs) if zs else 0

        ax = axes[idx]
        vmax = max(abs(z_matrix.min()), abs(z_matrix.max()), 5)
        vmax = min(vmax, 30)  # cap for readability
        sns.heatmap(z_matrix, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    xticklabels=motif_labels, yticklabels=tasks,
                    annot=True, fmt=".1f", annot_kws={"size": 7},
                    linewidths=0.5, cbar_kws={"label": "Z-score", "shrink": 0.8})
        ax.set_title(NULL_LABELS[nt], fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_zscore_heatmaps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 3: Task similarity matrix + dendrogram ────────────────────

def fig_task_similarity(pilot: dict):
    """Cosine similarity heatmap + dendrogram for LPC-shuf."""
    null_type = "layer_pair_config"
    nt_label = NULL_LABELS[null_type]

    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]

    # Compute mean SP per task
    task_sp = {}
    for task in tasks:
        sp_vecs = []
        for g in pilot:
            if g.startswith(task + "/") and null_type in pilot[g]:
                sp_vecs.append(compute_sp(pilot[g][null_type]["z_scores"]))
        task_sp[task] = np.mean(sp_vecs, axis=0)

    # Similarity matrix
    n = len(tasks)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                a, b = task_sp[tasks[i]], task_sp[tasks[j]]
                if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
                    sim_matrix[i, j] = 0.0
                else:
                    sim_matrix[i, j] = 1.0 - cosine(a, b)

    # Condensed distance matrix for clustering
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(1.0 - sim_matrix[i, j])
    Z = linkage(np.array(dists), method="average")

    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2], wspace=0.05)

    # Dendrogram
    ax_dendro = fig.add_subplot(gs[0])
    dendro = dendrogram(Z, labels=tasks, orientation="left", ax=ax_dendro,
                        leaf_font_size=10, color_threshold=0.1)
    ax_dendro.set_xlabel("Cosine distance", fontsize=11)
    ax_dendro.set_title("Hierarchical clustering", fontsize=11)
    ax_dendro.spines["top"].set_visible(False)
    ax_dendro.spines["right"].set_visible(False)

    # Reorder similarity matrix by dendrogram
    order = dendro["leaves"]
    ordered_tasks = [tasks[i] for i in order]
    sim_ordered = sim_matrix[np.ix_(order, order)]

    ax_heat = fig.add_subplot(gs[1])
    vmin = sim_ordered[sim_ordered < 1.0].min() if np.any(sim_ordered < 1.0) else 0.0
    vmin = max(vmin - 0.02, 0.0)
    sns.heatmap(sim_ordered, ax=ax_heat, cmap="YlOrRd",
                vmin=vmin, vmax=1.0,
                xticklabels=ordered_tasks, yticklabels=ordered_tasks,
                annot=True, fmt=".3f", annot_kws={"size": 8},
                linewidths=0.5, square=True,
                cbar_kws={"label": "Cosine similarity", "shrink": 0.8})
    ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax_heat.set_yticklabels(ax_heat.get_yticklabels(), fontsize=9)
    ax_heat.set_title(f"Task similarity ({nt_label})", fontsize=11)

    fig.suptitle("Cross-task similarity of unrolled motif profiles",
                 fontsize=14, fontweight="bold", y=1.02)
    path = FIGURES_DIR / "fig_unrolled_task_similarity.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 4: Sign effect comparison ─────────────────────────────────

def fig_sign_effect(pilot: dict):
    """Grouped bar chart: LPC-shuf vs LPC-sign mean Z-scores."""
    motif_labels = [SHORT_NAMES_ONELINE[t] for t in TEMPLATE_NAMES]
    n_motifs = len(TEMPLATE_NAMES)

    z_shuf = np.zeros(n_motifs)
    z_sign = np.zeros(n_motifs)
    z_shuf_se = np.zeros(n_motifs)
    z_sign_se = np.zeros(n_motifs)

    for mi, mname in enumerate(TEMPLATE_NAMES):
        shuf_vals = [pilot[g]["layer_pair_config"]["z_scores"][mname] for g in pilot]
        sign_vals = [pilot[g]["layer_pair_config_signs"]["z_scores"][mname] for g in pilot]
        z_shuf[mi] = np.mean(shuf_vals)
        z_sign[mi] = np.mean(sign_vals)
        z_shuf_se[mi] = np.std(shuf_vals) / np.sqrt(len(shuf_vals))
        z_sign_se[mi] = np.std(sign_vals) / np.sqrt(len(sign_vals))

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_motifs)
    width = 0.35

    bars1 = ax.bar(x - width / 2, z_shuf, width, yerr=z_shuf_se,
                   label="LPC (shuffled signs)", color="#4393c3",
                   capsize=4, alpha=0.85, error_kw={"elinewidth": 1.2})
    bars2 = ax.bar(x + width / 2, z_sign, width, yerr=z_sign_se,
                   label="LPC (preserved signs)", color="#d6604d",
                   capsize=4, alpha=0.85, error_kw={"elinewidth": 1.2})

    # Threshold lines
    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Delta annotations
    for mi in range(n_motifs):
        delta = z_sign[mi] - z_shuf[mi]
        if abs(delta) > 0.5:
            y_pos = max(z_shuf[mi], z_sign[mi]) + max(z_shuf_se[mi], z_sign_se[mi]) + 1
            label = "TOPO" if delta > 0 else "SIGN"
            color = "#2166ac" if delta > 0 else "#b2182b"
            ax.annotate(f"{label}\n({delta:+.1f})",
                        xy=(x[mi], y_pos), ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(motif_labels, fontsize=9)
    ax.set_ylabel("Mean Z-score", fontsize=12)
    ax.set_title("Sign effect: topology vs sign coherence in unrolled motifs",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.95, fancybox=True, edgecolor="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_sign_effect.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 5: SP profiles per task ───────────────────────────────────

def fig_sp_profiles(pilot: dict):
    """Line plot of mean SP vectors per task for LPC-shuf."""
    null_type = "layer_pair_config"
    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]
    motif_labels = [SHORT_NAMES_ONELINE[t] for t in TEMPLATE_NAMES]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(TEMPLATE_NAMES))

    for task in tasks:
        sp_vecs = []
        for g in pilot:
            if g.startswith(task + "/") and null_type in pilot[g]:
                sp_vecs.append(compute_sp(pilot[g][null_type]["z_scores"]))
        mean_sp = np.mean(sp_vecs, axis=0)
        se_sp = np.std(sp_vecs, axis=0) / np.sqrt(len(sp_vecs))

        color = TASK_COLORS.get(task, "#333333")
        ax.plot(x, mean_sp, "o-", color=color, label=task, linewidth=1.5,
                markersize=5, alpha=0.85)
        ax.fill_between(x, mean_sp - se_sp, mean_sp + se_sp,
                        color=color, alpha=0.12)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean SP value", fontsize=12)
    ax.set_title(f"Significance profiles by task ({NULL_LABELS[null_type]})",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, ncol=3, framealpha=0.95, fancybox=True,
              edgecolor="#cccccc", loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_sp_profiles.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 6: Coherent FFL deep dive (the star motif) ────────────────

def fig_coherent_ffl_detail(pilot: dict):
    """Box plot of coherent FFL Z-scores per task across null models."""
    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]
    null_types_focus = ["layer_preserving_er", "layer_pair_config", "layer_pair_config_signs"]
    motif = "coherent_ffl"

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Coherent FFL enrichment by task and null model",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, nt in enumerate(null_types_focus):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for task in tasks:
            zs = [pilot[g][nt]["z_scores"][motif]
                  for g in pilot if g.startswith(task + "/") and nt in pilot[g]]
            data.append(zs)
            labels.append(task)
            colors.append(TASK_COLORS.get(task, "#333333"))

        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels,
                        widths=0.6, showfliers=True,
                        flierprops={"markersize": 3, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.4)
        ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.4)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(NULL_LABELS[nt], fontsize=11)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("Z-score", fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_coherent_ffl_detail.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 7: SP heatmaps (tasks x motifs) ───────────────────────────

def fig_sp_heatmaps(pilot: dict):
    """Side-by-side SP heatmaps for LP-ER, LPC-shuf, and LPC-sign."""
    null_types_to_show = ["layer_preserving_er", "layer_pair_config", "layer_pair_config_signs"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Mean significance profiles by task category (layer-aware null models)",
                 fontsize=14, fontweight="bold", y=1.02)

    motif_labels = [SHORT_NAMES_ONELINE[t] for t in TEMPLATE_NAMES]
    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]

    for idx, nt in enumerate(null_types_to_show):
        # Compute per-graph SP, then average per task
        sp_matrix = np.zeros((len(tasks), len(TEMPLATE_NAMES)))
        for ti, task in enumerate(tasks):
            sp_vecs = []
            for g in pilot:
                if g.startswith(task + "/") and nt in pilot[g]:
                    sp_vecs.append(compute_sp(pilot[g][nt]["z_scores"]))
            if sp_vecs:
                sp_matrix[ti, :] = np.mean(sp_vecs, axis=0)

        ax = axes[idx]
        vmax = np.max(np.abs(sp_matrix))
        vmax = max(vmax, 0.1)  # ensure a minimum range
        sns.heatmap(sp_matrix, ax=ax, cmap="RdBu_r", center=0,
                    vmin=-vmax, vmax=vmax,
                    xticklabels=motif_labels, yticklabels=tasks,
                    annot=True, fmt=".2f", annot_kws={"size": 7},
                    linewidths=0.5, cbar_kws={"label": "SP value", "shrink": 0.8})
        ax.set_title(NULL_LABELS[nt], fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_sp_heatmaps.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 8: Coherent FFL SP detail ─────────────────────────────────

def fig_coherent_ffl_sp_detail(pilot: dict):
    """Box plot of coherent FFL SP values per task across null models."""
    tasks = [t for t in TASK_ORDER if any(g.startswith(t + "/") for g in pilot)]
    null_types_focus = ["layer_preserving_er", "layer_pair_config", "layer_pair_config_signs"]
    motif_idx = TEMPLATE_NAMES.index("coherent_ffl")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    fig.suptitle("Coherent FFL significance profile by task and null model",
                 fontsize=14, fontweight="bold", y=1.02)

    for idx, nt in enumerate(null_types_focus):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for task in tasks:
            sp_vals = []
            for g in pilot:
                if g.startswith(task + "/") and nt in pilot[g]:
                    sp = compute_sp(pilot[g][nt]["z_scores"])
                    sp_vals.append(sp[motif_idx])
            data.append(sp_vals)
            labels.append(task)
            colors.append(TASK_COLORS.get(task, "#333333"))

        bp = ax.boxplot(data, patch_artist=True, tick_labels=labels,
                        widths=0.6, showfliers=True,
                        flierprops={"markersize": 3, "alpha": 0.5})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title(NULL_LABELS[nt], fontsize=11)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if idx == 0:
            ax.set_ylabel("SP value", fontsize=11)

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_coherent_ffl_sp_detail.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {path}")
    plt.close(fig)


# ── Figure 9: Graph-size confound diagnostic ─────────────────────────

def fig_size_confound(pilot: dict):
    """Scatter of coherent FFL Z/SP vs graph properties, colored by task.

    4-panel figure:
      A: Coherent FFL Z-score (LPC-shuf) vs edge count
      B: Coherent FFL SP (LPC-shuf) vs edge count
      C: Coherent FFL SP (LPC-shuf) vs density (edges / possible edges)
      D: Graph property distributions by task (edge count box plot)
    """
    from scipy.stats import spearmanr
    from src.graph_loader import load_attribution_graph

    null_type = "layer_pair_config"
    motif = "coherent_ffl"
    motif_idx = TEMPLATE_NAMES.index(motif)

    # Collect per-graph data: task, z, sp, node count, edge count
    rows = []
    data_dir = _REPO / "data" / "raw"
    for gname in pilot:
        if null_type not in pilot[gname]:
            continue
        task = gname.split("/")[0]
        z = pilot[gname][null_type]["z_scores"][motif]
        sp = compute_sp(pilot[gname][null_type]["z_scores"])[motif_idx]

        # Load graph to get size stats
        gpath = data_dir / (gname + ".json")
        if not gpath.exists():
            # Try without extension variations
            continue
        g = load_attribution_graph(gpath)
        n, m = g.vcount(), g.ecount()
        density = m / (n * (n - 1)) if n > 1 else 0

        rows.append({
            "gname": gname, "task": task,
            "z": z, "sp": sp,
            "nodes": n, "edges": m, "density": density,
        })

    print(f"  Loaded graph stats for {len(rows)}/{len(pilot)} graphs")

    tasks_present = sorted(set(r["task"] for r in rows))
    edges = np.array([r["edges"] for r in rows])
    density = np.array([r["density"] for r in rows])
    z_vals = np.array([r["z"] for r in rows])
    sp_vals = np.array([r["sp"] for r in rows])
    task_labels = [r["task"] for r in rows]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("Is coherent FFL enrichment driven by graph size?",
                 fontsize=14, fontweight="bold", y=1.01)

    # ── Panel A: Z-score vs edge count ────────────────────────────────
    ax = axes[0, 0]
    for task in tasks_present:
        mask = [t == task for t in task_labels]
        ax.scatter(edges[mask], z_vals[mask], c=TASK_COLORS.get(task, "#333"),
                   label=task, s=30, alpha=0.75, edgecolors="white", linewidths=0.3)
    rho, p = spearmanr(edges, z_vals)
    ax.set_xlabel("Edge count", fontsize=11)
    ax.set_ylabel("Coherent FFL Z-score (LPC-shuf)", fontsize=11)
    ax.set_title(f"A. Z-score vs edge count (Spearman r={rho:.2f}, p={p:.3f})",
                 fontsize=11)
    ax.axhline(y=2, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=-2, color="red", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: SP vs edge count ─────────────────────────────────────
    ax = axes[0, 1]
    for task in tasks_present:
        mask = [t == task for t in task_labels]
        ax.scatter(edges[mask], sp_vals[mask], c=TASK_COLORS.get(task, "#333"),
                   label=task, s=30, alpha=0.75, edgecolors="white", linewidths=0.3)
    rho_sp, p_sp = spearmanr(edges, sp_vals)
    ax.set_xlabel("Edge count", fontsize=11)
    ax.set_ylabel("Coherent FFL SP (LPC-shuf)", fontsize=11)
    ax.set_title(f"B. SP vs edge count (Spearman r={rho_sp:.2f}, p={p_sp:.3f})",
                 fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9, fancybox=True,
              edgecolor="#cccccc", loc="lower right")

    # ── Panel C: SP vs density ────────────────────────────────────────
    ax = axes[1, 0]
    for task in tasks_present:
        mask = [t == task for t in task_labels]
        ax.scatter(density[mask], sp_vals[mask], c=TASK_COLORS.get(task, "#333"),
                   label=task, s=30, alpha=0.75, edgecolors="white", linewidths=0.3)
    rho_d, p_d = spearmanr(density, sp_vals)
    ax.set_xlabel("Graph density (edges / possible edges)", fontsize=11)
    ax.set_ylabel("Coherent FFL SP (LPC-shuf)", fontsize=11)
    ax.set_title(f"C. SP vs density (Spearman r={rho_d:.2f}, p={p_d:.3f})",
                 fontsize=11)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel D: Edge count distribution by task ──────────────────────
    ax = axes[1, 1]
    task_edges = {task: [] for task in tasks_present}
    for r in rows:
        task_edges[r["task"]].append(r["edges"])

    ordered_tasks = [t for t in TASK_ORDER if t in task_edges]
    box_data = [task_edges[t] for t in ordered_tasks]
    box_colors = [TASK_COLORS.get(t, "#333") for t in ordered_tasks]

    bp = ax.boxplot(box_data, patch_artist=True, tick_labels=ordered_tasks,
                    widths=0.6, showfliers=True,
                    flierprops={"markersize": 3, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.set_ylabel("Edge count", fontsize=11)
    ax.set_title("D. Graph size distribution by task", fontsize=11)
    ax.set_xticklabels(ordered_tasks, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Print summary stats
    print("\n  Graph size summary by task:")
    print(f"  {'Task':15s} {'N':>4s} {'med edges':>10s} {'med nodes':>10s} {'med density':>12s} {'med FFL SP':>10s}")
    for task in ordered_tasks:
        t_rows = [r for r in rows if r["task"] == task]
        med_e = np.median([r["edges"] for r in t_rows])
        med_n = np.median([r["nodes"] for r in t_rows])
        med_d = np.median([r["density"] for r in t_rows])
        med_sp = np.median([r["sp"] for r in t_rows])
        print(f"  {task:15s} {len(t_rows):4d} {med_e:10.0f} {med_n:10.0f} {med_d:12.4f} {med_sp:+10.3f}")

    print(f"\n  Overall correlations:")
    print(f"    FFL Z  vs edges:   Spearman r={rho:.3f}, p={p:.4f}")
    print(f"    FFL SP vs edges:   Spearman r={rho_sp:.3f}, p={p_sp:.4f}")
    print(f"    FFL SP vs density: Spearman r={rho_d:.3f}, p={p_d:.4f}")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_size_confound.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved {path}")
    plt.close(fig)


# ── Figure 10: Density-controlled FFL SP ──────────────────────────────

def fig_density_controlled(pilot: dict):
    """Raw vs density-adjusted coherent FFL SP by task.

    3-panel figure:
      A: Raw coherent FFL SP by task (box plot)
      B: Density-adjusted SP by task (residuals from SP ~ density regression)
      C: Scatter of residual SP vs density (confirming confound is removed)

    Also runs Kruskal-Wallis on both raw and adjusted values to test whether
    cross-task variation survives density correction.
    """
    from scipy.stats import spearmanr, kruskal as kruskal_test
    from src.graph_loader import load_attribution_graph

    null_type = "layer_pair_config"
    motif_idx = TEMPLATE_NAMES.index("coherent_ffl")

    # Collect per-graph data
    rows = []
    data_dir = _REPO / "data" / "raw"
    for gname in pilot:
        if null_type not in pilot[gname]:
            continue
        task = gname.split("/")[0]
        sp = compute_sp(pilot[gname][null_type]["z_scores"])[motif_idx]
        gpath = data_dir / (gname + ".json")
        if not gpath.exists():
            continue
        g = load_attribution_graph(gpath)
        n, m = g.vcount(), g.ecount()
        density = m / (n * (n - 1)) if n > 1 else 0
        rows.append({"gname": gname, "task": task, "sp": sp, "density": density,
                      "edges": m, "nodes": n})

    sp_arr = np.array([r["sp"] for r in rows])
    dens_arr = np.array([r["density"] for r in rows])
    task_arr = [r["task"] for r in rows]

    # Linear regression: SP ~ density (OLS)
    # residual = SP - (a * density + b)
    coeffs = np.polyfit(dens_arr, sp_arr, 1)
    sp_predicted = np.polyval(coeffs, dens_arr)
    sp_residual = sp_arr - sp_predicted

    ordered_tasks = [t for t in TASK_ORDER if t in set(task_arr)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Coherent FFL enrichment: raw vs density-controlled (LPC shuffled signs)",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Panel A: Raw SP ───────────────────────────────────────────────
    ax = axes[0]
    raw_data = []
    raw_colors = []
    for task in ordered_tasks:
        vals = [sp_arr[i] for i in range(len(rows)) if task_arr[i] == task]
        raw_data.append(vals)
        raw_colors.append(TASK_COLORS.get(task, "#333"))

    bp = ax.boxplot(raw_data, patch_artist=True, tick_labels=ordered_tasks,
                    widths=0.6, showfliers=True,
                    flierprops={"markersize": 3, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], raw_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Kruskal-Wallis on raw
    kw_groups_raw = [np.array(d) for d in raw_data if len(d) >= 1]
    h_raw, p_raw = kruskal_test(*kw_groups_raw)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Coherent FFL SP", fontsize=11)
    ax.set_title(f"A. Raw SP\nKruskal-Wallis H={h_raw:.1f}, p={p_raw:.4f}",
                 fontsize=11)
    ax.set_xticklabels(ordered_tasks, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel B: Density-adjusted SP ──────────────────────────────────
    ax = axes[1]
    adj_data = []
    for task in ordered_tasks:
        vals = [sp_residual[i] for i in range(len(rows)) if task_arr[i] == task]
        adj_data.append(vals)

    bp = ax.boxplot(adj_data, patch_artist=True, tick_labels=ordered_tasks,
                    widths=0.6, showfliers=True,
                    flierprops={"markersize": 3, "alpha": 0.5})
    for patch, color in zip(bp["boxes"], raw_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    # Kruskal-Wallis on adjusted
    kw_groups_adj = [np.array(d) for d in adj_data if len(d) >= 1]
    h_adj, p_adj = kruskal_test(*kw_groups_adj)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Density-adjusted SP (residual)", fontsize=11)
    ax.set_title(f"B. Density-controlled SP\nKruskal-Wallis H={h_adj:.1f}, p={p_adj:.4f}",
                 fontsize=11)
    ax.set_xticklabels(ordered_tasks, rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel C: Residual vs density (confound removed?) ──────────────
    ax = axes[2]
    for task in ordered_tasks:
        mask = [task_arr[i] == task for i in range(len(rows))]
        ax.scatter(dens_arr[mask], sp_residual[mask],
                   c=TASK_COLORS.get(task, "#333"),
                   label=task, s=30, alpha=0.75,
                   edgecolors="white", linewidths=0.3)

    rho_res, p_res = spearmanr(dens_arr, sp_residual)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xlabel("Graph density", fontsize=11)
    ax.set_ylabel("Residual SP", fontsize=11)
    ax.set_title(f"C. Confound check\nSpearman r={rho_res:.3f}, p={p_res:.3f}",
                 fontsize=11)
    ax.legend(fontsize=7, ncol=3, framealpha=0.9, fancybox=True,
              edgecolor="#cccccc", loc="lower left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Print summary
    print(f"\n  Density control for coherent FFL SP (LPC-shuf):")
    print(f"    Regression: SP = {coeffs[0]:.3f} * density + {coeffs[1]:.3f}")
    print(f"    Raw Kruskal-Wallis:      H={h_raw:.2f}, p={p_raw:.6f}")
    print(f"    Adjusted Kruskal-Wallis: H={h_adj:.2f}, p={p_adj:.6f}")
    print(f"    Residual vs density:     Spearman r={rho_res:.4f}, p={p_res:.4f}")

    # Per-task medians
    print(f"\n  {'Task':15s} {'N':>4s} {'raw SP':>8s} {'adj SP':>8s} {'density':>8s}")
    for ti, task in enumerate(ordered_tasks):
        n = len(raw_data[ti])
        med_raw = np.median(raw_data[ti])
        med_adj = np.median(adj_data[ti])
        task_dens = np.median([dens_arr[i] for i in range(len(rows)) if task_arr[i] == task])
        print(f"  {task:15s} {n:4d} {med_raw:+8.3f} {med_adj:+8.3f} {task_dens:8.4f}")

    plt.tight_layout()
    path = FIGURES_DIR / "fig_unrolled_density_controlled.png"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved {path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(exist_ok=True)
    pilot, analysis = load_data()
    print(f"Loaded {len(pilot)} graphs")

    fig_universal_enrichment(pilot)
    fig_zscore_heatmaps(pilot)
    fig_task_similarity(pilot)
    fig_sign_effect(pilot)
    fig_sp_profiles(pilot)
    fig_coherent_ffl_detail(pilot)
    fig_sp_heatmaps(pilot)
    fig_coherent_ffl_sp_detail(pilot)
    fig_size_confound(pilot)
    fig_density_controlled(pilot)

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
