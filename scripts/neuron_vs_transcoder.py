"""Paired comparison of neuron-level vs transcoder-level motif signatures.

For the 8 prompts with both graph types, computes:
1. Spearman correlation of Z-score profiles per prompt
2. Cosine similarity of SP vectors per prompt
3. Paired Wilcoxon signed-rank per motif across 8 prompts
4. Cohen's d effect size of neuron-vs-transcoder Z-score differences

Generates 3 figures:
- fig_neuron_vs_transcoder_zscores.png  — side-by-side Z-score heatmaps
- fig_neuron_vs_transcoder_scatter.png  — mean Z scatter per motif
- fig_neuron_vs_transcoder_properties.png — paired bar charts of graph properties

Usage:
    python scripts/neuron_vs_transcoder.py
    python scripts/neuron_vs_transcoder.py --neuron-results data/results/neuron_motif_pilot/pilot_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import stats

matplotlib.use("Agg")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ── Configuration ─────────────────────────────────────────────────────

# Mapping from neuron graph slug to transcoder graph name
# (category/stem in unrolled_null_pilot results)
MATCHED_PROMPTS = {
    "arithmetic/count-by-sevens":      "arithmetic/count-by-sevens-clt-clean",
    "arithmetic/five-plus-three":      "arithmetic/five-plus-three-clt-clean",
    "factual_recall/capital-france":    "factual_recall/opposite_of_small",
    "factual_recall/opposite-small":    "factual_recall/opposite_of_small",
    "multihop/capital-state-dallas":    "multihop/capital-analogy-clt-clean",
    "multihop/currency-france":         "multihop/capital-analogy-clt-18l",
    "reasoning/medical-diagnosis":      "reasoning/sally-school-clt-clean",
    "reasoning/sally-school":           "reasoning/sally-school-clt-clean",
}

MOTIF_NAMES = [
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
    "cross_chain_inhibition": "XChainInhib",
    "feedforward_damping": "FF_Damp",
    "feedforward_amplification": "FF_Amp",
    "residual_self_loop_positive": "SelfLoop+",
    "residual_self_loop_negative": "SelfLoop-",
    "coherent_ffl": "CoherFFL",
    "incoherent_ffl": "IncoherFFL",
    "cross_chain_toggle": "XChainTog",
}

# Which null model to use for the primary comparison
PRIMARY_NULL = "layer_pair_config"


# ── Analysis functions ────────────────────────────────────────────────

def extract_zscores(results: dict, graph_name: str, null_type: str) -> np.ndarray:
    """Extract Z-score vector for a graph from results dict."""
    zs = results[graph_name][null_type]["z_scores"]
    return np.array([zs[m] for m in MOTIF_NAMES])


def compute_sp(z_scores: np.ndarray) -> np.ndarray:
    """Compute Significance Profile from Z-score vector."""
    norm = np.sqrt(np.sum(z_scores ** 2))
    if norm < 1e-10:
        return np.zeros_like(z_scores)
    return z_scores / norm


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cohen's d for paired samples."""
    diff = x - y
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-10:
        return 0.0
    return float(mean_diff / std_diff)


# ── Figure: Side-by-side Z-score heatmaps ────────────────────────────

def plot_zscores_heatmap(
    neuron_z: np.ndarray,
    transcoder_z: np.ndarray,
    prompt_labels: list[str],
    output_path: Path,
) -> None:
    """Side-by-side Z-score heatmaps for neuron vs transcoder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    motif_labels = [SHORT_NAMES[m] for m in MOTIF_NAMES]

    # Shared color scale
    vmax = max(np.abs(neuron_z).max(), np.abs(transcoder_z).max())
    vmin = -vmax

    im1 = ax1.imshow(neuron_z, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax1.set_title("Neuron-Level Z-scores", fontsize=12, fontweight="bold")
    ax1.set_xticks(range(len(motif_labels)))
    ax1.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=9)
    ax1.set_yticks(range(len(prompt_labels)))
    ax1.set_yticklabels(prompt_labels, fontsize=9)

    # Annotate cells
    for i in range(neuron_z.shape[0]):
        for j in range(neuron_z.shape[1]):
            val = neuron_z[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax1.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=7, color=color)

    im2 = ax2.imshow(transcoder_z, aspect="auto", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax2.set_title("Transcoder-Level Z-scores", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(len(motif_labels)))
    ax2.set_xticklabels(motif_labels, rotation=45, ha="right", fontsize=9)

    for i in range(transcoder_z.shape[0]):
        for j in range(transcoder_z.shape[1]):
            val = transcoder_z[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                     fontsize=7, color=color)

    fig.colorbar(im2, ax=[ax1, ax2], label="Z-score", shrink=0.8)
    fig.suptitle(
        f"Neuron vs Transcoder Motif Signatures ({PRIMARY_NULL} null)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure: Mean Z scatter ───────────────────────────────────────────

def plot_mean_z_scatter(
    neuron_z: np.ndarray,
    transcoder_z: np.ndarray,
    output_path: Path,
) -> None:
    """Scatter plot of mean Z-score per motif (neuron vs transcoder)."""
    mean_neuron = neuron_z.mean(axis=0)
    mean_transcoder = transcoder_z.mean(axis=0)
    motif_labels = [SHORT_NAMES[m] for m in MOTIF_NAMES]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(mean_transcoder, mean_neuron, s=100, c="steelblue", zorder=5)

    for i, label in enumerate(motif_labels):
        ax.annotate(label, (mean_transcoder[i], mean_neuron[i]),
                     textcoords="offset points", xytext=(8, 5), fontsize=9)

    # Identity line
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.3, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Correlation
    r, p = stats.spearmanr(mean_transcoder, mean_neuron)
    ax.text(
        0.05, 0.95,
        f"Spearman r = {r:.3f} (p = {p:.3f})",
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("Mean Transcoder Z-score", fontsize=12)
    ax.set_ylabel("Mean Neuron Z-score", fontsize=12)
    ax.set_title(
        f"Neuron vs Transcoder: Mean Z per Motif ({PRIMARY_NULL})",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Figure: Graph property comparison ─────────────────────────────────

def plot_properties(
    neuron_chars: dict,
    transcoder_chars: dict,
    output_path: Path,
) -> None:
    """Paired bar charts of graph properties."""
    properties = ["n_nodes", "n_edges", "density", "degree_gini", "excitatory_fraction"]
    prop_labels = ["Nodes", "Edges", "Density", "Degree Gini", "Excitatory %"]

    fig, axes = plt.subplots(1, len(properties), figsize=(18, 5))

    for ax, prop, label in zip(axes, properties, prop_labels):
        neuron_vals = [neuron_chars[g][prop] for g in sorted(neuron_chars)]
        trans_vals = [transcoder_chars[g][prop] for g in sorted(transcoder_chars)]

        x = np.arange(len(neuron_vals))
        width = 0.35

        ax.bar(x - width / 2, neuron_vals, width, label="Neuron", color="steelblue", alpha=0.8)
        ax.bar(x + width / 2, trans_vals, width, label="Transcoder", color="coral", alpha=0.8)

        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{i+1}" for i in range(len(x))], fontsize=8)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("Graph Properties: Neuron vs Transcoder", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Paired comparison: neuron vs transcoder motif signatures"
    )
    parser.add_argument(
        "--neuron-results", type=str,
        default=str(_REPO / "data" / "results" / "neuron_motif_pilot" / "pilot_results.json"),
        help="Path to neuron motif results JSON.",
    )
    parser.add_argument(
        "--transcoder-results", type=str,
        default=str(_REPO / "data" / "results" / "unrolled_null_pilot" / "pilot_results.json"),
        help="Path to transcoder motif results JSON.",
    )
    parser.add_argument(
        "--neuron-chars", type=str,
        default=str(_REPO / "data" / "results" / "neuron_motif_pilot" / "characterization.json"),
        help="Path to neuron characterization JSON.",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(_REPO / "figures"),
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--null-type", type=str, default=PRIMARY_NULL,
        help=f"Null model type for comparison. Default: {PRIMARY_NULL}",
    )
    args = parser.parse_args()

    null_type = args.null_type
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print("Loading results...", flush=True)
    with open(args.neuron_results) as f:
        neuron_results = json.load(f)
    print(f"  Neuron: {len(neuron_results)} graphs", flush=True)

    with open(args.transcoder_results) as f:
        transcoder_results = json.load(f)
    print(f"  Transcoder: {len(transcoder_results)} graphs", flush=True)

    # Find matched pairs
    matched = []
    for n_name, t_name in MATCHED_PROMPTS.items():
        if n_name in neuron_results and t_name in transcoder_results:
            if null_type in neuron_results[n_name] and null_type in transcoder_results[t_name]:
                matched.append((n_name, t_name))

    if not matched:
        print("\nERROR: No matched pairs found. Check that both result files exist "
              "and contain the expected graph names.", flush=True)
        print(f"\nNeuron graphs: {list(neuron_results.keys())}", flush=True)
        print(f"Transcoder graphs: {list(transcoder_results.keys())[:20]}...", flush=True)
        print(f"\nExpected matches:", flush=True)
        for n, t in MATCHED_PROMPTS.items():
            n_ok = n in neuron_results
            t_ok = t in transcoder_results
            print(f"  {n:40s} → {t:45s} [N:{'Y' if n_ok else 'N'} T:{'Y' if t_ok else 'N'}]")
        sys.exit(1)

    print(f"\nFound {len(matched)} matched pairs:", flush=True)
    for n_name, t_name in matched:
        print(f"  {n_name} <-> {t_name}", flush=True)

    # Extract Z-score matrices
    n_prompts = len(matched)
    n_motifs = len(MOTIF_NAMES)
    neuron_z = np.zeros((n_prompts, n_motifs))
    transcoder_z = np.zeros((n_prompts, n_motifs))
    prompt_labels = []

    for i, (n_name, t_name) in enumerate(matched):
        neuron_z[i] = extract_zscores(neuron_results, n_name, null_type)
        transcoder_z[i] = extract_zscores(transcoder_results, t_name, null_type)
        prompt_labels.append(n_name.split("/")[1])

    # ── Analysis 1: Spearman correlation per prompt ──────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"SPEARMAN CORRELATION per prompt ({null_type})", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Prompt':25s} {'rho':>8s} {'p-value':>10s}", flush=True)

    rhos = []
    for i, label in enumerate(prompt_labels):
        r, p = stats.spearmanr(neuron_z[i], transcoder_z[i])
        rhos.append(r)
        sig = "**" if p < 0.01 else "* " if p < 0.05 else "  "
        print(f"  {label:25s} {r:+8.3f} {p:>10.4f} {sig}", flush=True)

    mean_rho = np.mean(rhos)
    print(f"  {'Mean':25s} {mean_rho:+8.3f}", flush=True)

    # ── Analysis 2: Cosine similarity per prompt ─────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"COSINE SIMILARITY of SP vectors", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"  {'Prompt':25s} {'cosine':>8s}", flush=True)

    cosines = []
    for i, label in enumerate(prompt_labels):
        sp_n = compute_sp(neuron_z[i])
        sp_t = compute_sp(transcoder_z[i])
        cos = float(np.dot(sp_n, sp_t))
        cosines.append(cos)
        print(f"  {label:25s} {cos:+8.3f}", flush=True)

    mean_cos = np.mean(cosines)
    print(f"  {'Mean':25s} {mean_cos:+8.3f}", flush=True)

    # ── Analysis 3: Paired Wilcoxon per motif ────────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"PAIRED WILCOXON SIGNED-RANK per motif (N={n_prompts} prompts)", flush=True)
    print(f"{'='*80}", flush=True)
    print(
        f"  {'Motif':15s} {'mean_N':>8s} {'mean_T':>8s} {'W':>8s} {'p':>10s} {'d':>8s}",
        flush=True,
    )

    for j, motif in enumerate(MOTIF_NAMES):
        n_vals = neuron_z[:, j]
        t_vals = transcoder_z[:, j]
        mean_n = np.mean(n_vals)
        mean_t = np.mean(t_vals)
        d = cohens_d(n_vals, t_vals)

        # Wilcoxon requires non-zero differences
        diffs = n_vals - t_vals
        if np.all(np.abs(diffs) < 1e-10):
            w_stat, p_val = 0.0, 1.0
        elif len(diffs[np.abs(diffs) > 1e-10]) < 2:
            w_stat, p_val = 0.0, 1.0
        else:
            w_stat, p_val = stats.wilcoxon(n_vals, t_vals)

        sn = SHORT_NAMES[motif]
        sig = "**" if p_val < 0.01 else "* " if p_val < 0.05 else "  "
        print(
            f"  {sn:15s} {mean_n:+8.1f} {mean_t:+8.1f} {w_stat:>8.0f} "
            f"{p_val:>10.4f} {d:+8.2f} {sig}",
            flush=True,
        )

    # ── Analysis 4: Overall summary ──────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print("KEY PREDICTIONS", flush=True)
    print(f"{'='*80}", flush=True)

    # Coherent FFL enriched?
    cffl_n = neuron_z[:, MOTIF_NAMES.index("coherent_ffl")]
    cffl_enriched = np.mean(cffl_n > 2.0)
    print(f"  Coherent FFL enriched (z>2) at neuron level: "
          f"{cffl_enriched*100:.0f}% of prompts", flush=True)
    if cffl_enriched > 0.5:
        print("    -> CONFIRMED: Motif structure is fundamental", flush=True)
    else:
        print("    -> NOT CONFIRMED: May be feature decomposition artifact", flush=True)

    # Incoherent FFL depleted?
    iffl_n = neuron_z[:, MOTIF_NAMES.index("incoherent_ffl")]
    iffl_depleted = np.mean(iffl_n < -2.0)
    print(f"  Incoherent FFL depleted (z<-2) at neuron level: "
          f"{iffl_depleted*100:.0f}% of prompts", flush=True)

    # Effect sizes
    all_d = [cohens_d(neuron_z[:, j], transcoder_z[:, j]) for j in range(n_motifs)]
    print(f"  Mean |Cohen's d| across motifs: {np.mean(np.abs(all_d)):.2f}", flush=True)

    # ── Figures ───────────────────────────────────────────────────────
    print(f"\nGenerating figures...", flush=True)

    plot_zscores_heatmap(
        neuron_z, transcoder_z, prompt_labels,
        output_dir / "fig_neuron_vs_transcoder_zscores.png",
    )

    plot_mean_z_scatter(
        neuron_z, transcoder_z,
        output_dir / "fig_neuron_vs_transcoder_scatter.png",
    )

    # Property comparison (if characterization available)
    if Path(args.neuron_chars).exists():
        with open(args.neuron_chars) as f:
            neuron_chars = json.load(f)

        # Build matched property dicts
        matched_neuron_chars = {}
        matched_trans_chars = {}
        for i, (n_name, t_name) in enumerate(matched):
            label = f"P{i+1}"
            if n_name in neuron_chars:
                matched_neuron_chars[label] = neuron_chars[n_name]
            # For transcoder, compute from the graph
            t_path = _REPO / "data" / "raw" / (t_name + ".json")
            if t_path.exists():
                with open(t_path) as f:
                    t_json = json.load(f)
                from src.neuron_graph import characterize_graph
                matched_trans_chars[label] = characterize_graph(t_json)

        if matched_neuron_chars and matched_trans_chars:
            plot_properties(
                matched_neuron_chars, matched_trans_chars,
                output_dir / "fig_neuron_vs_transcoder_properties.png",
            )
        else:
            print("  Skipping property figure (missing characterization data)", flush=True)
    else:
        print("  Skipping property figure (no neuron characterization file)", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
