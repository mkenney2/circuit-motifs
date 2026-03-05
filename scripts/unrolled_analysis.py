"""Cross-task comparison analysis for unrolled motif Z-scores.

Loads pilot_results.json, computes significance profiles (SP vectors),
groups by task category, and runs the full comparison suite:
  - Cosine similarity matrix between task categories
  - Kruskal-Wallis test per motif across all tasks
  - Pairwise Mann-Whitney U between task pairs per motif
  - Hierarchical clustering of task categories
  - Sign-effect analysis (LPC-shuf vs LPC-sign)

Runs separately for each of the 5 null model types.

Usage:
    python scripts/unrolled_analysis.py
    python scripts/unrolled_analysis.py --results-file data/results/unrolled_null_pilot/pilot_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import kruskal, mannwhitneyu
from scipy.cluster.hierarchy import linkage, leaves_list

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

# ── Constants ─────────────────────────────────────────────────────────

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
    "cross_chain_inhibition": "XChainInhib",
    "feedforward_damping": "FF_Damp",
    "feedforward_amplification": "FF_Amp",
    "residual_self_loop_positive": "SelfLoop+",
    "residual_self_loop_negative": "SelfLoop-",
    "coherent_ffl": "CoherFFL",
    "incoherent_ffl": "IncoherFFL",
    "cross_chain_toggle": "XChainTog",
}

NULL_TYPES = [
    "configuration",
    "erdos_renyi",
    "layer_preserving_er",
    "layer_pair_config",
    "layer_pair_config_signs",
]

NULL_SHORT = {
    "configuration": "config",
    "erdos_renyi": "ER",
    "layer_preserving_er": "LP-ER",
    "layer_pair_config": "LPC-shuf",
    "layer_pair_config_signs": "LPC-sign",
}


# ── SP computation ────────────────────────────────────────────────────

def z_to_sp(z_dict: dict[str, float]) -> np.ndarray:
    """Convert a Z-score dict to a significance profile (SP) vector.

    SP_i = Z_i / sqrt(sum(Z_j^2)).  If all Z-scores are zero, returns
    a zero vector.
    """
    z_vec = np.array([z_dict[t] for t in TEMPLATE_NAMES])
    norm = np.sqrt(np.sum(z_vec ** 2))
    if norm < 1e-10:
        return np.zeros_like(z_vec)
    return z_vec / norm


# ── Task profile building ────────────────────────────────────────────

def build_profiles(
    results: dict,
    null_type: str,
) -> dict[str, dict]:
    """Build per-task profiles from pilot results for a given null type.

    Returns dict mapping task_name to:
        sp_vectors: list of SP vectors (one per graph)
        z_vectors: list of Z-score vectors
        mean_sp, std_sp, mean_z: aggregated stats
        n_graphs: count
    """
    task_data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"sp_vectors": [], "z_vectors": [], "graph_names": []}
    )

    for gname, null_results in results.items():
        task = gname.split("/")[0]
        if null_type not in null_results:
            continue
        z_dict = null_results[null_type]["z_scores"]
        z_vec = np.array([z_dict[t] for t in TEMPLATE_NAMES])
        sp_vec = z_to_sp(z_dict)
        task_data[task]["sp_vectors"].append(sp_vec)
        task_data[task]["z_vectors"].append(z_vec)
        task_data[task]["graph_names"].append(gname)

    profiles = {}
    for task, data in task_data.items():
        sp_arr = np.array(data["sp_vectors"])
        z_arr = np.array(data["z_vectors"])
        profiles[task] = {
            "sp_vectors": data["sp_vectors"],
            "z_vectors": data["z_vectors"],
            "graph_names": data["graph_names"],
            "mean_sp": sp_arr.mean(axis=0),
            "std_sp": sp_arr.std(axis=0),
            "mean_z": z_arr.mean(axis=0),
            "std_z": z_arr.std(axis=0),
            "n_graphs": len(data["sp_vectors"]),
        }

    return profiles


# ── Cosine similarity matrix ─────────────────────────────────────────

def cosine_sim_matrix(profiles: dict) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity between task mean SP vectors."""
    tasks = sorted(profiles.keys())
    n = len(tasks)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim[i, j] = 1.0
            else:
                a = profiles[tasks[i]]["mean_sp"]
                b = profiles[tasks[j]]["mean_sp"]
                if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
                    sim[i, j] = 0.0
                else:
                    sim[i, j] = 1.0 - cosine(a, b)
    return sim, tasks


# ── Kruskal-Wallis per motif ─────────────────────────────────────────

def kruskal_per_motif(profiles: dict) -> list[dict]:
    """Run Kruskal-Wallis test per motif across task categories."""
    tasks = sorted(profiles.keys())
    results = []
    for mi, mname in enumerate(TEMPLATE_NAMES):
        groups = []
        for task in tasks:
            sp_arr = np.array(profiles[task]["sp_vectors"])
            groups.append(sp_arr[:, mi])

        valid = [g for g in groups if len(g) >= 1]
        if len(valid) < 2:
            results.append({"motif": mname, "H": np.nan, "p": np.nan, "sig": False})
            continue
        try:
            h, p = kruskal(*valid)
            results.append({"motif": mname, "H": float(h), "p": float(p), "sig": bool(p < 0.05)})
        except ValueError:
            results.append({"motif": mname, "H": np.nan, "p": np.nan, "sig": False})
    return results


# ── Pairwise Mann-Whitney U ──────────────────────────────────────────

def pairwise_mannwhitney(profiles: dict, alpha: float = 0.05) -> list[dict]:
    """Run pairwise Mann-Whitney U per motif between all task pairs."""
    tasks = sorted(profiles.keys())
    comparisons = []
    for i, ta in enumerate(tasks):
        for tb in tasks[i + 1:]:
            sp_a = np.array(profiles[ta]["sp_vectors"])
            sp_b = np.array(profiles[tb]["sp_vectors"])
            cos_sim = 1.0 - cosine(
                profiles[ta]["mean_sp"], profiles[tb]["mean_sp"]
            )
            p_values = np.ones(len(TEMPLATE_NAMES))
            sig_motifs = []
            for mi in range(len(TEMPLATE_NAMES)):
                va, vb = sp_a[:, mi], sp_b[:, mi]
                if len(va) < 2 or len(vb) < 2:
                    continue
                if np.std(va) == 0 and np.std(vb) == 0:
                    continue
                try:
                    _, p = mannwhitneyu(va, vb, alternative="two-sided")
                    p_values[mi] = p
                    if p < alpha:
                        sig_motifs.append(TEMPLATE_NAMES[mi])
                except ValueError:
                    pass
            comparisons.append({
                "task_a": ta,
                "task_b": tb,
                "cosine_similarity": float(cos_sim),
                "n_significant_motifs": len(sig_motifs),
                "significant_motifs": sig_motifs,
                "p_values": {TEMPLATE_NAMES[mi]: float(p_values[mi])
                             for mi in range(len(TEMPLATE_NAMES))},
            })
    return comparisons


# ── Hierarchical clustering ──────────────────────────────────────────

def cluster_tasks(profiles: dict) -> tuple[np.ndarray, list[str]]:
    """Hierarchical clustering on task mean SP vectors (cosine distance)."""
    tasks = sorted(profiles.keys())
    n = len(tasks)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            a = profiles[tasks[i]]["mean_sp"]
            b = profiles[tasks[j]]["mean_sp"]
            if np.linalg.norm(a) < 1e-10 or np.linalg.norm(b) < 1e-10:
                dists.append(1.0)
            else:
                dists.append(cosine(a, b))
    dist_arr = np.array(dists)
    dist_arr = np.nan_to_num(dist_arr, nan=1.0)
    Z = linkage(dist_arr, method="average")
    return Z, tasks


# ── Printing ──────────────────────────────────────────────────────────

def print_section(title: str, width: int = 100):
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def print_profiles(profiles: dict, null_type: str):
    """Print mean Z-scores and SP vectors per task."""
    tasks = sorted(profiles.keys())
    nt_short = NULL_SHORT[null_type]

    print_section(f"TASK PROFILES -- {nt_short} null model")

    # Mean Z-scores
    print(f"\n  Mean Z-scores per task ({nt_short}):")
    hdr = f"  {'Task':15s} {'N':>4s}"
    for t in TEMPLATE_NAMES:
        hdr += f"  {SHORT_NAMES[t]:>12s}"
    print(hdr)

    for task in tasks:
        p = profiles[task]
        line = f"  {task:15s} {p['n_graphs']:4d}"
        for mi, t in enumerate(TEMPLATE_NAMES):
            z = p["mean_z"][mi]
            marker = "**" if abs(z) > 2 else "  "
            line += f"  {z:+9.1f} {marker}"
        print(line)

    # Mean SP vectors
    print(f"\n  Mean SP vectors per task ({nt_short}):")
    hdr = f"  {'Task':15s}"
    for t in TEMPLATE_NAMES:
        hdr += f"  {SHORT_NAMES[t]:>12s}"
    print(hdr)

    for task in tasks:
        p = profiles[task]
        line = f"  {task:15s}"
        for mi in range(len(TEMPLATE_NAMES)):
            sp = p["mean_sp"][mi]
            line += f"  {sp:+12.3f}"
        print(line)


def print_similarity(sim_matrix: np.ndarray, tasks: list[str], null_type: str):
    """Print cosine similarity matrix."""
    nt_short = NULL_SHORT[null_type]
    print(f"\n  Cosine similarity matrix ({nt_short}):")
    hdr = f"  {'':15s}"
    for t in tasks:
        hdr += f"  {t[:8]:>8s}"
    print(hdr)
    for i, ta in enumerate(tasks):
        line = f"  {ta:15s}"
        for j in range(len(tasks)):
            line += f"  {sim_matrix[i, j]:8.3f}"
        print(line)

    # Most/least similar pairs
    pairs = []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            pairs.append((sim_matrix[i, j], tasks[i], tasks[j]))
    pairs.sort(reverse=True)
    print(f"\n  Most similar:  {pairs[0][1]} / {pairs[0][2]} ({pairs[0][0]:.4f})")
    print(f"  Least similar: {pairs[-1][1]} / {pairs[-1][2]} ({pairs[-1][0]:.4f})")


def print_kruskal(kw_results: list[dict], null_type: str):
    """Print Kruskal-Wallis results."""
    nt_short = NULL_SHORT[null_type]
    print(f"\n  Kruskal-Wallis test per motif ({nt_short}):")
    print(f"  {'Motif':15s} {'H':>8s} {'p-value':>10s} {'sig':>5s}")
    for r in kw_results:
        sn = SHORT_NAMES.get(r["motif"], r["motif"][:15])
        sig = "**" if r["sig"] else ""
        if np.isnan(r["H"]):
            print(f"  {sn:15s} {'N/A':>8s} {'N/A':>10s}")
        else:
            print(f"  {sn:15s} {r['H']:8.2f} {r['p']:10.4f}   {sig}")
    n_sig = sum(1 for r in kw_results if r["sig"])
    print(f"  => {n_sig}/{len(kw_results)} motifs show significant cross-task variation")


def print_pairwise(comparisons: list[dict], null_type: str):
    """Print pairwise comparison summary."""
    nt_short = NULL_SHORT[null_type]
    print(f"\n  Pairwise task comparisons ({nt_short}):")
    print(f"  {'Task A':15s} {'Task B':15s} {'cos_sim':>8s} {'#sig':>5s}  significant motifs")

    comparisons_sorted = sorted(comparisons, key=lambda c: c["cosine_similarity"])
    for c in comparisons_sorted:
        sig_short = [SHORT_NAMES.get(m, m[:10]) for m in c["significant_motifs"]]
        print(
            f"  {c['task_a']:15s} {c['task_b']:15s} "
            f"{c['cosine_similarity']:8.4f} {c['n_significant_motifs']:5d}  "
            f"{', '.join(sig_short) if sig_short else '--'}"
        )


def print_sign_effect(results: dict):
    """Compare LPC-shuf vs LPC-sign across all graphs."""
    print_section("SIGN EFFECT ANALYSIS: LPC-shuf vs LPC-sign")
    print("  Comparing Z-scores with shuffled signs vs preserved signs.")
    print("  Positive delta -> topology alone enriches (sign structure irrelevant)")
    print("  Negative delta -> sign coherence drives enrichment")

    # Gather per-graph Z-scores for both null types
    print(f"\n  {'Motif':15s} {'shuf_z':>8s} {'sign_z':>8s} {'delta':>8s}  "
          f"{'%agree':>7s}  interpretation")

    for mi, mname in enumerate(TEMPLATE_NAMES):
        sn = SHORT_NAMES[mname]
        z_shuf = []
        z_sign = []
        for gname in results:
            if "layer_pair_config" in results[gname] and "layer_pair_config_signs" in results[gname]:
                z_shuf.append(results[gname]["layer_pair_config"]["z_scores"][mname])
                z_sign.append(results[gname]["layer_pair_config_signs"]["z_scores"][mname])

        if not z_shuf:
            continue

        mz_shuf = np.mean(z_shuf)
        mz_sign = np.mean(z_sign)
        delta = mz_sign - mz_shuf

        # Per-graph agreement: what fraction have same sign of delta?
        per_graph_delta = [s - h for s, h in zip(z_sign, z_shuf)]
        if delta > 0:
            agree = sum(1 for d in per_graph_delta if d > 0) / len(per_graph_delta)
        elif delta < 0:
            agree = sum(1 for d in per_graph_delta if d < 0) / len(per_graph_delta)
        else:
            agree = 1.0

        if abs(delta) < 0.5:
            interp = "no sign effect"
        elif delta > 0:
            interp = "TOPOLOGICAL"
        else:
            interp = "SIGN-DRIVEN"

        print(f"  {sn:15s} {mz_shuf:+8.1f} {mz_sign:+8.1f} {delta:+8.1f}  "
              f"{agree:6.0%}   {interp}")


def print_universal_signature(results: dict):
    """Print motif enrichment/depletion table across all graphs, all null types."""
    print_section("UNIVERSAL SIGNATURE: Enrichment across all 99 graphs")

    for null_type in NULL_TYPES:
        nt_short = NULL_SHORT[null_type]
        print(f"\n  {nt_short}:")
        print(f"  {'Motif':15s} {'mean_z':>8s} {'%enrich':>8s} {'%deplete':>9s} {'%neutral':>9s}")

        for mname in TEMPLATE_NAMES:
            sn = SHORT_NAMES[mname]
            zs = [results[g][null_type]["z_scores"][mname]
                  for g in results if null_type in results[g]]
            mz = np.mean(zs)
            pct_enrich = sum(1 for z in zs if z > 2.0) / len(zs)
            pct_deplete = sum(1 for z in zs if z < -2.0) / len(zs)
            pct_neutral = 1.0 - pct_enrich - pct_deplete
            print(f"  {sn:15s} {mz:+8.1f} {pct_enrich:7.0%} {pct_deplete:8.0%} {pct_neutral:8.0%}")


def print_cross_null_comparison(all_profiles: dict):
    """Compare how many significant motifs each null model reveals."""
    print_section("CROSS-NULL-MODEL COMPARISON")
    print("  Which null model reveals the most cross-task variation?")
    print(f"\n  {'Null model':15s} {'KW sig motifs':>14s} {'mean cos_sim':>13s} {'sim range':>12s}")

    for null_type in NULL_TYPES:
        nt_short = NULL_SHORT[null_type]
        profiles = all_profiles[null_type]["profiles"]
        kw = all_profiles[null_type]["kruskal"]
        sim = all_profiles[null_type]["sim_matrix"]

        n_sig = sum(1 for r in kw if r["sig"])

        # Off-diagonal similarities
        n = sim.shape[0]
        off_diag = [sim[i, j] for i in range(n) for j in range(i + 1, n)]
        mean_sim = np.mean(off_diag)
        min_sim = np.min(off_diag)
        max_sim = np.max(off_diag)

        print(f"  {nt_short:15s} {n_sig:>7d}/8      {mean_sim:13.4f} "
              f"[{min_sim:.3f}, {max_sim:.3f}]")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-task comparison analysis for unrolled motif Z-scores"
    )
    parser.add_argument(
        "--results-file", type=str,
        default=str(_REPO / "data" / "results" / "unrolled_null_pilot" / "pilot_results.json"),
        help="Path to pilot_results.json",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Path to save analysis JSON. Default: alongside results file.",
    )
    args = parser.parse_args()

    # Load results
    with open(args.results_file) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} graphs from {args.results_file}")

    # ── Universal signature ───────────────────────────────────────────
    print_universal_signature(results)

    # ── Per-null-model analysis ───────────────────────────────────────
    all_profiles: dict[str, dict] = {}

    for null_type in NULL_TYPES:
        nt_short = NULL_SHORT[null_type]
        print_section(f"NULL MODEL: {nt_short}")

        profiles = build_profiles(results, null_type)
        print_profiles(profiles, null_type)

        sim_matrix, task_order = cosine_sim_matrix(profiles)
        print_similarity(sim_matrix, task_order, null_type)

        kw = kruskal_per_motif(profiles)
        print_kruskal(kw, null_type)

        pw = pairwise_mannwhitney(profiles)
        print_pairwise(pw, null_type)

        Z_link, link_tasks = cluster_tasks(profiles)
        leaf_order = leaves_list(Z_link)
        print(f"\n  Clustering order ({nt_short}): "
              f"{' -> '.join(link_tasks[i] for i in leaf_order)}")

        all_profiles[null_type] = {
            "profiles": profiles,
            "sim_matrix": sim_matrix,
            "task_order": task_order,
            "kruskal": kw,
            "pairwise": pw,
            "linkage": Z_link,
        }

    # ── Sign effect analysis ──────────────────────────────────────────
    print_sign_effect(results)

    # ── Cross-null comparison ─────────────────────────────────────────
    print_cross_null_comparison(all_profiles)

    # ── Save analysis results ─────────────────────────────────────────
    out_path = args.output_file or str(
        Path(args.results_file).parent / "unrolled_analysis.json"
    )

    save_data = {}
    for null_type in NULL_TYPES:
        ap = all_profiles[null_type]
        save_data[null_type] = {
            "task_profiles": {
                task: {
                    "mean_sp": p["mean_sp"].tolist(),
                    "std_sp": p["std_sp"].tolist(),
                    "mean_z": p["mean_z"].tolist(),
                    "std_z": p["std_z"].tolist(),
                    "n_graphs": p["n_graphs"],
                }
                for task, p in ap["profiles"].items()
            },
            "cosine_similarity_matrix": ap["sim_matrix"].tolist(),
            "task_order": ap["task_order"],
            "kruskal_wallis": ap["kruskal"],
            "pairwise_comparisons": ap["pairwise"],
        }

    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nAnalysis saved to {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
