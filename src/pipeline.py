"""Full analysis pipeline: motif census + null model + cross-task comparison.

Iterates over all attribution graphs in data/raw/, runs the motif analysis
pipeline on each, aggregates results by task category, and saves outputs
to data/results/.

Can be run as a script: python -m src.pipeline
or imported and called programmatically.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from src.graph_loader import load_attribution_graph, graph_summary
from src.motif_census import (
    compute_motif_census,
    TRIAD_LABELS,
    CONNECTED_TRIAD_INDICES,
    MOTIF_FFL,
    MOTIF_CHAIN,
    MOTIF_FAN_IN,
    MOTIF_FAN_OUT,
    MOTIF_CYCLE,
)
from src.null_model import generate_configuration_null, NullModelResult
from src.comparison import (
    build_task_profile,
    all_pairwise_comparisons,
    cosine_similarity_matrix,
    kruskal_wallis_per_motif,
    hierarchical_clustering,
    TaskProfile,
)


# --- Graph discovery ---

def discover_graphs(
    data_dir: str | Path,
) -> dict[str, list[Path]]:
    """Discover all JSON attribution graphs organized by task category.

    Args:
        data_dir: Path to data/raw/ directory.

    Returns:
        Dict mapping category name to list of JSON file paths.
    """
    data_dir = Path(data_dir)
    categories: dict[str, list[Path]] = {}

    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue
        json_files = sorted(subdir.glob("*.json"))
        if json_files:
            categories[subdir.name] = json_files

    return categories


# --- Per-graph analysis ---

def analyze_single_graph(
    json_path: Path,
    n_random: int = 1000,
    motif_size: int = 3,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run the full motif analysis pipeline on a single graph.

    Args:
        json_path: Path to the JSON attribution graph.
        n_random: Number of null model random graphs.
        motif_size: Motif size (3 or 4).
        show_progress: Whether to show progress bar for null model.

    Returns:
        Dict with keys: path, summary, census, null_result, motif_instances.
    """
    g = load_attribution_graph(json_path)
    summary = graph_summary(g)

    # Motif census
    census = compute_motif_census(g, size=motif_size)

    # Null model + Z-scores
    null_result = generate_configuration_null(
        g, n_random=n_random, motif_size=motif_size,
        show_progress=show_progress,
    )

    # Get motif instance counts from census (already computed, no VF2 needed)
    raw = census.raw_counts
    instance_counts = {
        "FFL": raw[MOTIF_FFL],
        "chain": raw[MOTIF_CHAIN],
        "fan_in": raw[MOTIF_FAN_IN],
        "fan_out": raw[MOTIF_FAN_OUT],
        "cycle": raw[MOTIF_CYCLE],
    }

    return {
        "path": str(json_path),
        "name": json_path.stem,
        "summary": summary,
        "census": census,
        "null_result": null_result,
        "instance_counts": instance_counts,
    }


# --- Full pipeline ---

def run_pipeline(
    data_dir: str | Path = "data/raw",
    results_dir: str | Path = "data/results",
    n_random: int = 1000,
    motif_size: int = 3,
) -> dict[str, Any]:
    """Run the full analysis pipeline on all graphs.

    Args:
        data_dir: Path to data/raw/ directory.
        results_dir: Path to output directory for results.
        n_random: Number of null model random graphs per real graph.
        motif_size: Motif size (3 or 4).

    Returns:
        Dict with keys: per_graph, task_profiles, comparisons,
        similarity_matrix, clustering, kruskal_wallis.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    categories = discover_graphs(data_dir)

    total_graphs = sum(len(files) for files in categories.values())
    print(f"Found {total_graphs} graphs in {len(categories)} categories:")
    for cat, files in categories.items():
        print(f"  {cat}: {len(files)} graphs")
    print()

    # --- Phase 1: Per-graph analysis ---
    print("=" * 60)
    print("Phase 1: Per-graph motif census + null model")
    print("=" * 60)

    per_graph: dict[str, list[dict[str, Any]]] = {}
    graph_count = 0

    for category, files in categories.items():
        per_graph[category] = []
        for json_path in files:
            graph_count += 1
            print(f"\n[{graph_count}/{total_graphs}] {category}/{json_path.stem}")
            t0 = time.time()

            try:
                result = analyze_single_graph(
                    json_path,
                    n_random=n_random,
                    motif_size=motif_size,
                    show_progress=True,
                )
                per_graph[category].append(result)

                # Print summary
                s = result["summary"]
                z = result["null_result"].z_scores
                ic = result["instance_counts"]
                elapsed = time.time() - t0
                print(f"  Nodes={s['n_nodes']}, Edges={s['n_edges']}, "
                      f"FFLs={ic['FFL']}, Chains={ic['chain']}, "
                      f"Fan-in={ic['fan_in']}, Fan-out={ic['fan_out']}")
                print(f"  Top Z-scores: ", end="")
                top_z = sorted(
                    [(TRIAD_LABELS[i], z[i]) for i in CONNECTED_TRIAD_INDICES],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:3]
                print(", ".join(f"{label}={zscore:+.1f}" for label, zscore in top_z))
                print(f"  Completed in {elapsed:.1f}s")

            except Exception as e:
                print(f"  ERROR: {e}")
                continue

    # Save per-graph results (pickle for full objects)
    per_graph_path = results_dir / "per_graph_results.pkl"
    with open(per_graph_path, "wb") as f:
        pickle.dump(per_graph, f)
    print(f"\nSaved per-graph results to {per_graph_path}")

    # --- Phase 2: Cross-task aggregation ---
    print("\n" + "=" * 60)
    print("Phase 2: Cross-task aggregation")
    print("=" * 60)

    task_profiles: dict[str, TaskProfile] = {}
    for category, results_list in per_graph.items():
        if not results_list:
            print(f"  Skipping {category}: no successful results")
            continue
        null_results = [r["null_result"] for r in results_list]
        profile = build_task_profile(category, null_results)
        task_profiles[category] = profile
        print(f"  {category}: {profile.n_graphs} graphs, "
              f"mean |Z| = {np.mean(np.abs(profile.mean_z)):.2f}")

    # --- Phase 3: Statistical comparisons ---
    print("\n" + "=" * 60)
    print("Phase 3: Statistical comparisons")
    print("=" * 60)

    # Pairwise comparisons
    comparisons = all_pairwise_comparisons(task_profiles)
    print(f"\nPairwise comparisons ({len(comparisons)} pairs):")
    for comp in sorted(comparisons, key=lambda c: c.cosine_similarity, reverse=True):
        sig_str = f", {len(comp.significant_motifs)} sig. motifs" if comp.significant_motifs else ""
        print(f"  {comp.task_a} vs {comp.task_b}: "
              f"cos_sim={comp.cosine_similarity:.3f}{sig_str}")

    # Cosine similarity matrix
    sim_matrix, task_names = cosine_similarity_matrix(task_profiles)
    print(f"\nCosine similarity matrix computed ({len(task_names)}x{len(task_names)})")

    # Kruskal-Wallis
    kw_results = kruskal_wallis_per_motif(task_profiles)
    sig_motifs = [r for r in kw_results if r["significant"]]
    print(f"\nKruskal-Wallis: {len(sig_motifs)}/{len(kw_results)} motif classes "
          f"show significant cross-task differences (p < 0.05)")
    for r in sig_motifs:
        idx = r["motif_index"]
        label = TRIAD_LABELS[idx] if idx < 16 else str(idx)
        print(f"  {label}: H={r['H_statistic']:.2f}, p={r['p_value']:.4f}")

    # Hierarchical clustering
    if len(task_profiles) >= 2:
        linkage_matrix, cluster_names = hierarchical_clustering(task_profiles)
        print(f"\nHierarchical clustering computed ({len(cluster_names)} tasks)")
    else:
        linkage_matrix, cluster_names = np.array([]), list(task_profiles.keys())

    # --- Save summary results ---
    summary_data = _build_summary_json(
        per_graph, task_profiles, comparisons, kw_results, sim_matrix, task_names,
    )

    summary_path = results_dir / "analysis_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=_json_serializer)
    print(f"\nSaved analysis summary to {summary_path}")

    # Save task profiles (pickle)
    profiles_path = results_dir / "task_profiles.pkl"
    with open(profiles_path, "wb") as f:
        pickle.dump(task_profiles, f)
    print(f"Saved task profiles to {profiles_path}")

    # Save clustering
    cluster_path = results_dir / "clustering.pkl"
    with open(cluster_path, "wb") as f:
        pickle.dump({"linkage": linkage_matrix, "names": cluster_names}, f)
    print(f"Saved clustering to {cluster_path}")

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)

    return {
        "per_graph": per_graph,
        "task_profiles": task_profiles,
        "comparisons": comparisons,
        "similarity_matrix": (sim_matrix, task_names),
        "clustering": (linkage_matrix, cluster_names),
        "kruskal_wallis": kw_results,
    }


def _build_summary_json(
    per_graph: dict[str, list[dict]],
    task_profiles: dict[str, TaskProfile],
    comparisons: list,
    kw_results: list[dict],
    sim_matrix: np.ndarray,
    task_names: list[str],
) -> dict[str, Any]:
    """Build a JSON-serializable summary of all results."""
    summary: dict[str, Any] = {}

    # Per-graph summaries
    graph_summaries = []
    for category, results_list in per_graph.items():
        for r in results_list:
            gs = {
                "category": category,
                "name": r["name"],
                "n_nodes": r["summary"]["n_nodes"],
                "n_edges": r["summary"]["n_edges"],
                "prompt": r["summary"]["prompt"],
                "z_scores": {
                    TRIAD_LABELS[i]: float(r["null_result"].z_scores[i])
                    for i in CONNECTED_TRIAD_INDICES
                },
                "significance_profile": {
                    TRIAD_LABELS[i]: float(r["null_result"].significance_profile[i])
                    for i in CONNECTED_TRIAD_INDICES
                },
                "instance_counts": r["instance_counts"],
            }
            graph_summaries.append(gs)
    summary["graphs"] = graph_summaries

    # Task profiles
    profile_summaries = {}
    for name, profile in task_profiles.items():
        profile_summaries[name] = {
            "n_graphs": profile.n_graphs,
            "mean_z": {
                TRIAD_LABELS[i]: float(profile.mean_z[i])
                for i in CONNECTED_TRIAD_INDICES
            },
            "mean_sp": {
                TRIAD_LABELS[i]: float(profile.mean_sp[i])
                for i in CONNECTED_TRIAD_INDICES
            },
        }
    summary["task_profiles"] = profile_summaries

    # Pairwise comparisons
    summary["pairwise_comparisons"] = [
        {
            "task_a": c.task_a,
            "task_b": c.task_b,
            "cosine_similarity": c.cosine_similarity,
            "n_significant_motifs": len(c.significant_motifs),
            "significant_motif_indices": c.significant_motifs,
        }
        for c in comparisons
    ]

    # Kruskal-Wallis
    summary["kruskal_wallis"] = [
        {
            "motif_index": r["motif_index"],
            "label": TRIAD_LABELS[r["motif_index"]] if r["motif_index"] < 16 else str(r["motif_index"]),
            "H_statistic": r["H_statistic"],
            "p_value": r["p_value"],
            "significant": r["significant"],
        }
        for r in kw_results
    ]

    # Similarity matrix
    summary["similarity_matrix"] = {
        "task_names": task_names,
        "matrix": sim_matrix.tolist(),
    }

    # Motif zoo: top enriched/depleted motifs across all graphs
    motif_zoo = _build_motif_zoo(per_graph)
    summary["motif_zoo"] = motif_zoo

    return summary


def _build_motif_zoo(
    per_graph: dict[str, list[dict]],
) -> dict[str, Any]:
    """Build the motif zoo: summary of enrichment patterns across all graphs."""
    zoo: dict[str, Any] = {}

    # Collect Z-scores per motif across all graphs
    all_z: dict[str, list[float]] = {}
    for i in CONNECTED_TRIAD_INDICES:
        label = TRIAD_LABELS[i]
        all_z[label] = []

    for category, results_list in per_graph.items():
        for r in results_list:
            z = r["null_result"].z_scores
            for i in CONNECTED_TRIAD_INDICES:
                label = TRIAD_LABELS[i]
                all_z[label].append(float(z[i]))

    # Summary statistics per motif
    motif_stats = []
    for label, z_values in all_z.items():
        if not z_values:
            continue
        z_arr = np.array(z_values)
        z_finite = z_arr[np.isfinite(z_arr)]
        if len(z_finite) == 0:
            continue

        n_enriched = int(np.sum(z_finite > 2.0))
        n_depleted = int(np.sum(z_finite < -2.0))

        motif_stats.append({
            "label": label,
            "mean_z": float(np.mean(z_finite)),
            "std_z": float(np.std(z_finite)),
            "median_z": float(np.median(z_finite)),
            "min_z": float(np.min(z_finite)),
            "max_z": float(np.max(z_finite)),
            "n_enriched": n_enriched,
            "n_depleted": n_depleted,
            "n_graphs": len(z_finite),
            "pct_enriched": round(100 * n_enriched / len(z_finite), 1),
            "pct_depleted": round(100 * n_depleted / len(z_finite), 1),
        })

    # Sort by absolute mean Z-score
    motif_stats.sort(key=lambda x: abs(x["mean_z"]), reverse=True)
    zoo["motif_summary"] = motif_stats

    # Find most interesting graphs (extreme Z-scores)
    interesting = []
    for category, results_list in per_graph.items():
        for r in results_list:
            z = r["null_result"].z_scores
            z_finite = z[np.isfinite(z)]
            if len(z_finite) == 0:
                continue
            max_abs_z = float(np.max(np.abs(z_finite)))
            n_sig = int(np.sum(np.abs(z_finite) > 2.0))
            interesting.append({
                "category": category,
                "name": r["name"],
                "max_abs_z": max_abs_z,
                "n_significant_motifs": n_sig,
                "n_nodes": r["summary"]["n_nodes"],
                "n_edges": r["summary"]["n_edges"],
                "prompt": r["summary"]["prompt"][:100],
            })

    interesting.sort(key=lambda x: x["max_abs_z"], reverse=True)
    zoo["interesting_graphs"] = interesting[:20]  # Top 20

    return zoo


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if hasattr(obj, "__dict__"):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# --- Multi-model / scale pipeline ---

def discover_graphs_multi_model(
    data_dir: str | Path,
) -> dict[str, dict[str, list[Path]]]:
    """Discover attribution graphs organized by model and task category.

    Expects a directory layout like:
        data_dir/{model_id}/{category}/*.json

    Detects model-level subdirectories by checking for nested category
    subdirectories (i.e., subdirs that themselves contain .json files).

    Args:
        data_dir: Path to the top-level data directory.

    Returns:
        Dict mapping model_id → {category_name → [json paths]}.
    """
    data_dir = Path(data_dir)
    models: dict[str, dict[str, list[Path]]] = {}

    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        # Check if this looks like a model directory (has subdirs with JSON)
        categories: dict[str, list[Path]] = {}
        for subdir in sorted(model_dir.iterdir()):
            if not subdir.is_dir():
                continue
            json_files = sorted(subdir.glob("*.json"))
            if json_files:
                categories[subdir.name] = json_files

        if categories:
            models[model_dir.name] = categories

    return models


def run_scale_pipeline(
    data_dir: str | Path = "data/raw",
    results_dir: str | Path = "data/results/scale",
    n_random: int = 1000,
    motif_size: int = 3,
    model_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Run the motif analysis pipeline across multiple model scales.

    Discovers model-specific subdirectories under data_dir, runs the
    existing run_pipeline() for each model, then performs cross-scale
    comparison.

    Args:
        data_dir: Path to data directory with model subdirectories.
        results_dir: Path to output directory for scale results.
        n_random: Number of null model random graphs per real graph.
        motif_size: Motif size (3 or 4).
        model_ids: Optional list of model IDs to analyze. If None, all
            detected models are analyzed.

    Returns:
        Dict with keys: model_results, model_profiles, scale_comparison.
    """
    from src.models import get_model, ALL_MODELS
    from src.scale_comparison import (
        build_model_profile,
        run_scale_comparison,
        ModelProfile,
    )

    data_dir = Path(data_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover available models
    all_model_graphs = discover_graphs_multi_model(data_dir)

    if model_ids:
        model_graph_subset = {
            m: cats for m, cats in all_model_graphs.items() if m in model_ids
        }
    else:
        model_graph_subset = all_model_graphs

    if not model_graph_subset:
        print("No model subdirectories found. Ensure data_dir has "
              "{model_id}/{category}/*.json structure.")
        return {}

    print(f"Found {len(model_graph_subset)} model(s) to analyze:")
    for model_id, cats in model_graph_subset.items():
        total = sum(len(f) for f in cats.values())
        print(f"  {model_id}: {total} graphs in {len(cats)} categories")
    print()

    # Phase 1: Run existing pipeline per model
    model_results: dict[str, dict[str, Any]] = {}
    for model_id, categories in model_graph_subset.items():
        print("=" * 60)
        print(f"Model: {model_id}")
        print("=" * 60)

        model_data_dir = data_dir / model_id
        model_results_dir = results_dir / model_id
        model_results_dir.mkdir(parents=True, exist_ok=True)

        result = run_pipeline(
            data_dir=model_data_dir,
            results_dir=model_results_dir,
            n_random=n_random,
            motif_size=motif_size,
        )
        model_results[model_id] = result

    # Phase 2: Build ModelProfiles
    print("\n" + "=" * 60)
    print("Phase 4: Cross-scale comparison")
    print("=" * 60)

    model_profiles: dict[str, ModelProfile] = {}
    for model_id, result in model_results.items():
        task_profiles = result.get("task_profiles", {})
        if not task_profiles:
            print(f"  Skipping {model_id}: no task profiles")
            continue

        # Look up model spec (or create a minimal one if not in registry)
        if model_id in ALL_MODELS:
            spec = get_model(model_id)
        else:
            from src.models import ModelSpec
            print(f"  Warning: {model_id} not in model registry, using minimal spec")
            spec = ModelSpec(
                model_id=model_id,
                family="unknown",
                variant="unknown",
                n_params=0,
                n_layers=0,
                hidden_dim=0,
            )

        mp = build_model_profile(spec, task_profiles)
        model_profiles[model_id] = mp
        print(f"  {model_id}: {mp.n_total_graphs} total graphs, "
              f"{len(mp.task_profiles)} tasks")

    if len(model_profiles) < 2:
        print("\nNeed at least 2 models for cross-scale comparison.")
        return {"model_results": model_results, "model_profiles": model_profiles}

    # Phase 3: Scale comparison
    scale_result = run_scale_comparison(model_profiles)

    # Save results
    profiles_path = results_dir / "scale_profiles.pkl"
    with open(profiles_path, "wb") as f:
        pickle.dump(model_profiles, f)
    print(f"\nSaved model profiles to {profiles_path}")

    # Save scale analysis summary as JSON
    scale_summary = _build_scale_summary_json(scale_result)
    summary_path = results_dir / "scale_analysis.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(scale_summary, f, indent=2, default=_json_serializer)
    print(f"Saved scale analysis to {summary_path}")

    print("\n" + "=" * 60)
    print("Scale pipeline complete!")
    print("=" * 60)

    return {
        "model_results": model_results,
        "model_profiles": model_profiles,
        "scale_comparison": scale_result,
    }


def _build_scale_summary_json(
    scale_result: Any,
) -> dict[str, Any]:
    """Build a JSON-serializable summary of scale comparison results."""
    summary: dict[str, Any] = {}

    # Model overview
    summary["models"] = {
        model_id: {
            "n_params": mp.model_spec.n_params,
            "n_layers": mp.model_spec.n_layers,
            "n_total_graphs": mp.n_total_graphs,
            "n_tasks": len(mp.task_profiles),
            "tasks": list(mp.task_profiles.keys()),
        }
        for model_id, mp in scale_result.model_profiles.items()
    }

    # Scale trends (Z-score based)
    summary["scale_trends"] = [
        {
            "motif_index": t.motif_index,
            "motif_label": t.motif_label,
            "slope": t.slope,
            "r_squared": t.r_squared,
            "p_value": t.p_value,
            "spearman_rho": t.spearman_rho,
            "spearman_p": t.spearman_p,
            "trend_direction": t.trend_direction,
            "is_significant": t.is_significant,
            "values": t.values,
        }
        for t in scale_result.scale_trends
    ]

    # Scale trends (SP-based — normalized, controls for graph size)
    summary["sp_scale_trends"] = [
        {
            "motif_index": t.motif_index,
            "motif_label": t.motif_label,
            "slope": t.slope,
            "r_squared": t.r_squared,
            "p_value": t.p_value,
            "spearman_rho": t.spearman_rho,
            "spearman_p": t.spearman_p,
            "trend_direction": t.trend_direction,
            "is_significant": t.is_significant,
            "values": t.values,
        }
        for t in scale_result.sp_scale_trends
    ]

    # Phase transitions (Z-score based)
    summary["phase_transitions"] = [
        {
            "motif_index": pt.motif_index,
            "motif_label": pt.motif_label,
            "transition_point": pt.transition_point,
            "before_mean": pt.before_mean,
            "after_mean": pt.after_mean,
            "effect_size": pt.effect_size,
        }
        for pt in scale_result.phase_transitions
    ]

    # Phase transitions (SP-based — normalized, controls for graph size)
    summary["sp_phase_transitions"] = [
        {
            "motif_index": pt.motif_index,
            "motif_label": pt.motif_label,
            "transition_point": pt.transition_point,
            "before_mean": pt.before_mean,
            "after_mean": pt.after_mean,
            "effect_size": pt.effect_size,
        }
        for pt in scale_result.sp_phase_transitions
    ]

    # FFL backbone
    summary["ffl_backbone"] = {
        "universal": scale_result.ffl_backbone_universal,
        "details": scale_result.ffl_details,
    }

    # Similarity matrix
    summary["similarity_matrix"] = {
        "model_names": scale_result.model_names,
        "matrix": scale_result.similarity_matrix.tolist()
        if hasattr(scale_result.similarity_matrix, "tolist")
        else [],
    }

    return summary


# --- CLI entry point ---

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run motif analysis pipeline")
    parser.add_argument(
        "--data-dir", type=str, default="data/raw",
        help="Path to raw data directory",
    )
    parser.add_argument(
        "--results-dir", type=str, default="data/results",
        help="Path to results output directory",
    )
    parser.add_argument(
        "--n-random", type=int, default=1000,
        help="Number of null model random graphs per real graph",
    )
    parser.add_argument(
        "--motif-size", type=int, default=3,
        help="Motif size (3 or 4)",
    )
    parser.add_argument(
        "--scale-mode", action="store_true",
        help="Run cross-scale pipeline (expects model subdirectories in data-dir)",
    )
    parser.add_argument(
        "--models", type=str, nargs="*", default=None,
        help="Model IDs to analyze in scale mode (default: all detected)",
    )
    args = parser.parse_args()

    if args.scale_mode:
        scale_results_dir = args.results_dir
        if not scale_results_dir.endswith("/scale") and not scale_results_dir.endswith("\\scale"):
            scale_results_dir = str(Path(args.results_dir) / "scale")
        run_scale_pipeline(
            data_dir=args.data_dir,
            results_dir=scale_results_dir,
            n_random=args.n_random,
            motif_size=args.motif_size,
            model_ids=args.models,
        )
    else:
        run_pipeline(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            n_random=args.n_random,
            motif_size=args.motif_size,
        )
