"""Unrolled motif census across 5 null model types on smallest Haiku graphs.

Parallelizes null iterations across CPU cores. Includes both sign-shuffled
and sign-preserving variants of the layer-pair config null to separate
topological from sign-coherence effects.

Null models:
  1. configuration           — degree-preserving rewire. Preserves edge
     attributes (sign, weight) but breaks layer ordering.
  2. erdos_renyi             — random directed graph, same n/m. Random signs.
  3. layer_preserving_er     — random bipartite edges within each layer pair,
     preserving edge budget per pair. Random signs.
  4. layer_pair_config       — degree-preserving swaps within each layer pair.
     Random signs. Tests topology + sign structure jointly.
  5. layer_pair_config_signs — same swaps as (4), but signs travel with their
     edge (preserved). Tests pure topology, controlling for sign structure.

Usage:
    python scripts/unrolled_null_pilot.py --n-random 100 --n-workers 14
    python scripts/unrolled_null_pilot.py --n-random 1000 --n-workers -1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import igraph as ig

# Allow running from repo root or scripts/
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.pipeline import discover_graphs
from src.graph_loader import load_attribution_graph
from src.unrolled_motifs import build_catalog, get_effective_layer
from src.unrolled_census import run_unrolled_census, unrolled_census_counts

# ── Defaults ───────────────────────────────────────────────────────────
DEFAULT_N_RANDOM = 100
DEFAULT_DATA_DIR = str(_REPO / "data" / "raw")

# 20 smallest Haiku graphs by node count
TARGET_NAMES = [
    "factual_recall/iasg-clt-18l-p70",               #  24 nodes,   82 edges
    "arithmetic/polymer-add-9",                        #  46 nodes,  433 edges
    "factual_recall/iasg-clt-18l-p80",                #  47 nodes,  345 edges
    "factual_recall/iasg-clt-clean",                   #  47 nodes,  345 edges
    "factual_recall/uspto-telephone-clt-18l",          #  52 nodes,  535 edges
    "factual_recall/uspto-telephone-clt-clean",        #  52 nodes,  535 edges
    "factual_recall/ndag-18l",                         #  57 nodes,  581 edges
    "factual_recall/ndag-18l-analytics",               #  57 nodes,  581 edges
    "factual_recall/ndag-clt-clean",                   #  57 nodes,  581 edges
    "multihop/capital-analogy-clt-18l",                #  57 nodes,  653 edges
    "multihop/capital-analogy-clt-18l-path-highlight", #  57 nodes,  653 edges
    "multihop/capital-analogy-clt-clean",              #  57 nodes,  653 edges
    "multilingual/opposite_of_small_zh",               #  57 nodes,  821 edges
    "arithmetic/order-of-operations-paren",            #  68 nodes,  854 edges
    "safety/bomb-baseline",                            #  69 nodes,  876 edges
    "arithmetic/calc-11-plus-4",                       #  76 nodes, 1495 edges
    "factual_recall/opposite_of_small",                #  78 nodes, 1282 edges
    "arithmetic/calc-17-plus-22",                      #  78 nodes, 1651 edges
    "safety/bon-errors",                               #  79 nodes,  670 edges
    "arithmetic/calc-6-plus-9",                        #  80 nodes, 1619 edges
]

NULL_TYPES = [
    "configuration",
    "erdos_renyi",
    "layer_preserving_er",
    "layer_pair_config",
    "layer_pair_config_signs",
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

NULL_SHORT = {
    "configuration": "config",
    "erdos_renyi": "ER",
    "layer_preserving_er": "LP-ER",
    "layer_pair_config": "LPC-shuf",
    "layer_pair_config_signs": "LPC-sign",
}


# ── Null graph generators ─────────────────────────────────────────────
# Module-level functions for picklability with multiprocessing.

def _assign_random_signs(real_graph: ig.Graph, null_graph: ig.Graph, rng):
    """Assign random edge signs to null graph, matching real distribution."""
    if null_graph.ecount() == 0:
        return
    if "sign" in real_graph.es.attributes() and real_graph.ecount() > 0:
        signs = real_graph.es["sign"]
        p_exc = sum(1 for s in signs if s == "excitatory") / len(signs)
    else:
        p_exc = 0.5
    n = null_graph.ecount()
    is_exc = rng.random(n) < p_exc
    null_graph.es["sign"] = ["excitatory" if e else "inhibitory" for e in is_exc]
    null_graph.es["weight"] = [1.0] * n
    null_graph.es["raw_weight"] = [1.0 if e else -1.0 for e in is_exc]


def gen_configuration(graph: ig.Graph, rng) -> ig.Graph:
    """Degree-preserving rewire. Preserves edge attributes (sign, weight)."""
    g = graph.copy()
    g.rewire(n=max(g.ecount() * 10, 1))
    return g


def gen_erdos_renyi(graph: ig.Graph, rng) -> ig.Graph:
    """ER random graph, same n/m. Random sign assignment."""
    g = ig.Graph.Erdos_Renyi(n=graph.vcount(), m=graph.ecount(), directed=True)
    for attr in graph.vs.attributes():
        g.vs[attr] = graph.vs[attr]
    _assign_random_signs(graph, g, rng)
    return g


def gen_layer_preserving_er(graph: ig.Graph, rng) -> ig.Graph:
    """Layer-pair ER: random bipartite edges per layer pair, random signs."""
    layers = [get_effective_layer(graph, v.index) for v in graph.vs]
    nodes_by_layer: dict[int, list[int]] = defaultdict(list)
    for idx, ly in enumerate(layers):
        nodes_by_layer[ly].append(idx)

    pair_counts: dict[tuple[int, int], int] = defaultdict(int)
    for e in graph.es:
        pair_counts[(layers[e.source], layers[e.target])] += 1

    g = ig.Graph(n=graph.vcount(), directed=True)
    for attr in graph.vs.attributes():
        g.vs[attr] = graph.vs[attr]

    for (sl, tl), ne in pair_counts.items():
        src = nodes_by_layer[sl]
        tgt = nodes_by_layer[tl]
        mx = len(src) * len(tgt)
        if ne >= mx:
            edges = [(s, t) for s in src for t in tgt]
        else:
            idxs = rng.choice(mx, size=ne, replace=False)
            edges = [(src[i // len(tgt)], tgt[i % len(tgt)]) for i in idxs]
        g.add_edges(edges)

    _assign_random_signs(graph, g, rng)
    return g


def _layer_pair_config_rewire(graph: ig.Graph, rng, preserve_signs: bool) -> ig.Graph:
    """Layer-pair config: degree-preserving swaps within each layer pair.

    Args:
        graph: Real attribution graph.
        rng: NumPy random generator.
        preserve_signs: If True, edge signs travel with the edge during swaps.
            If False, signs are randomly reassigned after rewiring.
    """
    layers = [get_effective_layer(graph, v.index) for v in graph.vs]

    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False

    # Group edges by layer pair, optionally with attributes
    pair_edges: dict[tuple[int, int], list] = defaultdict(list)
    for e in graph.es:
        key = (layers[e.source], layers[e.target])
        if preserve_signs and has_sign:
            pair_edges[key].append(
                (e.source, e.target, e["sign"], e["weight"], e["raw_weight"])
            )
        else:
            pair_edges[key].append((e.source, e.target))

    g = ig.Graph(n=graph.vcount(), directed=True)
    for attr in graph.vs.attributes():
        g.vs[attr] = graph.vs[attr]

    all_edges = []
    all_signs = []
    all_weights = []
    all_raw_weights = []

    for (sl, tl), edges in pair_edges.items():
        np_ = len(edges)
        if np_ < 2:
            # Can't swap with < 2 edges, keep as-is
            for edge_data in edges:
                all_edges.append((edge_data[0], edge_data[1]))
                if preserve_signs and has_sign:
                    all_signs.append(edge_data[2])
                    all_weights.append(edge_data[3])
                    all_raw_weights.append(edge_data[4])
            continue

        cur = list(edges)
        # Edge set for O(1) existence check (topology only)
        eset = {(e[0], e[1]) for e in cur}

        for _ in range(np_ * 10):
            i1, i2 = rng.choice(np_, size=2, replace=False)
            s1, t1 = cur[i1][0], cur[i1][1]
            s2, t2 = cur[i2][0], cur[i2][1]
            if s1 == s2 or t1 == t2:
                continue
            if (s1, t2) in eset or (s2, t1) in eset:
                continue

            # Perform swap — attributes travel with their original edge
            eset.discard((s1, t1))
            eset.discard((s2, t2))
            eset.add((s1, t2))
            eset.add((s2, t1))

            if preserve_signs and has_sign:
                # Edge 1's attributes go to new edge 1 (s1→t2)
                # Edge 2's attributes go to new edge 2 (s2→t1)
                attrs1 = cur[i1][2:]  # (sign, weight, raw_weight)
                attrs2 = cur[i2][2:]
                cur[i1] = (s1, t2) + attrs1
                cur[i2] = (s2, t1) + attrs2
            else:
                cur[i1] = (s1, t2)
                cur[i2] = (s2, t1)

        for edge_data in cur:
            all_edges.append((edge_data[0], edge_data[1]))
            if preserve_signs and has_sign:
                all_signs.append(edge_data[2])
                all_weights.append(edge_data[3])
                all_raw_weights.append(edge_data[4])

    g.add_edges(all_edges)

    if preserve_signs and has_sign:
        g.es["sign"] = all_signs
        g.es["weight"] = all_weights
        g.es["raw_weight"] = all_raw_weights
    else:
        _assign_random_signs(graph, g, rng)

    return g


def gen_layer_pair_config(graph: ig.Graph, rng) -> ig.Graph:
    """Layer-pair config with shuffled signs."""
    return _layer_pair_config_rewire(graph, rng, preserve_signs=False)


def gen_layer_pair_config_signs(graph: ig.Graph, rng) -> ig.Graph:
    """Layer-pair config with signs preserved (travel with edge)."""
    return _layer_pair_config_rewire(graph, rng, preserve_signs=True)


GENERATORS = {
    "configuration": gen_configuration,
    "erdos_renyi": gen_erdos_renyi,
    "layer_preserving_er": gen_layer_preserving_er,
    "layer_pair_config": gen_layer_pair_config,
    "layer_pair_config_signs": gen_layer_pair_config_signs,
}


# ── Summary table ─────────────────────────────────────────────────────

def print_summary(results: dict, template_names: list[str]) -> None:
    """Print running summary tables of all results so far."""
    n_done = len(results)
    if n_done == 0:
        return

    # Per-graph table
    print(f"\n{'='*120}", flush=True)
    print(f"Z-SCORE TABLE ({n_done} graph{'s' if n_done != 1 else ''})", flush=True)
    print(f"{'='*120}", flush=True)

    for gname in results:
        print(f"\n--- {gname} ---", flush=True)
        hdr = f"  {'Motif':15s} {'real':>5s}"
        for nt in NULL_TYPES:
            hdr += f"  {NULL_SHORT[nt]:>12s}"
        print(hdr, flush=True)

        for tname in template_names:
            sn = SHORT_NAMES.get(tname, tname[:15])
            rc = results[gname][NULL_TYPES[0]]["real_counts"][tname]
            line = f"  {sn:15s} {rc:5d}"
            for nt in NULL_TYPES:
                z = results[gname][nt]["z_scores"][tname]
                mn = results[gname][nt]["mean_null"][tname]
                if abs(z) > 2.0:
                    marker = "**"
                elif abs(z) > 1.0:
                    marker = "* "
                else:
                    marker = "  "
                line += f"  {z:+6.1f}{marker}({mn:4.0f})"
            print(line, flush=True)

    if n_done < 2:
        return

    # Cross-graph mean Z-scores
    print(f"\n{'='*120}", flush=True)
    print(
        f"SUMMARY: Mean Z-score across {n_done} graphs "
        f"(** = |z|>2 in >50%, * = |z|>2 in any)",
        flush=True,
    )
    print(f"{'='*120}", flush=True)

    hdr = f"  {'Motif':15s}"
    for nt in NULL_TYPES:
        hdr += f"  {NULL_SHORT[nt]:>12s}"
    print(hdr, flush=True)

    for tname in template_names:
        sn = SHORT_NAMES.get(tname, tname[:15])
        line = f"  {sn:15s}"
        for nt in NULL_TYPES:
            zs = [results[gn][nt]["z_scores"][tname] for gn in results]
            mz = np.mean(zs)
            n_sig = sum(1 for z in zs if abs(z) > 2.0)
            marker = "**" if n_sig > n_done / 2 else "* " if n_sig >= 1 else "  "
            line += f"  {mz:+8.1f} {marker} "
        print(line, flush=True)

    # Sign effect comparison
    print(f"\n{'='*120}", flush=True)
    print(
        "SIGN EFFECT: layer_pair_config sign-shuffled vs sign-preserved",
        flush=True,
    )
    print(
        "  Positive delta = sign preservation increases z (topology alone enriches more)",
        flush=True,
    )
    print(
        "  Negative delta = sign preservation decreases z (sign coherence was boosting enrichment)",
        flush=True,
    )
    print(f"{'='*120}", flush=True)

    print(f"  {'Motif':15s} {'shuf_z':>8s} {'sign_z':>8s} {'delta':>8s}  interpretation", flush=True)
    for tname in template_names:
        sn = SHORT_NAMES.get(tname, tname[:15])
        zs_shuf = [results[gn]["layer_pair_config"]["z_scores"][tname] for gn in results]
        zs_sign = [results[gn]["layer_pair_config_signs"]["z_scores"][tname] for gn in results]
        mz_shuf = np.mean(zs_shuf)
        mz_sign = np.mean(zs_sign)
        delta = mz_sign - mz_shuf

        if abs(delta) < 0.5:
            interp = "no sign effect"
        elif delta > 0:
            interp = "TOPOLOGICAL (enriched even with real signs)"
        else:
            interp = "SIGN-DRIVEN (enrichment from sign shuffling)"

        print(f"  {sn:15s} {mz_shuf:+8.1f} {mz_sign:+8.1f} {delta:+8.1f}  {interp}", flush=True)

    print(flush=True)


# ── Parallel worker ───────────────────────────────────────────────────

def _worker(args: tuple) -> dict[str, int]:
    """Generate one null graph and run unrolled census on it.

    Args:
        args: (graph, null_type, seed, templates)

    Returns:
        Dict mapping motif name to instance count.
    """
    graph, null_type, seed, templates = args
    rng = np.random.default_rng(seed=seed)
    gen_fn = GENERATORS[null_type]
    g_null = gen_fn(graph, rng)
    census = run_unrolled_census(g_null, templates)
    return unrolled_census_counts(census)


# ── Z-score computation ───────────────────────────────────────────────

def compute_unrolled_zscores(
    graph: ig.Graph,
    null_type: str,
    n_random: int,
    n_workers: int,
    templates,
) -> tuple[dict, dict, dict, dict]:
    """Compute unrolled motif z-scores with parallel null iterations."""

    # Real census (single, sequential)
    real_census = run_unrolled_census(graph, templates)
    real_counts = unrolled_census_counts(real_census)

    # Null ensemble — parallel
    null_counts: dict[str, list[int]] = {t.name: [] for t in templates}
    args_list = [
        (graph, null_type, seed, templates)
        for seed in range(n_random)
    ]

    if n_workers == 1:
        for a in args_list:
            nc = _worker(a)
            for t in templates:
                null_counts[t.name].append(nc[t.name])
    else:
        max_w = os.cpu_count() if n_workers == -1 else n_workers
        with ProcessPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_worker, a): i for i, a in enumerate(args_list)}
            for future in as_completed(futures):
                nc = future.result()
                for t in templates:
                    null_counts[t.name].append(nc[t.name])

    # Z-scores
    z_scores: dict[str, float] = {}
    mean_null: dict[str, float] = {}
    std_null: dict[str, float] = {}
    for t in templates:
        obs = real_counts[t.name]
        arr = np.array(null_counts[t.name], dtype=float)
        m, s = float(arr.mean()), float(arr.std())
        mean_null[t.name] = m
        std_null[t.name] = s
        if s > 1e-10:
            z_scores[t.name] = (obs - m) / s
        elif abs(obs - m) < 1e-10:
            z_scores[t.name] = 0.0
        else:
            z_scores[t.name] = float(np.sign(obs - m)) * 100.0

    return real_counts, z_scores, mean_null, std_null


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unrolled motif census: N graphs × 5 null models"
    )
    parser.add_argument(
        "--n-random", type=int, default=DEFAULT_N_RANDOM,
        help=f"Null iterations per (graph, null_type). Default: {DEFAULT_N_RANDOM}",
    )
    parser.add_argument(
        "--n-workers", type=int, default=-1,
        help="Parallel workers. -1 = all cores. Default: -1",
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Path to data/raw/ directory.",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save results. Default: data/results/unrolled_null_pilot/",
    )
    parser.add_argument(
        "--n-graphs", type=int, default=len(TARGET_NAMES),
        help=f"Number of smallest graphs to use (max {len(TARGET_NAMES)}). Default: all {len(TARGET_NAMES)}",
    )
    parser.add_argument(
        "--skip-graphs", type=int, default=0,
        help="Skip the first N graphs. Useful for resuming. Default: 0",
    )
    args = parser.parse_args()

    n_random = args.n_random
    n_workers = args.n_workers
    data_dir = args.data_dir
    output_dir = args.output_dir or str(_REPO / "data" / "results" / "unrolled_null_pilot")
    n_graphs = min(args.n_graphs, len(TARGET_NAMES))
    target_names = TARGET_NAMES[args.skip_graphs:n_graphs]

    effective_workers = os.cpu_count() if n_workers == -1 else n_workers
    print(f"Config: {n_random} null iterations, {effective_workers} workers, "
          f"{n_graphs} graphs, {len(NULL_TYPES)} null types", flush=True)
    print(f"Data:   {data_dir}", flush=True)
    print(f"Null types: {', '.join(NULL_SHORT[nt] for nt in NULL_TYPES)}", flush=True)

    # Discover graphs
    categories = discover_graphs(data_dir)
    name_to_path: dict[str, Path] = {}
    for cat, paths in categories.items():
        for p in paths:
            name_to_path[f"{cat}/{p.stem}"] = p

    templates = build_catalog()
    template_names = [t.name for t in templates]

    # ── Load any previously saved incremental results ───────────────
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_skip{args.skip_graphs}" if args.skip_graphs > 0 else ""
    out_path = Path(output_dir) / f"pilot_results{suffix}.json"

    if out_path.exists():
        with open(out_path) as f:
            results: dict[str, dict] = json.load(f)
        print(f"Loaded {len(results)} previously saved graph(s) from {out_path}", flush=True)
    else:
        results: dict[str, dict] = {}

    total_t0 = time.time()

    for gi, gname in enumerate(target_names):
        if gname not in name_to_path:
            print(f"WARNING: {gname} not found, skipping", flush=True)
            continue

        # Skip graphs already computed in a previous (interrupted) run
        if gname in results and len(results[gname]) == len(NULL_TYPES):
            print(f"\n[{gi+1}/{len(target_names)}] {gname}  — already done, skipping", flush=True)
            continue

        g = load_attribution_graph(name_to_path[gname])
        print(f"\n{'='*80}", flush=True)
        print(
            f"[{gi+1}/{len(target_names)}] {gname}  "
            f"({g.vcount()} nodes, {g.ecount()} edges)",
            flush=True,
        )
        print(f"{'='*80}", flush=True)

        results[gname] = {}

        for ni, null_type in enumerate(NULL_TYPES):
            t0 = time.time()
            nt_short = NULL_SHORT[null_type]
            print(
                f"  [{ni+1}/{len(NULL_TYPES)}] {nt_short:12s} ...",
                end="", flush=True,
            )

            real_counts, z_scores, mean_null, std_null = compute_unrolled_zscores(
                g, null_type,
                n_random=n_random,
                n_workers=n_workers,
                templates=templates,
            )
            elapsed = time.time() - t0

            results[gname][null_type] = {
                "real_counts": real_counts,
                "z_scores": z_scores,
                "mean_null": mean_null,
                "std_null": std_null,
            }

            # One-line summary
            nonzero = {k: v for k, v in z_scores.items() if abs(v) > 0.1}
            if nonzero:
                top = max(nonzero, key=lambda k: abs(nonzero[k]))
                top_sn = SHORT_NAMES.get(top, top[:15])
                print(
                    f" {elapsed:6.1f}s  "
                    f"top: {top_sn:15s} z={z_scores[top]:+7.1f}  "
                    f"real={real_counts[top]:5d}  null={mean_null[top]:7.1f}",
                    flush=True,
                )
            else:
                print(f" {elapsed:6.1f}s  all z-scores ~0", flush=True)

        # Incremental save + summary after each graph
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  >> saved ({len(results)} graphs so far)", flush=True)
        print_summary(results, template_names)

        # Elapsed estimate
        graphs_done = gi + 1
        elapsed_total = time.time() - total_t0
        avg_per_graph = elapsed_total / graphs_done
        remaining = (len(target_names) - graphs_done) * avg_per_graph
        print(
            f"  >> {elapsed_total:.0f}s elapsed, "
            f"~{remaining:.0f}s remaining ({remaining/60:.0f} min)",
            flush=True,
        )

    total_elapsed = time.time() - total_t0
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)", flush=True)

    # ── Final save + summary ─────────────────────────────────────────
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFinal save: {out_path} ({len(results)} graphs)", flush=True)
    print_summary(results, template_names)
    print("Done!", flush=True)


if __name__ == "__main__":
    main()
