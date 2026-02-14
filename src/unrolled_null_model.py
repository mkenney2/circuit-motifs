"""Layer-preserving null model for unrolled motif enrichment analysis.

Implements degree-preserving edge rewiring that maintains forward layer
ordering and edge sign distribution. This ensures that enrichment scores
for unrolled motifs reflect genuine structural preferences, not just the
architectural constraint that edges go forward in layer index.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np
from tqdm import tqdm

from src.unrolled_motifs import (
    UnrolledMotifTemplate,
    get_effective_layer,
    build_catalog,
)
from src.unrolled_census import (
    UnrolledMotifInstance,
    find_unrolled_instances,
    run_unrolled_census,
    unrolled_census_counts,
)


@dataclass
class UnrolledNullResult:
    """Result of null model computation for unrolled motifs.

    Attributes:
        real_counts: Dict mapping motif name to instance count in real graph.
        null_counts: Dict mapping motif name to list of counts across null ensemble.
        z_scores: Dict mapping motif name to Z-score.
        sp: Dict mapping motif name to Significance Profile value.
            SP_i = Z_i / sqrt(sum(Z_j^2)). Normalizes to unit length so
            profiles are comparable across graphs of different sizes.
        mean_null: Dict mapping motif name to mean count in null ensemble.
        std_null: Dict mapping motif name to std of count in null ensemble.
        n_random: Number of random graphs in the null ensemble.
        acceptance_rate: Fraction of proposed swaps accepted during rewiring.
    """
    real_counts: dict[str, int]
    null_counts: dict[str, list[int]]
    z_scores: dict[str, float]
    sp: dict[str, float]
    mean_null: dict[str, float]
    std_null: dict[str, float]
    n_random: int
    acceptance_rate: float


def layer_preserving_rewire(
    graph: ig.Graph,
    n_swaps: int | None = None,
    max_attempts_factor: int = 10,
    seed: int | None = None,
    preserve_signs: bool = True,
) -> tuple[ig.Graph, float]:
    """Degree-preserving random rewiring that maintains forward layer ordering.

    Uses the standard edge-switching algorithm but rejects swaps that would:
    1. Create backward edges (target layer <= source layer)
    2. Create multi-edges (duplicate edges between same node pair)
    3. Create self-loops
    4. Mix excitatory and inhibitory edges (if preserve_signs=True)

    Args:
        graph: A directed igraph.Graph with layer attributes.
        n_swaps: Number of successful swaps to perform. Defaults to edge count.
        max_attempts_factor: Maximum attempts = n_swaps * this factor.
        seed: Random seed for reproducibility.
        preserve_signs: If True, only swap edges with matching signs.

    Returns:
        Tuple of (rewired graph, acceptance rate).
    """
    g = graph.copy()
    rng = np.random.default_rng(seed)

    if n_swaps is None:
        n_swaps = g.ecount()

    max_attempts = n_swaps * max_attempts_factor
    n_edges = g.ecount()

    if n_edges < 2:
        return g, 0.0

    has_sign = "sign" in g.es.attributes()
    has_raw_weight = "raw_weight" in g.es.attributes()
    has_weight = "weight" in g.es.attributes()

    # Precompute effective layers for all nodes
    layers = [get_effective_layer(g, v.index) for v in g.vs]

    n_accepted = 0
    n_attempted = 0

    while n_accepted < n_swaps and n_attempted < max_attempts:
        n_attempted += 1

        # Pick two random edges
        e1_idx, e2_idx = rng.choice(n_edges, size=2, replace=False)
        e1 = g.es[e1_idx]
        e2 = g.es[e2_idx]

        # If preserving signs, only swap edges with matching signs
        if preserve_signs and has_sign:
            if e1["sign"] != e2["sign"]:
                continue

        s1, t1 = e1.source, e1.target
        s2, t2 = e2.source, e2.target

        # Avoid if edges share nodes (would create self-loops or degenerate swaps)
        if len({s1, t1, s2, t2}) < 4:
            continue

        # Proposed swap: (s1→t1, s2→t2) → (s1→t2, s2→t1)
        # Check layer ordering is preserved
        if layers[s1] >= layers[t2] or layers[s2] >= layers[t1]:
            continue

        # Check no multi-edges would be created
        if g.are_adjacent(s1, t2) or g.are_adjacent(s2, t1):
            continue

        # Perform the swap: save attributes, delete old edges, add new ones
        # Save edge attributes before deletion
        e1_attrs = {attr: e1[attr] for attr in g.es.attributes()}
        e2_attrs = {attr: e2[attr] for attr in g.es.attributes()}

        # Delete edges (delete higher index first to avoid index shift)
        to_delete = sorted([e1_idx, e2_idx], reverse=True)
        g.delete_edges(to_delete)

        # Add new edges with swapped targets, preserving original attributes
        g.add_edge(s1, t2, **e1_attrs)
        g.add_edge(s2, t1, **e2_attrs)

        # Update edge count (should be same, but recalculate indices)
        n_edges = g.ecount()
        n_accepted += 1

    acceptance_rate = n_accepted / n_attempted if n_attempted > 0 else 0.0
    return g, acceptance_rate


def _null_iteration_worker(
    args: tuple[ig.Graph, list[UnrolledMotifTemplate], float, int | None, int | None],
) -> tuple[dict[str, int], float]:
    """Worker for parallel null model iterations.

    Defined at module level so it is picklable on Windows (spawn).

    Args:
        args: Tuple of (graph, templates, weight_threshold, max_layer_gap, seed).

    Returns:
        Tuple of (motif counts dict, acceptance rate).
    """
    graph, templates, weight_threshold, max_layer_gap, seed = args
    g_null, acc_rate = layer_preserving_rewire(graph, seed=seed)
    null_census = run_unrolled_census(
        g_null, templates,
        weight_threshold=weight_threshold,
        max_layer_gap=max_layer_gap,
    )
    null_c = unrolled_census_counts(null_census)
    return null_c, acc_rate


def compute_unrolled_zscores(
    graph: ig.Graph,
    templates: list[UnrolledMotifTemplate] | None = None,
    n_random: int = 100,
    weight_threshold: float = 0.0,
    max_layer_gap: int | None = None,
    show_progress: bool = True,
    seed: int | None = None,
    n_jobs: int = 1,
) -> UnrolledNullResult:
    """Compute Z-scores for unrolled motifs against a layer-preserving null model.

    For each template:
    1. Count instances in the real graph
    2. Count instances in n_random layer-preserving rewirings
    3. Compute Z = (real - mean_null) / std_null

    Args:
        graph: A directed igraph.Graph.
        templates: List of templates. Defaults to full catalog.
        n_random: Number of null model random graphs.
        weight_threshold: Minimum absolute edge weight for matching.
        max_layer_gap: Maximum layer gap per edge.
        show_progress: Whether to show a progress bar.
        seed: Base random seed (each null graph uses seed + i).
        n_jobs: Number of parallel workers. 1 = sequential (default).
            Use -1 for all available CPU cores.

    Returns:
        UnrolledNullResult with Z-scores and null ensemble statistics.
    """
    if templates is None:
        templates = build_catalog()

    # Count instances in real graph
    real_census = run_unrolled_census(
        graph, templates,
        weight_threshold=weight_threshold,
        max_layer_gap=max_layer_gap,
    )
    real_counts = unrolled_census_counts(real_census)

    # Initialize null count tracking
    null_counts: dict[str, list[int]] = {t.name: [] for t in templates}
    total_acceptance = 0.0

    # Build argument list for all iterations
    args_list = [
        (graph, templates, weight_threshold, max_layer_gap,
         (seed + i) if seed is not None else None)
        for i in range(n_random)
    ]

    if n_jobs == 1:
        # Sequential execution
        iterator = range(n_random)
        if show_progress:
            iterator = tqdm(iterator, desc="Unrolled null model", unit="graph")

        for i in iterator:
            null_c, acc_rate = _null_iteration_worker(args_list[i])
            total_acceptance += acc_rate
            for t in templates:
                null_counts[t.name].append(null_c[t.name])
    else:
        # Parallel execution
        import os
        if n_jobs == -1:
            max_workers = os.cpu_count()
        else:
            max_workers = n_jobs

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_null_iteration_worker, a)
                for a in args_list
            ]

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(
                    iterator, total=n_random,
                    desc=f"Unrolled null model ({max_workers} workers)",
                    unit="graph",
                )

            for future in iterator:
                null_c, acc_rate = future.result()
                total_acceptance += acc_rate
                for t in templates:
                    null_counts[t.name].append(null_c[t.name])

    # Compute Z-scores
    z_scores: dict[str, float] = {}
    mean_null: dict[str, float] = {}
    std_null: dict[str, float] = {}

    for t in templates:
        obs = real_counts[t.name]
        null_arr = np.array(null_counts[t.name], dtype=np.float64)
        m = float(null_arr.mean())
        s = float(null_arr.std())
        mean_null[t.name] = m
        std_null[t.name] = s

        if s > 1e-10:
            z_scores[t.name] = (obs - m) / s
        else:
            # If std is ~0, Z is 0 if obs == mean, else signed infinity
            if abs(obs - m) < 1e-10:
                z_scores[t.name] = 0.0
            else:
                z_scores[t.name] = float(np.sign(obs - m)) * 100.0

    # Compute Significance Profile: SP_i = Z_i / sqrt(sum(Z_j^2))
    z_arr = np.array([z_scores[t.name] for t in templates])
    z_finite = np.where(np.isinf(z_arr), np.sign(z_arr) * 100.0, z_arr)
    norm = np.sqrt(np.sum(z_finite ** 2))
    if norm > 0:
        sp_arr = z_finite / norm
    else:
        sp_arr = np.zeros_like(z_finite)
    sp = {t.name: float(sp_arr[i]) for i, t in enumerate(templates)}

    avg_acceptance = total_acceptance / n_random if n_random > 0 else 0.0

    return UnrolledNullResult(
        real_counts=real_counts,
        null_counts=null_counts,
        z_scores=z_scores,
        sp=sp,
        mean_null=mean_null,
        std_null=std_null,
        n_random=n_random,
        acceptance_rate=avg_acceptance,
    )


def verify_layer_preservation(original: ig.Graph, rewired: ig.Graph) -> bool:
    """Verify that a rewired graph preserves layer ordering on all edges.

    Args:
        original: The original graph.
        rewired: The rewired graph.

    Returns:
        True if all edges in the rewired graph go forward in layer index.
    """
    for edge in rewired.es:
        src_layer = get_effective_layer(rewired, edge.source)
        tgt_layer = get_effective_layer(rewired, edge.target)
        if src_layer >= tgt_layer:
            return False
    return True


def verify_sign_preservation(original: ig.Graph, rewired: ig.Graph) -> bool:
    """Verify that a rewired graph preserves the edge sign distribution.

    Args:
        original: The original graph.
        rewired: The rewired graph.

    Returns:
        True if the count of excitatory and inhibitory edges matches.
    """
    if "sign" not in original.es.attributes() or "sign" not in rewired.es.attributes():
        return True

    orig_signs = sorted(original.es["sign"])
    rewired_signs = sorted(rewired.es["sign"])
    return orig_signs == rewired_signs
