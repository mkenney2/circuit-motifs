"""Null model generation and Z-score computation for motif analysis.

Implements four null models for motif enrichment testing:
  1. Configuration model (degree-preserving rewiring)
  2. Erdos-Renyi random graphs
  3. Layer-pair ER (preserves edge count per layer pair)
  4. Layer-pair configuration (preserves degree sequence per layer pair)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import igraph as ig
import numpy as np
from tqdm import tqdm

from src.motif_census import MotifCensusResult, compute_motif_census


@dataclass
class NullModelResult:
    """Result of null model computation with Z-scores and significance profiles.

    Attributes:
        real_counts: Raw motif counts from the real graph.
        null_counts: Array of motif counts from null ensemble, shape (n_random, n_classes).
        z_scores: Z-scores per motif class.
        significance_profile: Normalized Z-score vector (unit length).
        mean_null: Mean counts across null ensemble per class.
        std_null: Standard deviation across null ensemble per class.
        n_random: Number of random graphs in the null ensemble.
        null_type: Type of null model ("configuration", "erdos_renyi",
            "layer_preserving", or "layer_pair_config").
    """
    real_counts: np.ndarray
    null_counts: np.ndarray
    z_scores: np.ndarray
    significance_profile: np.ndarray
    mean_null: np.ndarray
    std_null: np.ndarray
    n_random: int
    null_type: str


def generate_configuration_null(
    graph: ig.Graph,
    n_random: int = 1000,
    motif_size: int = 3,
    rewire_factor: int = 10,
    show_progress: bool = True,
) -> NullModelResult:
    """Generate a degree-preserving null ensemble and compute Z-scores.

    Uses igraph's rewire() method which preserves the in-degree and out-degree
    sequence while randomizing edge targets. Each random graph is rewired with
    n_edges * rewire_factor swap attempts for thorough mixing.

    Args:
        graph: The real directed igraph.Graph.
        n_random: Number of random graphs to generate.
        motif_size: Motif size (3 or 4).
        rewire_factor: Multiplier for edge count to determine rewiring attempts.
        show_progress: Whether to show a progress bar.

    Returns:
        NullModelResult with Z-scores and significance profile.
    """
    # Compute real motif counts
    real_result = compute_motif_census(graph, size=motif_size)
    real_counts = real_result.as_array()

    n_rewires = max(graph.ecount() * rewire_factor, 1)

    # Generate null ensemble
    null_counts_list: list[np.ndarray] = []
    iterator = range(n_random)
    if show_progress:
        iterator = tqdm(iterator, desc="Null model", unit="graph")

    for _ in iterator:
        g_random = graph.copy()
        g_random.rewire(n=n_rewires)
        null_result = compute_motif_census(g_random, size=motif_size)
        null_counts_list.append(null_result.as_array())

    null_counts = np.array(null_counts_list)

    # Compute Z-scores
    z_scores, mean_null, std_null = _compute_z_scores(real_counts, null_counts)

    # Compute significance profile
    sp = _compute_significance_profile(z_scores)

    return NullModelResult(
        real_counts=real_counts,
        null_counts=null_counts,
        z_scores=z_scores,
        significance_profile=sp,
        mean_null=mean_null,
        std_null=std_null,
        n_random=n_random,
        null_type="configuration",
    )


def generate_erdos_renyi_null(
    graph: ig.Graph,
    n_random: int = 1000,
    motif_size: int = 3,
    show_progress: bool = True,
) -> NullModelResult:
    """Generate an Erdos-Renyi null ensemble and compute Z-scores.

    Creates random directed graphs with the same node count and edge count
    as the real graph, but without preserving degree distribution.

    Args:
        graph: The real directed igraph.Graph.
        n_random: Number of random graphs to generate.
        motif_size: Motif size (3 or 4).
        show_progress: Whether to show a progress bar.

    Returns:
        NullModelResult with Z-scores and significance profile.
    """
    real_result = compute_motif_census(graph, size=motif_size)
    real_counts = real_result.as_array()

    n_nodes = graph.vcount()
    n_edges = graph.ecount()

    null_counts_list: list[np.ndarray] = []
    iterator = range(n_random)
    if show_progress:
        iterator = tqdm(iterator, desc="ER null model", unit="graph")

    for _ in iterator:
        g_random = ig.Graph.Erdos_Renyi(n=n_nodes, m=n_edges, directed=True)
        null_result = compute_motif_census(g_random, size=motif_size)
        null_counts_list.append(null_result.as_array())

    null_counts = np.array(null_counts_list)
    z_scores, mean_null, std_null = _compute_z_scores(real_counts, null_counts)
    sp = _compute_significance_profile(z_scores)

    return NullModelResult(
        real_counts=real_counts,
        null_counts=null_counts,
        z_scores=z_scores,
        significance_profile=sp,
        mean_null=mean_null,
        std_null=std_null,
        n_random=n_random,
        null_type="erdos_renyi",
    )


def generate_layer_preserving_null(
    graph: ig.Graph,
    n_random: int = 1000,
    motif_size: int = 3,
    show_progress: bool = True,
) -> NullModelResult:
    """Generate a layer-pair-preserving null ensemble and compute Z-scores.

    For each (source_layer, target_layer) pair in the real graph, generates
    a random bipartite graph between the same node groups with the same
    edge count, but randomized connectivity. This preserves:
      - DAG structure (forward-only edges)
      - Edge budget per layer pair (architectural skeleton)
      - Total node and edge counts
    While randomizing:
      - Degree distribution within each layer pair
      - Specific connectivity patterns (hubs, fan-in/fan-out structure)

    This is the appropriate null for layered DAGs where degree-preserving
    rewiring (configuration model) produces identical motif counts because
    the degree sequence fully determines the triad census.

    Requires that graph vertices have a 'layer' attribute (set by
    graph_loader.load_attribution_graph).

    Args:
        graph: The real directed igraph.Graph with layer attributes.
        n_random: Number of random graphs to generate.
        motif_size: Motif size (3 or 4).
        show_progress: Whether to show a progress bar.

    Returns:
        NullModelResult with Z-scores and significance profile.
    """
    from src.unrolled_motifs import get_effective_layer
    from collections import defaultdict

    # Compute real motif counts
    real_result = compute_motif_census(graph, size=motif_size)
    real_counts = real_result.as_array()

    n_nodes = graph.vcount()

    # Precompute layer assignments and edge budget per layer pair
    layers = [get_effective_layer(graph, v.index) for v in graph.vs]
    nodes_by_layer: dict[int, list[int]] = defaultdict(list)
    for v_idx, layer in enumerate(layers):
        nodes_by_layer[layer].append(v_idx)

    # Count edges per (src_layer, tgt_layer) pair
    layer_pair_edges: dict[tuple[int, int], int] = defaultdict(int)
    for e in graph.es:
        src_l = layers[e.source]
        tgt_l = layers[e.target]
        layer_pair_edges[(src_l, tgt_l)] += 1

    # Generate null ensemble
    null_counts_list: list[np.ndarray] = []
    iterator = range(n_random)
    if show_progress:
        iterator = tqdm(iterator, desc="Layer-pair null", unit="graph")

    for i in iterator:
        rng = np.random.default_rng(seed=i)
        g_random = ig.Graph(n=n_nodes, directed=True)

        # Copy vertex attributes (needed for motif census to work)
        for attr in graph.vs.attributes():
            g_random.vs[attr] = graph.vs[attr]

        # For each layer pair, generate random bipartite edges
        for (src_l, tgt_l), n_edges in layer_pair_edges.items():
            src_nodes = nodes_by_layer[src_l]
            tgt_nodes = nodes_by_layer[tgt_l]

            n_src = len(src_nodes)
            n_tgt = len(tgt_nodes)
            max_possible = n_src * n_tgt

            if n_edges >= max_possible:
                # Complete bipartite — add all edges
                edges = [(s, t) for s in src_nodes for t in tgt_nodes]
            else:
                # Sample random edges without replacement
                # Encode each possible edge as a single integer for efficiency
                edge_indices = rng.choice(max_possible, size=n_edges, replace=False)
                edges = [
                    (src_nodes[idx // n_tgt], tgt_nodes[idx % n_tgt])
                    for idx in edge_indices
                ]

            g_random.add_edges(edges)

        null_result = compute_motif_census(g_random, size=motif_size)
        null_counts_list.append(null_result.as_array())

    null_counts = np.array(null_counts_list)

    # Compute Z-scores
    z_scores, mean_null, std_null = _compute_z_scores(real_counts, null_counts)

    # Compute significance profile
    sp = _compute_significance_profile(z_scores)

    return NullModelResult(
        real_counts=real_counts,
        null_counts=null_counts,
        z_scores=z_scores,
        significance_profile=sp,
        mean_null=mean_null,
        std_null=std_null,
        n_random=n_random,
        null_type="layer_preserving",
    )


def generate_layer_pair_config_null(
    graph: ig.Graph,
    n_random: int = 1000,
    motif_size: int = 3,
    rewire_factor: int = 10,
    show_progress: bool = True,
) -> NullModelResult:
    """Generate a layer-pair configuration null ensemble and compute Z-scores.

    Preserves BOTH the edge count per (source_layer, target_layer) pair AND
    the degree sequence within each pair. This is the strictest null model
    for layered DAGs: it controls for the architectural skeleton (which layers
    connect) and for hub structure (degree heterogeneity), while randomizing
    which specific nodes connect.

    Algorithm: for each layer pair, perform bipartite degree-preserving
    edge swaps. Pick two random edges (s1->t1, s2->t2), require s1!=s2
    and t1!=t2 and that (s1->t2) and (s2->t1) don't already exist, then
    swap. Repeat n_edges * rewire_factor attempts per pair.

    Requires that graph vertices have a 'layer' attribute.

    Args:
        graph: The real directed igraph.Graph with layer attributes.
        n_random: Number of random graphs to generate.
        motif_size: Motif size (3 or 4).
        rewire_factor: Multiplier for edge count to determine swap attempts.
        show_progress: Whether to show a progress bar.

    Returns:
        NullModelResult with Z-scores, significance profile, and
        acceptance_rate metadata.
    """
    from src.unrolled_motifs import get_effective_layer
    from collections import defaultdict

    # Compute real motif counts
    real_result = compute_motif_census(graph, size=motif_size)
    real_counts = real_result.as_array()

    n_nodes = graph.vcount()

    # Precompute layer assignments and nodes per layer
    layers = [get_effective_layer(graph, v.index) for v in graph.vs]
    nodes_by_layer: dict[int, list[int]] = defaultdict(list)
    for v_idx, layer in enumerate(layers):
        nodes_by_layer[layer].append(v_idx)

    # Group edges by (src_layer, tgt_layer) pair
    layer_pair_edges: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for e in graph.es:
        src_l = layers[e.source]
        tgt_l = layers[e.target]
        layer_pair_edges[(src_l, tgt_l)].append((e.source, e.target))

    # Generate null ensemble
    null_counts_list: list[np.ndarray] = []
    total_accepted = 0
    total_attempted = 0

    iterator = range(n_random)
    if show_progress:
        iterator = tqdm(iterator, desc="Layer-pair config null", unit="graph")

    for i in iterator:
        rng = np.random.default_rng(seed=i)
        g_random = ig.Graph(n=n_nodes, directed=True)

        # Copy vertex attributes
        for attr in graph.vs.attributes():
            g_random.vs[attr] = graph.vs[attr]

        graph_accepted = 0
        graph_attempted = 0

        # For each layer pair, rewire edges preserving degree sequence
        for (src_l, tgt_l), edges in layer_pair_edges.items():
            n_pair_edges = len(edges)
            if n_pair_edges < 2:
                # Can't rewire with fewer than 2 edges
                g_random.add_edges(edges)
                continue

            # Work with a mutable copy of edges for this pair
            current_edges = list(edges)
            # Build edge set for O(1) existence checks
            edge_set = set(current_edges)

            n_attempts = max(n_pair_edges * rewire_factor, 1)
            pair_accepted = 0

            for _ in range(n_attempts):
                # Pick two random edge indices
                idx1, idx2 = rng.choice(n_pair_edges, size=2, replace=False)
                s1, t1 = current_edges[idx1]
                s2, t2 = current_edges[idx2]

                # Require distinct sources and targets (otherwise no real swap)
                if s1 == s2 or t1 == t2:
                    continue

                # Check that swapped edges don't already exist
                new_e1 = (s1, t2)
                new_e2 = (s2, t1)
                if new_e1 in edge_set or new_e2 in edge_set:
                    continue

                # Perform swap
                edge_set.discard((s1, t1))
                edge_set.discard((s2, t2))
                edge_set.add(new_e1)
                edge_set.add(new_e2)
                current_edges[idx1] = new_e1
                current_edges[idx2] = new_e2
                pair_accepted += 1

            graph_accepted += pair_accepted
            graph_attempted += n_attempts
            g_random.add_edges(current_edges)

        total_accepted += graph_accepted
        total_attempted += graph_attempted

        null_result = compute_motif_census(g_random, size=motif_size)
        null_counts_list.append(null_result.as_array())

    null_counts = np.array(null_counts_list)

    # Compute Z-scores
    z_scores, mean_null, std_null = _compute_z_scores(real_counts, null_counts)

    # Compute significance profile
    sp = _compute_significance_profile(z_scores)

    result = NullModelResult(
        real_counts=real_counts,
        null_counts=null_counts,
        z_scores=z_scores,
        significance_profile=sp,
        mean_null=mean_null,
        std_null=std_null,
        n_random=n_random,
        null_type="layer_pair_config",
    )

    # Attach acceptance rate as extra attribute
    if total_attempted > 0:
        result.acceptance_rate = total_accepted / total_attempted
    else:
        result.acceptance_rate = 0.0

    return result


def _compute_z_scores(
    real_counts: np.ndarray,
    null_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Z-scores from real counts and null ensemble.

    Args:
        real_counts: Array of real motif counts, shape (n_classes,).
        null_counts: Array of null counts, shape (n_random, n_classes).

    Returns:
        Tuple of (z_scores, mean_null, std_null), each shape (n_classes,).
    """
    mean_null = null_counts.mean(axis=0)
    std_null = null_counts.std(axis=0)

    # Avoid division by zero: where std is 0, Z-score is 0 if real == mean, else inf
    with np.errstate(divide="ignore", invalid="ignore"):
        z_scores = np.where(
            std_null > 0,
            (real_counts - mean_null) / std_null,
            np.where(real_counts != mean_null, np.sign(real_counts - mean_null) * np.inf, 0.0),
        )

    return z_scores, mean_null, std_null


def _compute_significance_profile(z_scores: np.ndarray) -> np.ndarray:
    """Compute the Significance Profile (SP) vector from Z-scores.

    SP_i = Z_i / sqrt(sum(Z_j^2))

    This normalizes the Z-score vector to unit length, making profiles
    comparable across networks of different sizes (Milo et al., 2004).

    Args:
        z_scores: Array of Z-scores.

    Returns:
        Normalized SP vector. Zero vector if all Z-scores are zero.
    """
    # Handle inf values by capping at a large finite value
    z_finite = np.where(np.isinf(z_scores), np.sign(z_scores) * 100.0, z_scores)

    norm = np.sqrt(np.sum(z_finite ** 2))
    if norm == 0:
        return np.zeros_like(z_finite)
    return z_finite / norm


def verify_degree_preservation(original: ig.Graph, rewired: ig.Graph) -> bool:
    """Verify that a rewired graph preserves the degree sequence.

    Args:
        original: The original graph.
        rewired: The rewired graph.

    Returns:
        True if in-degree and out-degree sequences match.
    """
    return (
        sorted(original.indegree()) == sorted(rewired.indegree())
        and sorted(original.outdegree()) == sorted(rewired.outdegree())
    )
