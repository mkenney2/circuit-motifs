"""Null model generation and Z-score computation for motif analysis.

Implements degree-preserving edge rewiring (configuration model) and
Erdos-Renyi random graphs as null models for motif enrichment testing.
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
        null_type: Type of null model ("configuration" or "erdos_renyi").
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
