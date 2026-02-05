"""Cross-task comparison of motif profiles.

Computes significance profile vectors per task category, runs statistical
tests (Mann-Whitney U, Kruskal-Wallis), and builds similarity matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import kruskal, mannwhitneyu
from scipy.cluster.hierarchy import linkage, leaves_list

from src.null_model import NullModelResult


@dataclass
class TaskProfile:
    """Aggregated motif profile for a task category.

    Attributes:
        task_name: Name of the task category.
        sp_vectors: List of SP vectors, one per graph in the category.
        z_score_vectors: List of Z-score vectors, one per graph.
        mean_sp: Mean SP vector across graphs.
        std_sp: Standard deviation of SP vectors.
        mean_z: Mean Z-score vector.
        n_graphs: Number of graphs in this category.
    """
    task_name: str
    sp_vectors: list[np.ndarray] = field(default_factory=list)
    z_score_vectors: list[np.ndarray] = field(default_factory=list)
    mean_sp: np.ndarray = field(default_factory=lambda: np.array([]))
    std_sp: np.ndarray = field(default_factory=lambda: np.array([]))
    mean_z: np.ndarray = field(default_factory=lambda: np.array([]))
    n_graphs: int = 0


@dataclass
class PairwiseComparison:
    """Result of a pairwise comparison between two task categories.

    Attributes:
        task_a: First task name.
        task_b: Second task name.
        cosine_similarity: Cosine similarity between mean SP vectors.
        per_motif_p_values: P-values from Mann-Whitney U per motif class.
        significant_motifs: Motif indices with p < 0.05.
    """
    task_a: str
    task_b: str
    cosine_similarity: float
    per_motif_p_values: np.ndarray
    significant_motifs: list[int]


def build_task_profile(
    task_name: str,
    null_results: list[NullModelResult],
) -> TaskProfile:
    """Build an aggregated motif profile for a task category.

    Args:
        task_name: Name of the task category.
        null_results: List of NullModelResult, one per graph in the category.

    Returns:
        TaskProfile with aggregated statistics.
    """
    sp_vectors = [r.significance_profile for r in null_results]
    z_vectors = [r.z_scores for r in null_results]

    sp_array = np.array(sp_vectors)
    z_array = np.array(z_vectors)

    return TaskProfile(
        task_name=task_name,
        sp_vectors=sp_vectors,
        z_score_vectors=z_vectors,
        mean_sp=sp_array.mean(axis=0),
        std_sp=sp_array.std(axis=0),
        mean_z=z_array.mean(axis=0),
        n_graphs=len(null_results),
    )


def pairwise_comparison(
    profile_a: TaskProfile,
    profile_b: TaskProfile,
    alpha: float = 0.05,
) -> PairwiseComparison:
    """Compare two task profiles with cosine similarity and per-motif Mann-Whitney U.

    Args:
        profile_a: First task profile.
        profile_b: Second task profile.
        alpha: Significance threshold for Mann-Whitney U.

    Returns:
        PairwiseComparison with similarity and per-motif p-values.
    """
    # Cosine similarity between mean SP vectors
    cos_sim = 1.0 - cosine(profile_a.mean_sp, profile_b.mean_sp)

    # Per-motif Mann-Whitney U test
    n_motifs = len(profile_a.mean_sp)
    p_values = np.ones(n_motifs)
    significant = []

    sp_a = np.array(profile_a.sp_vectors)
    sp_b = np.array(profile_b.sp_vectors)

    for i in range(n_motifs):
        vals_a = sp_a[:, i]
        vals_b = sp_b[:, i]

        # Need at least 2 samples per group for Mann-Whitney
        if len(vals_a) < 2 or len(vals_b) < 2:
            continue

        # Skip if both groups are constant
        if np.std(vals_a) == 0 and np.std(vals_b) == 0:
            continue

        try:
            _, p = mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            p_values[i] = p
            if p < alpha:
                significant.append(i)
        except ValueError:
            pass

    return PairwiseComparison(
        task_a=profile_a.task_name,
        task_b=profile_b.task_name,
        cosine_similarity=float(cos_sim),
        per_motif_p_values=p_values,
        significant_motifs=significant,
    )


def all_pairwise_comparisons(
    profiles: dict[str, TaskProfile],
    alpha: float = 0.05,
) -> list[PairwiseComparison]:
    """Run pairwise comparisons between all task category pairs.

    Args:
        profiles: Dict mapping task name to TaskProfile.
        alpha: Significance threshold.

    Returns:
        List of PairwiseComparison results.
    """
    tasks = sorted(profiles.keys())
    results = []
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1:]:
            comp = pairwise_comparison(profiles[task_a], profiles[task_b], alpha=alpha)
            results.append(comp)
    return results


def cosine_similarity_matrix(profiles: dict[str, TaskProfile]) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity matrix between task profiles.

    Args:
        profiles: Dict mapping task name to TaskProfile.

    Returns:
        Tuple of (similarity matrix, ordered task names).
    """
    tasks = sorted(profiles.keys())
    n = len(tasks)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sim_matrix[i, j] = 1.0 - cosine(
                    profiles[tasks[i]].mean_sp,
                    profiles[tasks[j]].mean_sp,
                )

    return sim_matrix, tasks


def kruskal_wallis_per_motif(
    profiles: dict[str, TaskProfile],
) -> list[dict[str, Any]]:
    """Run Kruskal-Wallis test per motif class across all task categories.

    Tests whether the distribution of each motif's SP value differs
    significantly across task categories.

    Args:
        profiles: Dict mapping task name to TaskProfile.

    Returns:
        List of dicts with motif_index, H_statistic, p_value, significant.
    """
    tasks = sorted(profiles.keys())
    n_motifs = len(next(iter(profiles.values())).mean_sp)
    results = []

    for motif_idx in range(n_motifs):
        groups = []
        for task in tasks:
            sp_array = np.array(profiles[task].sp_vectors)
            groups.append(sp_array[:, motif_idx])

        # Need at least 2 groups with >= 1 sample each
        valid_groups = [g for g in groups if len(g) >= 1]
        if len(valid_groups) < 2:
            results.append({
                "motif_index": motif_idx,
                "H_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
            })
            continue

        try:
            h_stat, p_val = kruskal(*valid_groups)
            results.append({
                "motif_index": motif_idx,
                "H_statistic": float(h_stat),
                "p_value": float(p_val),
                "significant": p_val < 0.05,
            })
        except ValueError:
            results.append({
                "motif_index": motif_idx,
                "H_statistic": np.nan,
                "p_value": np.nan,
                "significant": False,
            })

    return results


def hierarchical_clustering(
    profiles: dict[str, TaskProfile],
    method: str = "average",
) -> tuple[np.ndarray, list[str]]:
    """Perform hierarchical clustering on task profiles based on SP vectors.

    Uses cosine distance between mean SP vectors.

    Args:
        profiles: Dict mapping task name to TaskProfile.
        method: Linkage method (e.g., 'average', 'complete', 'ward').

    Returns:
        Tuple of (linkage matrix, ordered task names).
    """
    tasks = sorted(profiles.keys())
    n = len(tasks)

    # Compute condensed distance matrix (cosine distance)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = cosine(profiles[tasks[i]].mean_sp, profiles[tasks[j]].mean_sp)
            distances.append(dist)

    dist_array = np.array(distances)
    Z = linkage(dist_array, method=method)

    return Z, tasks
