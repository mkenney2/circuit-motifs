"""Cross-scale comparison of motif profiles across model sizes.

Provides types and functions for analyzing how motif enrichment patterns
change as model scale increases (e.g., Gemma 3 270M → 27B). Parallel
to comparison.py which handles cross-task comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import linregress, spearmanr
from scipy.cluster.hierarchy import linkage

from src.models import ModelSpec
from src.comparison import TaskProfile
from src.motif_census import TRIAD_LABELS, CONNECTED_TRIAD_INDICES


@dataclass
class ModelProfile:
    """Aggregated motif profile for a single model across all task categories.

    Attributes:
        model_spec: The ModelSpec for this model.
        task_profiles: Dict mapping task name to TaskProfile.
        overall_mean_sp: Mean SP vector across all graphs (all tasks).
        overall_mean_z: Mean Z-score vector across all graphs.
        overall_std_z: Std of Z-scores across all graphs.
        n_total_graphs: Total number of graphs analyzed.
        graph_stats: Summary statistics (mean nodes, edges, etc.).
    """

    model_spec: ModelSpec
    task_profiles: dict[str, TaskProfile] = field(default_factory=dict)
    overall_mean_sp: np.ndarray = field(default_factory=lambda: np.array([]))
    overall_mean_z: np.ndarray = field(default_factory=lambda: np.array([]))
    overall_std_z: np.ndarray = field(default_factory=lambda: np.array([]))
    n_total_graphs: int = 0
    graph_stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScaleTrend:
    """Per-motif trend across model scales.

    Attributes:
        motif_index: igraph isomorphism class index.
        motif_label: Human-readable label (e.g., "030T").
        param_counts: Parameter counts in millions for each model.
        values: Mean metric values at each scale point.
        std_values: Std of metric values at each scale point.
        slope: Linear regression slope (on log10-params).
        r_squared: R-squared of the linear fit.
        p_value: P-value from linear regression.
        spearman_rho: Spearman rank correlation coefficient.
        spearman_p: P-value for the Spearman correlation.
        trend_direction: "increasing", "decreasing", or "flat".
        is_significant: Whether p_value < 0.05 (either test).
    """

    motif_index: int
    motif_label: str
    param_counts: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    std_values: list[float] = field(default_factory=list)
    slope: float = 0.0
    r_squared: float = 0.0
    p_value: float = 1.0
    spearman_rho: float = 0.0
    spearman_p: float = 1.0
    trend_direction: str = "flat"
    is_significant: bool = False


@dataclass
class PhaseTransition:
    """Detected discontinuity in a motif's scaling behavior.

    Attributes:
        motif_index: igraph isomorphism class index.
        motif_label: Human-readable label.
        transition_point: Model size (n_params in millions) where the jump occurs.
        before_mean: Mean value before the transition.
        after_mean: Mean value after the transition.
        effect_size: Cohen's d between adjacent model sizes.
    """

    motif_index: int
    motif_label: str
    transition_point: int
    before_mean: float
    after_mean: float
    effect_size: float


@dataclass
class ScaleComparison:
    """Full results of cross-scale comparison.

    Attributes:
        model_profiles: Dict mapping model_id to ModelProfile.
        scale_trends: Per-motif trends across scales.
        phase_transitions: Detected discontinuities.
        similarity_matrix: Pairwise cosine similarity between models.
        model_names: Ordered model names matching similarity matrix.
        linkage_matrix: Hierarchical clustering linkage matrix.
        ffl_backbone_universal: Whether FFL enrichment holds at all scales.
        ffl_details: Per-model FFL Z-score details.
    """

    model_profiles: dict[str, ModelProfile] = field(default_factory=dict)
    scale_trends: list[ScaleTrend] = field(default_factory=list)
    sp_scale_trends: list[ScaleTrend] = field(default_factory=list)
    phase_transitions: list[PhaseTransition] = field(default_factory=list)
    sp_phase_transitions: list[PhaseTransition] = field(default_factory=list)
    similarity_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    model_names: list[str] = field(default_factory=list)
    linkage_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    ffl_backbone_universal: bool = False
    ffl_details: dict[str, Any] = field(default_factory=dict)


def build_model_profile(
    model_spec: ModelSpec,
    task_profiles: dict[str, TaskProfile],
) -> ModelProfile:
    """Build an aggregated motif profile for a model from its task profiles.

    Args:
        model_spec: ModelSpec for the model.
        task_profiles: Dict mapping task name to TaskProfile.

    Returns:
        ModelProfile with aggregated statistics across all tasks.
    """
    all_sp: list[np.ndarray] = []
    all_z: list[np.ndarray] = []

    for profile in task_profiles.values():
        all_sp.extend(profile.sp_vectors)
        all_z.extend(profile.z_score_vectors)

    n_total = len(all_z)

    if n_total > 0:
        sp_array = np.array(all_sp)
        z_array = np.array(all_z)
        overall_mean_sp = sp_array.mean(axis=0)
        overall_mean_z = z_array.mean(axis=0)
        overall_std_z = z_array.std(axis=0)
    else:
        overall_mean_sp = np.array([])
        overall_mean_z = np.array([])
        overall_std_z = np.array([])

    return ModelProfile(
        model_spec=model_spec,
        task_profiles=task_profiles,
        overall_mean_sp=overall_mean_sp,
        overall_mean_z=overall_mean_z,
        overall_std_z=overall_std_z,
        n_total_graphs=n_total,
    )


def compute_scale_trends(
    model_profiles: dict[str, ModelProfile],
    metric: str = "z_score",
) -> list[ScaleTrend]:
    """Compute per-motif trends across model scales.

    Fits linear regression and computes Spearman correlation on
    log10(params) vs the chosen metric for each motif class.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
            Must contain at least 3 models for meaningful regression.
        metric: "z_score" or "sp" — which vector to analyze.

    Returns:
        List of ScaleTrend, one per motif class.
    """
    # Sort models by param count
    sorted_models = sorted(
        model_profiles.values(),
        key=lambda mp: mp.model_spec.n_params,
    )

    param_counts = [mp.model_spec.n_params for mp in sorted_models]
    log_params = [mp.model_spec.log_params for mp in sorted_models]

    if metric == "z_score":
        vectors = [mp.overall_mean_z for mp in sorted_models]
        std_vectors = [mp.overall_std_z for mp in sorted_models]
    else:
        vectors = [mp.overall_mean_sp for mp in sorted_models]
        std_vectors = [np.zeros_like(mp.overall_mean_sp) for mp in sorted_models]

    if not vectors or len(vectors[0]) == 0:
        return []

    n_motifs = len(vectors[0])
    trends: list[ScaleTrend] = []

    for motif_idx in range(n_motifs):
        values = [float(v[motif_idx]) for v in vectors]
        std_values = [float(s[motif_idx]) for s in std_vectors]
        label = TRIAD_LABELS[motif_idx] if motif_idx < 16 else str(motif_idx)

        trend = ScaleTrend(
            motif_index=motif_idx,
            motif_label=label,
            param_counts=param_counts,
            values=values,
            std_values=std_values,
        )

        # Linear regression on log10(params) vs values
        x = np.array(log_params)
        y = np.array(values)

        if len(x) >= 3 and np.std(y) > 0:
            result = linregress(x, y)
            trend.slope = float(result.slope)
            trend.r_squared = float(result.rvalue ** 2)
            trend.p_value = float(result.pvalue)

            # Spearman rank correlation
            rho, sp_p = spearmanr(x, y)
            trend.spearman_rho = float(rho)
            trend.spearman_p = float(sp_p)

            # Determine direction
            if trend.p_value < 0.05 or trend.spearman_p < 0.05:
                trend.is_significant = True
                trend.trend_direction = "increasing" if trend.slope > 0 else "decreasing"
            else:
                trend.trend_direction = "flat"
        else:
            trend.trend_direction = "flat"

        trends.append(trend)

    return trends


def detect_phase_transitions(
    model_profiles: dict[str, ModelProfile],
    min_effect_size: float = 1.0,
    metric: str = "z_score",
) -> list[PhaseTransition]:
    """Detect discontinuities in motif scaling behavior.

    Computes Cohen's d between adjacent model sizes for each motif.
    A phase transition is flagged when the effect size exceeds
    the threshold.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        min_effect_size: Minimum Cohen's d to flag a transition.
        metric: "z_score" or "sp".

    Returns:
        List of PhaseTransition objects.
    """
    sorted_models = sorted(
        model_profiles.values(),
        key=lambda mp: mp.model_spec.n_params,
    )

    if len(sorted_models) < 2:
        return []

    transitions: list[PhaseTransition] = []

    for i in range(len(sorted_models) - 1):
        mp_before = sorted_models[i]
        mp_after = sorted_models[i + 1]

        if metric == "z_score":
            z_before_list = []
            z_after_list = []
            for tp in mp_before.task_profiles.values():
                z_before_list.extend(tp.z_score_vectors)
            for tp in mp_after.task_profiles.values():
                z_after_list.extend(tp.z_score_vectors)
        else:
            z_before_list = []
            z_after_list = []
            for tp in mp_before.task_profiles.values():
                z_before_list.extend(tp.sp_vectors)
            for tp in mp_after.task_profiles.values():
                z_after_list.extend(tp.sp_vectors)

        if not z_before_list or not z_after_list:
            continue

        z_before = np.array(z_before_list)
        z_after = np.array(z_after_list)
        n_motifs = z_before.shape[1] if z_before.ndim > 1 else 0

        for motif_idx in range(n_motifs):
            vals_before = z_before[:, motif_idx]
            vals_after = z_after[:, motif_idx]

            mean_b = float(np.mean(vals_before))
            mean_a = float(np.mean(vals_after))
            std_b = float(np.std(vals_before, ddof=1)) if len(vals_before) > 1 else 0.0
            std_a = float(np.std(vals_after, ddof=1)) if len(vals_after) > 1 else 0.0

            # Pooled standard deviation for Cohen's d
            n_b = len(vals_before)
            n_a = len(vals_after)
            if n_b + n_a < 3:
                continue

            pooled_var = ((n_b - 1) * std_b ** 2 + (n_a - 1) * std_a ** 2) / (n_b + n_a - 2)
            pooled_std = np.sqrt(pooled_var)

            # Skip if pooled std is effectively zero (identical distributions)
            if pooled_std < 1e-10:
                continue

            cohen_d = abs(mean_a - mean_b) / pooled_std

            if cohen_d >= min_effect_size:
                label = TRIAD_LABELS[motif_idx] if motif_idx < 16 else str(motif_idx)
                transitions.append(PhaseTransition(
                    motif_index=motif_idx,
                    motif_label=label,
                    transition_point=mp_after.model_spec.n_params,
                    before_mean=mean_b,
                    after_mean=mean_a,
                    effect_size=float(cohen_d),
                ))

    return transitions


def pairwise_model_similarity(
    model_profiles: dict[str, ModelProfile],
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity matrix between model profiles.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.

    Returns:
        Tuple of (similarity matrix, ordered model names).
    """
    names = sorted(model_profiles.keys())
    n = len(names)
    sim_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                sp_i = model_profiles[names[i]].overall_mean_sp
                sp_j = model_profiles[names[j]].overall_mean_sp
                if len(sp_i) > 0 and len(sp_j) > 0:
                    sim_matrix[i, j] = 1.0 - cosine(sp_i, sp_j)
                else:
                    sim_matrix[i, j] = 0.0

    return sim_matrix, names


def check_ffl_backbone(
    model_profiles: dict[str, ModelProfile],
    ffl_index: int = 7,
    enrichment_threshold: float = 2.0,
) -> tuple[bool, dict[str, Any]]:
    """Check whether the FFL backbone is universal across all model scales.

    The central research question: is 030T (feedforward loop) enriched
    at every model scale?

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        ffl_index: igraph isoclass index for FFL (default 7 = 030T).
        enrichment_threshold: Z-score threshold for enrichment.

    Returns:
        Tuple of:
        - bool: True if FFL is enriched (mean Z > threshold) at all scales.
        - dict: Per-model details with mean_z, std_z, pct_enriched.
    """
    details: dict[str, Any] = {}
    all_enriched = True

    for model_id, mp in model_profiles.items():
        if len(mp.overall_mean_z) == 0 or ffl_index >= len(mp.overall_mean_z):
            details[model_id] = {"mean_z": None, "enriched": False}
            all_enriched = False
            continue

        mean_z = float(mp.overall_mean_z[ffl_index])
        std_z = float(mp.overall_std_z[ffl_index]) if len(mp.overall_std_z) > ffl_index else 0.0

        # Count per-graph enrichment
        n_enriched = 0
        n_total = 0
        for tp in mp.task_profiles.values():
            for z_vec in tp.z_score_vectors:
                n_total += 1
                if z_vec[ffl_index] > enrichment_threshold:
                    n_enriched += 1

        pct = (100.0 * n_enriched / n_total) if n_total > 0 else 0.0
        enriched = mean_z > enrichment_threshold

        details[model_id] = {
            "mean_z": mean_z,
            "std_z": std_z,
            "n_enriched": n_enriched,
            "n_total": n_total,
            "pct_enriched": pct,
            "enriched": enriched,
        }

        if not enriched:
            all_enriched = False

    return all_enriched, details


def compare_task_across_scales(
    model_profiles: dict[str, ModelProfile],
    task_name: str,
    metric: str = "z_score",
) -> list[ScaleTrend]:
    """Compute scaling trends for a specific task category.

    Filters each model's profile to just the given task, then computes
    per-motif trends across scales.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        task_name: Task category to analyze (e.g., "arithmetic").
        metric: "z_score" or "sp".

    Returns:
        List of ScaleTrend for the specified task.
    """
    # Build filtered model profiles with only the specified task
    filtered: dict[str, ModelProfile] = {}
    for model_id, mp in model_profiles.items():
        if task_name in mp.task_profiles:
            tp = mp.task_profiles[task_name]
            filtered_mp = build_model_profile(
                mp.model_spec, {task_name: tp}
            )
            filtered[model_id] = filtered_mp

    if len(filtered) < 2:
        return []

    return compute_scale_trends(filtered, metric=metric)


def run_scale_comparison(
    model_profiles: dict[str, ModelProfile],
) -> ScaleComparison:
    """Run the full cross-scale comparison.

    Orchestrates all scale analysis functions and returns a single
    ScaleComparison result container.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.

    Returns:
        ScaleComparison with trends, transitions, similarity, and FFL analysis.
    """
    # Scale trends (both metrics)
    trends = compute_scale_trends(model_profiles, metric="z_score")
    sp_trends = compute_scale_trends(model_profiles, metric="sp")

    # Phase transitions (both metrics)
    transitions = detect_phase_transitions(model_profiles, metric="z_score")
    sp_transitions = detect_phase_transitions(model_profiles, metric="sp")

    # Pairwise similarity (already SP-based)
    sim_matrix, model_names = pairwise_model_similarity(model_profiles)

    # Hierarchical clustering (if >= 2 models)
    if len(model_names) >= 2:
        distances = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                distances.append(1.0 - sim_matrix[i, j])
        if distances:
            linkage_matrix = linkage(np.array(distances), method="average")
        else:
            linkage_matrix = np.array([])
    else:
        linkage_matrix = np.array([])

    # FFL backbone check
    ffl_universal, ffl_details = check_ffl_backbone(model_profiles)

    return ScaleComparison(
        model_profiles=model_profiles,
        scale_trends=trends,
        sp_scale_trends=sp_trends,
        phase_transitions=transitions,
        sp_phase_transitions=sp_transitions,
        similarity_matrix=sim_matrix,
        model_names=model_names,
        linkage_matrix=linkage_matrix,
        ffl_backbone_universal=ffl_universal,
        ffl_details=ffl_details,
    )
