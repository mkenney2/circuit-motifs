"""Tests for src/scale_comparison.py — cross-scale analysis."""

import numpy as np
import pytest

from src.models import ModelSpec, TranscoderConfig, GEMMA_3_MODELS, get_model
from src.comparison import TaskProfile
from src.scale_comparison import (
    ModelProfile,
    ScaleTrend,
    PhaseTransition,
    build_model_profile,
    compute_scale_trends,
    detect_phase_transitions,
    pairwise_model_similarity,
    check_ffl_backbone,
    compare_task_across_scales,
    run_scale_comparison,
)


# ── Fixtures ─────────────────────────────────────────────────────────────

def _make_z_vector(ffl_z: float = 10.0, chain_z: float = 5.0, seed: int = 0) -> np.ndarray:
    """Create a 16-element Z-score vector with controlled values."""
    rng = np.random.RandomState(seed)
    z = rng.normal(0, 2, 16)  # wider spread for realistic variance
    z[7] = ffl_z + rng.normal(0, 0.5)   # 030T with noise
    z[4] = chain_z + rng.normal(0, 0.5)  # 021C with noise
    return z


def _make_sp_vector(z: np.ndarray) -> np.ndarray:
    """Compute SP from a Z-score vector."""
    norm = np.sqrt(np.sum(z ** 2))
    if norm == 0:
        return np.zeros_like(z)
    return z / norm


def _make_task_profile(
    task_name: str,
    n_graphs: int = 5,
    ffl_z: float = 10.0,
    chain_z: float = 5.0,
    base_seed: int = 0,
) -> TaskProfile:
    """Create a synthetic TaskProfile with controlled Z-scores."""
    sp_vectors = []
    z_vectors = []
    for i in range(n_graphs):
        z = _make_z_vector(ffl_z=ffl_z, chain_z=chain_z, seed=base_seed + i)
        sp = _make_sp_vector(z)
        z_vectors.append(z)
        sp_vectors.append(sp)

    z_arr = np.array(z_vectors)
    sp_arr = np.array(sp_vectors)

    return TaskProfile(
        task_name=task_name,
        sp_vectors=sp_vectors,
        z_score_vectors=z_vectors,
        mean_sp=sp_arr.mean(axis=0),
        std_sp=sp_arr.std(axis=0),
        mean_z=z_arr.mean(axis=0),
        n_graphs=n_graphs,
    )


def _make_model_profile(
    model_id: str,
    ffl_z: float = 10.0,
    chain_z: float = 5.0,
    n_graphs_per_task: int = 3,
    tasks: list[str] | None = None,
) -> ModelProfile:
    """Create a synthetic ModelProfile for testing."""
    spec = get_model(model_id)
    if tasks is None:
        tasks = ["factual_recall", "arithmetic"]
    task_profiles = {}
    for i, task_name in enumerate(tasks):
        task_profiles[task_name] = _make_task_profile(
            task_name,
            n_graphs=n_graphs_per_task,
            ffl_z=ffl_z,
            chain_z=chain_z,
            base_seed=i * 100,
        )
    return build_model_profile(spec, task_profiles)


def _make_scaling_profiles(
    ffl_values: list[float] | None = None,
    chain_values: list[float] | None = None,
) -> dict[str, ModelProfile]:
    """Create 5 Gemma 3 ModelProfiles with controlled scaling behavior."""
    model_ids = [
        "gemma-3-270m-it",
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
    ]
    if ffl_values is None:
        ffl_values = [5.0, 10.0, 15.0, 20.0, 25.0]  # monotonic increase
    if chain_values is None:
        chain_values = [8.0, 8.0, 8.0, 8.0, 8.0]  # flat

    profiles = {}
    for i, model_id in enumerate(model_ids):
        profiles[model_id] = _make_model_profile(
            model_id,
            ffl_z=ffl_values[i],
            chain_z=chain_values[i],
        )
    return profiles


# ── Tests ────────────────────────────────────────────────────────────────

class TestBuildModelProfile:
    def test_aggregation_counts(self):
        mp = _make_model_profile("gemma-3-4b-it", n_graphs_per_task=4)
        # 2 tasks x 4 graphs each
        assert mp.n_total_graphs == 8

    def test_mean_z_shape(self):
        mp = _make_model_profile("gemma-3-4b-it")
        assert mp.overall_mean_z.shape == (16,)

    def test_mean_sp_shape(self):
        mp = _make_model_profile("gemma-3-4b-it")
        assert mp.overall_mean_sp.shape == (16,)

    def test_empty_profiles(self):
        spec = get_model("gemma-3-270m-it")
        mp = build_model_profile(spec, {})
        assert mp.n_total_graphs == 0
        assert len(mp.overall_mean_z) == 0

    def test_model_spec_preserved(self):
        mp = _make_model_profile("gemma-3-4b-it")
        assert mp.model_spec.model_id == "gemma-3-4b-it"
        assert mp.model_spec.n_layers == 34


class TestComputeScaleTrends:
    def test_detects_monotonic_increase(self):
        profiles = _make_scaling_profiles(
            ffl_values=[5.0, 10.0, 15.0, 20.0, 25.0],
        )
        trends = compute_scale_trends(profiles, metric="z_score")
        # FFL trend (index 7) should be increasing
        ffl_trend = [t for t in trends if t.motif_index == 7][0]
        assert ffl_trend.trend_direction == "increasing"
        assert ffl_trend.is_significant
        assert ffl_trend.slope > 0

    def test_detects_flat(self):
        profiles = _make_scaling_profiles(
            chain_values=[8.0, 8.0, 8.0, 8.0, 8.0],
        )
        trends = compute_scale_trends(profiles, metric="z_score")
        chain_trend = [t for t in trends if t.motif_index == 4][0]
        assert chain_trend.trend_direction == "flat"
        assert not chain_trend.is_significant

    def test_returns_all_motifs(self):
        profiles = _make_scaling_profiles()
        trends = compute_scale_trends(profiles, metric="z_score")
        assert len(trends) == 16

    def test_param_counts_in_order(self):
        profiles = _make_scaling_profiles()
        trends = compute_scale_trends(profiles, metric="z_score")
        for trend in trends:
            assert trend.param_counts == sorted(trend.param_counts)

    def test_empty_profiles(self):
        trends = compute_scale_trends({}, metric="z_score")
        assert trends == []


class TestDetectPhaseTransitions:
    def test_finds_jump(self):
        # Sharp jump between model 2 and 3 (50x difference should be detectable)
        model_ids = [
            "gemma-3-270m-it",
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ]
        ffl_values = [5.0, 5.0, 5.0, 50.0, 50.0]
        profiles = {}
        for i, model_id in enumerate(model_ids):
            profiles[model_id] = _make_model_profile(
                model_id,
                ffl_z=ffl_values[i],
                chain_z=8.0,
                n_graphs_per_task=10,  # More graphs for reliable variance
            )
        transitions = detect_phase_transitions(profiles, min_effect_size=3.0)
        ffl_transitions = [t for t in transitions if t.motif_index == 7]
        assert len(ffl_transitions) > 0
        # The transition should be at the 12B model
        big_jump = max(ffl_transitions, key=lambda t: t.effect_size)
        assert big_jump.transition_point == 12000

    def test_no_false_positives_on_smooth(self):
        # Very smooth gradual change — differences smaller than noise
        model_ids = [
            "gemma-3-270m-it",
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ]
        ffl_values = [10.0, 10.02, 10.04, 10.06, 10.08]  # very tiny steps
        profiles = {}
        for i, model_id in enumerate(model_ids):
            profiles[model_id] = _make_model_profile(
                model_id,
                ffl_z=ffl_values[i],
                chain_z=8.0,
                n_graphs_per_task=10,
            )
        transitions = detect_phase_transitions(profiles, min_effect_size=3.0)
        ffl_transitions = [t for t in transitions if t.motif_index == 7]
        assert len(ffl_transitions) == 0

    def test_needs_at_least_two_models(self):
        spec = get_model("gemma-3-4b-it")
        mp = _make_model_profile("gemma-3-4b-it")
        transitions = detect_phase_transitions({"gemma-3-4b-it": mp})
        assert transitions == []


class TestCheckFflBackbone:
    def test_all_enriched(self):
        profiles = _make_scaling_profiles(
            ffl_values=[10.0, 15.0, 20.0, 25.0, 30.0],
        )
        universal, details = check_ffl_backbone(profiles)
        assert universal is True
        for model_id, info in details.items():
            assert info["enriched"] is True

    def test_one_depleted(self):
        profiles = _make_scaling_profiles(
            ffl_values=[10.0, 15.0, -5.0, 25.0, 30.0],
        )
        universal, details = check_ffl_backbone(profiles)
        assert universal is False
        # gemma-3-4b-it should not be enriched
        assert details["gemma-3-4b-it"]["enriched"] is False

    def test_details_have_pct_enriched(self):
        profiles = _make_scaling_profiles(
            ffl_values=[10.0, 15.0, 20.0, 25.0, 30.0],
        )
        _, details = check_ffl_backbone(profiles)
        for info in details.values():
            assert "pct_enriched" in info
            assert 0.0 <= info["pct_enriched"] <= 100.0


class TestPairwiseModelSimilarity:
    def test_symmetric(self):
        profiles = _make_scaling_profiles()
        sim, names = pairwise_model_similarity(profiles)
        assert sim.shape == (5, 5)
        np.testing.assert_array_almost_equal(sim, sim.T)

    def test_diagonal_is_one(self):
        profiles = _make_scaling_profiles()
        sim, _ = pairwise_model_similarity(profiles)
        np.testing.assert_array_almost_equal(np.diag(sim), np.ones(5))

    def test_values_in_range(self):
        profiles = _make_scaling_profiles()
        sim, _ = pairwise_model_similarity(profiles)
        assert np.all(sim >= -1.0 - 1e-10)
        assert np.all(sim <= 1.0 + 1e-10)

    def test_names_sorted(self):
        profiles = _make_scaling_profiles()
        _, names = pairwise_model_similarity(profiles)
        assert names == sorted(names)


class TestCompareTaskAcrossScales:
    def test_returns_trends(self):
        profiles = _make_scaling_profiles()
        trends = compare_task_across_scales(profiles, "factual_recall")
        assert len(trends) == 16

    def test_missing_task_returns_empty(self):
        profiles = _make_scaling_profiles()
        trends = compare_task_across_scales(profiles, "nonexistent_task")
        assert trends == []


class TestRunScaleComparison:
    def test_full_pipeline(self):
        profiles = _make_scaling_profiles()
        result = run_scale_comparison(profiles)

        assert len(result.scale_trends) == 16
        assert len(result.sp_scale_trends) == 16
        assert result.similarity_matrix.shape == (5, 5)
        assert len(result.model_names) == 5
        assert isinstance(result.ffl_backbone_universal, bool)
        assert len(result.ffl_details) == 5
        assert result.linkage_matrix.shape[0] > 0

    def test_sp_trends_computed(self):
        """SP-based trends should be computed alongside Z-score trends."""
        profiles = _make_scaling_profiles()
        result = run_scale_comparison(profiles)

        # SP trends should have same count as Z-score trends
        assert len(result.sp_scale_trends) == len(result.scale_trends)

        # SP values should be normalized (smaller magnitude than Z)
        for sp_t in result.sp_scale_trends:
            for v in sp_t.values:
                assert abs(v) <= 1.0 + 1e-6, f"{sp_t.motif_label} SP={v} exceeds 1.0"

        # SP phase transitions should be a list (may differ from Z-score ones)
        assert isinstance(result.sp_phase_transitions, list)

    def test_two_models(self):
        profiles = _make_scaling_profiles()
        subset = {k: v for k, v in list(profiles.items())[:2]}
        result = run_scale_comparison(subset)
        assert result.similarity_matrix.shape == (2, 2)
        assert len(result.sp_scale_trends) == 16
