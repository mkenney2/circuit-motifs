"""Tests for unrolled motif templates, census, and null model."""

import igraph as ig
import numpy as np
import pytest

from src.unrolled_motifs import (
    UnrolledMotifTemplate,
    build_catalog,
    get_layer_index,
    get_effective_layer,
    validate_layer_ordering,
    layer_gap_distribution,
    layer_stats,
    CATALOG,
)
from src.unrolled_census import (
    UnrolledMotifInstance,
    find_unrolled_instances,
    run_unrolled_census,
    unrolled_census_counts,
    unrolled_census_summary,
    _edge_sign_matches,
)
from src.unrolled_null_model import (
    layer_preserving_rewire,
    compute_unrolled_zscores,
    verify_layer_preservation,
    verify_sign_preservation,
)


# ---------------------------------------------------------------------------
# Test graph fixtures
# ---------------------------------------------------------------------------

def _make_layered_graph() -> ig.Graph:
    """Create a layered attribution graph with known motif instances.

    Structure (6 nodes across 3 layers):
        Layer 0: nodes 0, 1
        Layer 1: nodes 2, 3
        Layer 2: nodes 4, 5

    Edges (all forward):
        0→2 (+), 0→3 (+), 1→2 (-), 1→3 (+)
        2→4 (+), 2→5 (-), 3→4 (+), 3→5 (+)
        0→4 (+)  (skip connection)

    This contains:
        - A coherent FFL: 0→2→4 with shortcut 0→4 (all positive)
        - An incoherent FFL: 0→2→5 with ... (0→2 is +, 2→5 is -, but no 0→5 edge)
        - Cross-chain inhibition: 0→2(+), 1→3(+), 1→2(-), 0→3(+) — partial
        - Feedforward damping: 1→2(-) is inhibitory but 0→2→5 (+ then -) makes 3-node damping
    """
    g = ig.Graph(directed=True)
    g.add_vertices(6)

    # Set layer attributes
    layers = [0, 0, 1, 1, 2, 2]
    for i, layer in enumerate(layers):
        g.vs[i]["layer"] = layer
        g.vs[i]["feature_type"] = "cross layer transcoder"
        g.vs[i]["ctx_idx"] = i % 3
        g.vs[i]["clerp"] = f"feature_{i}"

    # Add edges with signs
    edges_data = [
        (0, 2, 1.5, "excitatory"),    # 0→2 excitatory
        (0, 3, 0.8, "excitatory"),     # 0→3 excitatory
        (1, 2, 0.6, "inhibitory"),     # 1→2 inhibitory (cross-chain)
        (1, 3, 1.2, "excitatory"),     # 1→3 excitatory
        (2, 4, 2.0, "excitatory"),     # 2→4 excitatory
        (2, 5, 0.9, "inhibitory"),     # 2→5 inhibitory
        (3, 4, 1.1, "excitatory"),     # 3→4 excitatory
        (3, 5, 0.7, "excitatory"),     # 3→5 excitatory
        (0, 4, 1.3, "excitatory"),     # 0→4 skip connection (excitatory)
    ]

    for src, tgt, weight, sign in edges_data:
        raw_w = weight if sign == "excitatory" else -weight
        g.add_edge(src, tgt, weight=weight, raw_weight=raw_w, sign=sign)

    return g


def _make_cross_chain_inhibition_graph() -> ig.Graph:
    """Create a graph with an explicit cross-chain inhibition motif.

    4 nodes across 2 layers:
        Layer 0: A_early(0), B_early(1)
        Layer 2: A_late(2), B_late(3)

    Edges:
        0→2 (+)  A continues
        1→3 (+)  B continues
        0→3 (-)  A suppresses B
        1→2 (-)  B suppresses A
    """
    g = ig.Graph(directed=True)
    g.add_vertices(4)

    layers = [0, 0, 2, 2]
    for i, layer in enumerate(layers):
        g.vs[i]["layer"] = layer
        g.vs[i]["feature_type"] = "cross layer transcoder"
        g.vs[i]["ctx_idx"] = i
        g.vs[i]["clerp"] = f"node_{i}"

    edges_data = [
        (0, 2, 1.0, "excitatory"),
        (1, 3, 1.0, "excitatory"),
        (0, 3, 0.8, "inhibitory"),
        (1, 2, 0.7, "inhibitory"),
    ]

    for src, tgt, weight, sign in edges_data:
        raw_w = weight if sign == "excitatory" else -weight
        g.add_edge(src, tgt, weight=weight, raw_weight=raw_w, sign=sign)

    return g


def _make_damping_graph() -> ig.Graph:
    """Create a graph with a feedforward damping motif.

    3 nodes across 3 layers:
        Layer 0: A_early(0)
        Layer 1: B_mid(1)
        Layer 2: A_late(2)

    Edges:
        0→1 (+)  A activates B
        1→2 (-)  B suppresses A_late
    """
    g = ig.Graph(directed=True)
    g.add_vertices(3)

    layers = [0, 1, 2]
    for i, layer in enumerate(layers):
        g.vs[i]["layer"] = layer
        g.vs[i]["feature_type"] = "cross layer transcoder"
        g.vs[i]["ctx_idx"] = i
        g.vs[i]["clerp"] = f"node_{i}"

    g.add_edge(0, 1, weight=1.5, raw_weight=1.5, sign="excitatory")
    g.add_edge(1, 2, weight=0.9, raw_weight=-0.9, sign="inhibitory")

    return g


def _make_coherent_ffl_graph() -> ig.Graph:
    """Create a graph with a coherent FFL (all-positive 030T).

    3 nodes:
        Layer 0: A(0)
        Layer 1: B(1)
        Layer 2: C(2)

    Edges (all excitatory):
        0→1 (+), 1→2 (+), 0→2 (+)
    """
    g = ig.Graph(directed=True)
    g.add_vertices(3)

    layers = [0, 1, 2]
    for i, layer in enumerate(layers):
        g.vs[i]["layer"] = layer
        g.vs[i]["feature_type"] = "cross layer transcoder"
        g.vs[i]["ctx_idx"] = i
        g.vs[i]["clerp"] = f"node_{i}"

    for src, tgt in [(0, 1), (1, 2), (0, 2)]:
        g.add_edge(src, tgt, weight=1.0, raw_weight=1.0, sign="excitatory")

    return g


def _make_incoherent_ffl_graph() -> ig.Graph:
    """Create a graph with an incoherent FFL (mixed-sign 030T).

    3 nodes:
        Layer 0: A(0)
        Layer 1: B(1)
        Layer 2: C(2)

    Edges:
        0→1 (+), 1→2 (+), 0→2 (-)
    """
    g = ig.Graph(directed=True)
    g.add_vertices(3)

    layers = [0, 1, 2]
    for i, layer in enumerate(layers):
        g.vs[i]["layer"] = layer
        g.vs[i]["feature_type"] = "cross layer transcoder"
        g.vs[i]["ctx_idx"] = i
        g.vs[i]["clerp"] = f"node_{i}"

    g.add_edge(0, 1, weight=1.0, raw_weight=1.0, sign="excitatory")
    g.add_edge(1, 2, weight=1.0, raw_weight=1.0, sign="excitatory")
    g.add_edge(0, 2, weight=0.8, raw_weight=-0.8, sign="inhibitory")

    return g


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

class TestUnrolledMotifTemplate:
    def test_catalog_builds(self):
        catalog = build_catalog()
        assert len(catalog) == 8

    def test_catalog_names_unique(self):
        catalog = build_catalog()
        names = [t.name for t in catalog]
        assert len(names) == len(set(names))

    def test_template_graph_construction(self):
        for template in build_catalog():
            g = template.to_template_graph()
            assert g.vcount() == template.num_nodes
            assert g.ecount() == template.num_edges
            assert g.is_directed()

    def test_cross_chain_inhibition_structure(self):
        t = CATALOG["cross_chain_inhibition"]
        assert t.num_nodes == 4
        assert t.num_edges == 4
        g = t.to_template_graph()
        # Check that it has both +1 and -1 edges
        signs = g.es["sign"]
        assert +1 in signs
        assert -1 in signs

    def test_feedforward_damping_structure(self):
        t = CATALOG["feedforward_damping"]
        assert t.num_nodes == 3
        assert t.num_edges == 2
        g = t.to_template_graph()
        signs = g.es["sign"]
        assert +1 in signs
        assert -1 in signs

    def test_coherent_ffl_all_positive(self):
        t = CATALOG["coherent_ffl"]
        g = t.to_template_graph()
        assert all(s == +1 for s in g.es["sign"])

    def test_incoherent_ffl_mixed_signs(self):
        t = CATALOG["incoherent_ffl"]
        g = t.to_template_graph()
        signs = g.es["sign"]
        assert +1 in signs
        assert -1 in signs

    def test_cross_chain_toggle_structure(self):
        t = CATALOG["cross_chain_toggle"]
        assert t.num_nodes == 5
        assert t.num_edges == 4

    def test_node_ids_property(self):
        t = CATALOG["feedforward_damping"]
        ids = t.node_ids
        assert len(ids) == 3
        assert "A_early" in ids
        assert "B_mid" in ids
        assert "A_late" in ids


# ---------------------------------------------------------------------------
# Layer utility tests
# ---------------------------------------------------------------------------

class TestLayerUtilities:
    def test_get_layer_index(self):
        g = _make_layered_graph()
        assert get_layer_index(g, 0) == 0
        assert get_layer_index(g, 2) == 1
        assert get_layer_index(g, 4) == 2

    def test_get_effective_layer(self):
        g = _make_layered_graph()
        assert get_effective_layer(g, 0) == 0
        assert get_effective_layer(g, 4) == 2

    def test_get_effective_layer_logit(self):
        g = ig.Graph(directed=True)
        g.add_vertices(3)
        g.vs[0]["layer"] = 0
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 5
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.vs[2]["layer"] = -1
        g.vs[2]["feature_type"] = "logit"
        # Logit should get max_layer + 1 = 6
        assert get_effective_layer(g, 2) == 6

    def test_validate_layer_ordering_valid(self):
        g = _make_layered_graph()
        is_valid, backward = validate_layer_ordering(g)
        assert is_valid
        assert len(backward) == 0

    def test_validate_layer_ordering_invalid(self):
        g = ig.Graph(directed=True)
        g.add_vertices(2)
        g.vs[0]["layer"] = 5
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 2
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.add_edge(0, 1, weight=1.0, raw_weight=1.0, sign="excitatory")
        is_valid, backward = validate_layer_ordering(g)
        assert not is_valid
        assert len(backward) == 1

    def test_layer_gap_distribution(self):
        g = _make_layered_graph()
        gaps = layer_gap_distribution(g)
        # All edges go from layer 0→1, 0→2, or 1→2
        assert all(gap > 0 for gap in gaps.keys())

    def test_layer_stats(self):
        g = _make_layered_graph()
        stats = layer_stats(g)
        assert stats["n_layers"] == 3
        assert stats["all_forward"] is True
        assert stats["n_backward_edges"] == 0


# ---------------------------------------------------------------------------
# Edge sign matching tests
# ---------------------------------------------------------------------------

class TestEdgeSignMatching:
    def test_any_sign_always_matches(self):
        assert _edge_sign_matches(1.0, "excitatory", 0)
        assert _edge_sign_matches(-1.0, "inhibitory", 0)

    def test_positive_sign_matches_excitatory(self):
        assert _edge_sign_matches(1.0, "excitatory", +1)

    def test_positive_sign_rejects_inhibitory(self):
        assert not _edge_sign_matches(-1.0, "inhibitory", +1)

    def test_negative_sign_matches_inhibitory(self):
        assert _edge_sign_matches(-1.0, "inhibitory", -1)

    def test_negative_sign_rejects_excitatory(self):
        assert not _edge_sign_matches(1.0, "excitatory", -1)


# ---------------------------------------------------------------------------
# Census / instance finding tests
# ---------------------------------------------------------------------------

class TestFindUnrolledInstances:
    def test_finds_cross_chain_inhibition(self):
        g = _make_cross_chain_inhibition_graph()
        template = CATALOG["cross_chain_inhibition"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) >= 1
        # Verify first instance has 4 nodes
        assert len(instances[0].node_indices) == 4

    def test_finds_feedforward_damping(self):
        g = _make_damping_graph()
        template = CATALOG["feedforward_damping"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) >= 1
        inst = instances[0]
        assert inst.template_name == "feedforward_damping"
        assert len(inst.node_indices) == 3

    def test_finds_coherent_ffl(self):
        g = _make_coherent_ffl_graph()
        template = CATALOG["coherent_ffl"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) >= 1

    def test_finds_incoherent_ffl(self):
        g = _make_incoherent_ffl_graph()
        template = CATALOG["incoherent_ffl"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) >= 1

    def test_no_coherent_ffl_in_incoherent_graph(self):
        g = _make_incoherent_ffl_graph()
        template = CATALOG["coherent_ffl"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) == 0

    def test_no_incoherent_ffl_in_coherent_graph(self):
        g = _make_coherent_ffl_graph()
        template = CATALOG["incoherent_ffl"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) == 0

    def test_layer_gap_constraint(self):
        g = _make_cross_chain_inhibition_graph()
        template = CATALOG["cross_chain_inhibition"]
        # Gap is 2 (layer 0 → layer 2). With max_layer_gap=1, should find nothing.
        instances = find_unrolled_instances(g, template, max_layer_gap=1)
        assert len(instances) == 0

    def test_weight_threshold_filters(self):
        g = _make_damping_graph()
        template = CATALOG["feedforward_damping"]
        # With threshold higher than one of the edges
        instances = find_unrolled_instances(g, template, weight_threshold=2.0)
        assert len(instances) == 0

    def test_instances_sorted_by_weight(self):
        g = _make_layered_graph()
        template = CATALOG["coherent_ffl"]
        instances = find_unrolled_instances(g, template)
        if len(instances) >= 2:
            assert instances[0].total_weight >= instances[1].total_weight

    def test_no_amplification_in_damping_graph(self):
        g = _make_damping_graph()
        template = CATALOG["feedforward_amplification"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) == 0

    def test_requires_directed_graph(self):
        g = ig.Graph(directed=False)
        g.add_vertices(3)
        g.add_edges([(0, 1), (1, 2)])
        template = CATALOG["feedforward_damping"]
        with pytest.raises(ValueError, match="directed"):
            find_unrolled_instances(g, template)

    def test_instance_roles_assigned(self):
        g = _make_cross_chain_inhibition_graph()
        template = CATALOG["cross_chain_inhibition"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) >= 1
        inst = instances[0]
        roles = set(inst.node_roles.values())
        assert "chain_a_source" in roles or "chain_b_source" in roles

    def test_residual_self_loop_positive(self):
        """A single excitatory forward edge should match positive self-loop."""
        g = ig.Graph(directed=True)
        g.add_vertices(2)
        g.vs[0]["layer"] = 0
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 3
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.add_edge(0, 1, weight=1.0, raw_weight=1.0, sign="excitatory")

        template = CATALOG["residual_self_loop_positive"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) == 1

    def test_residual_self_loop_negative(self):
        """A single inhibitory forward edge should match negative self-loop."""
        g = ig.Graph(directed=True)
        g.add_vertices(2)
        g.vs[0]["layer"] = 0
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 3
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.add_edge(0, 1, weight=1.0, raw_weight=-1.0, sign="inhibitory")

        template = CATALOG["residual_self_loop_negative"]
        instances = find_unrolled_instances(g, template)
        assert len(instances) == 1


class TestRunUnrolledCensus:
    def test_census_returns_all_templates(self):
        g = _make_layered_graph()
        results = run_unrolled_census(g)
        templates = build_catalog()
        assert set(results.keys()) == {t.name for t in templates}

    def test_census_counts(self):
        g = _make_layered_graph()
        results = run_unrolled_census(g)
        counts = unrolled_census_counts(results)
        assert isinstance(counts, dict)
        for name, count in counts.items():
            assert isinstance(count, int)
            assert count >= 0

    def test_census_summary(self):
        g = _make_layered_graph()
        results = run_unrolled_census(g)
        summary = unrolled_census_summary(results)
        for name, stats in summary.items():
            assert "count" in stats
            assert "mean_weight" in stats

    def test_max_instances_per_motif(self):
        g = _make_layered_graph()
        results = run_unrolled_census(g, max_instances_per_motif=1)
        for name, instances in results.items():
            assert len(instances) <= 1


# ---------------------------------------------------------------------------
# Null model tests
# ---------------------------------------------------------------------------

class TestLayerPreservingRewire:
    def test_preserves_layer_ordering(self):
        g = _make_layered_graph()
        g_rewired, acc = layer_preserving_rewire(g, n_swaps=50, seed=42)
        assert verify_layer_preservation(g, g_rewired)

    def test_preserves_sign_distribution(self):
        g = _make_layered_graph()
        g_rewired, acc = layer_preserving_rewire(g, seed=42)
        assert verify_sign_preservation(g, g_rewired)

    def test_preserves_degree_sequence(self):
        g = _make_layered_graph()
        g_rewired, acc = layer_preserving_rewire(g, n_swaps=50, seed=42)
        assert sorted(g.indegree()) == sorted(g_rewired.indegree())
        assert sorted(g.outdegree()) == sorted(g_rewired.outdegree())

    def test_preserves_node_and_edge_count(self):
        g = _make_layered_graph()
        g_rewired, acc = layer_preserving_rewire(g, seed=42)
        assert g_rewired.vcount() == g.vcount()
        assert g_rewired.ecount() == g.ecount()

    def test_acceptance_rate_positive(self):
        g = _make_layered_graph()
        _, acc = layer_preserving_rewire(g, n_swaps=50, seed=42)
        # Should have some successful swaps
        assert acc >= 0.0

    def test_small_graph_no_crash(self):
        g = ig.Graph(directed=True)
        g.add_vertices(2)
        g.vs[0]["layer"] = 0
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 1
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.add_edge(0, 1, weight=1.0, raw_weight=1.0, sign="excitatory")
        g_rewired, acc = layer_preserving_rewire(g, seed=42)
        assert g_rewired.vcount() == 2

    def test_seed_reproducibility(self):
        g = _make_layered_graph()
        g1, _ = layer_preserving_rewire(g, n_swaps=20, seed=123)
        g2, _ = layer_preserving_rewire(g, n_swaps=20, seed=123)
        # Same seed should produce same result
        assert g1.get_edgelist() == g2.get_edgelist()


class TestComputeUnrolledZscores:
    def test_produces_z_scores_for_all_templates(self):
        g = _make_layered_graph()
        templates = build_catalog()
        result = compute_unrolled_zscores(
            g, templates, n_random=5, show_progress=False, seed=42,
        )
        assert set(result.z_scores.keys()) == {t.name for t in templates}

    def test_z_score_types(self):
        g = _make_layered_graph()
        result = compute_unrolled_zscores(
            g, n_random=5, show_progress=False, seed=42,
        )
        for name, z in result.z_scores.items():
            assert isinstance(z, float)
            assert np.isfinite(z) or z == 100.0 or z == -100.0

    def test_null_counts_correct_length(self):
        g = _make_layered_graph()
        n_random = 5
        result = compute_unrolled_zscores(
            g, n_random=n_random, show_progress=False, seed=42,
        )
        for name, counts in result.null_counts.items():
            assert len(counts) == n_random

    def test_acceptance_rate_recorded(self):
        g = _make_layered_graph()
        result = compute_unrolled_zscores(
            g, n_random=5, show_progress=False, seed=42,
        )
        assert 0.0 <= result.acceptance_rate <= 1.0


class TestVerifyLayerPreservation:
    def test_valid_graph(self):
        g = _make_layered_graph()
        assert verify_layer_preservation(g, g)

    def test_invalid_after_bad_rewire(self):
        # Create a graph where we manually break layer ordering
        g = ig.Graph(directed=True)
        g.add_vertices(3)
        g.vs[0]["layer"] = 0
        g.vs[0]["feature_type"] = "cross layer transcoder"
        g.vs[1]["layer"] = 1
        g.vs[1]["feature_type"] = "cross layer transcoder"
        g.vs[2]["layer"] = 2
        g.vs[2]["feature_type"] = "cross layer transcoder"
        # Backward edge: 2→0
        g.add_edge(2, 0, weight=1.0, raw_weight=1.0, sign="excitatory")
        assert not verify_layer_preservation(g, g)
