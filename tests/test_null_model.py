"""Tests for null_model.py."""

import igraph as ig
import numpy as np
import pytest

from src.null_model import (
    generate_configuration_null,
    generate_erdos_renyi_null,
    verify_degree_preservation,
    _compute_z_scores,
    _compute_significance_profile,
)


def _make_test_graph() -> ig.Graph:
    """Create a directed graph with enough structure for meaningful motif analysis."""
    g = ig.Graph(directed=True)
    g.add_vertices(10)
    edges = [
        (0, 1), (0, 2), (0, 3),  # fan-out from 0
        (1, 4), (2, 4), (3, 4),  # fan-in to 4
        (4, 5), (5, 6),          # chain
        (0, 4), (1, 2),          # extra edges
        (6, 7), (7, 8), (8, 9),  # another chain
        (6, 9),                   # shortcut
        (3, 5),                   # cross-link
    ]
    g.add_edges(edges)
    return g


class TestComputeZScores:
    def test_basic_z_scores(self):
        real = np.array([10.0, 5.0, 0.0])
        null = np.array([
            [8.0, 6.0, 1.0],
            [9.0, 5.0, 2.0],
            [7.0, 7.0, 0.0],
            [8.0, 4.0, 1.0],
        ])
        z, mean_n, std_n = _compute_z_scores(real, null)
        assert z.shape == (3,)
        assert mean_n.shape == (3,)
        assert std_n.shape == (3,)

    def test_z_score_values(self):
        real = np.array([10.0])
        null = np.array([[8.0], [8.0], [8.0], [8.0]])  # mean=8, std=0
        z, _, _ = _compute_z_scores(real, null)
        # std is 0, real != mean, so z should be inf
        assert np.isinf(z[0])
        assert z[0] > 0

    def test_zero_std_equal_mean(self):
        real = np.array([8.0])
        null = np.array([[8.0], [8.0], [8.0]])
        z, _, _ = _compute_z_scores(real, null)
        assert z[0] == 0.0


class TestComputeSignificanceProfile:
    def test_unit_length(self):
        z = np.array([3.0, 4.0, 0.0, -2.0, 1.0])
        sp = _compute_significance_profile(z)
        norm = np.sqrt(np.sum(sp ** 2))
        assert abs(norm - 1.0) < 1e-10

    def test_zero_vector(self):
        z = np.zeros(16)
        sp = _compute_significance_profile(z)
        assert np.allclose(sp, 0.0)

    def test_preserves_direction(self):
        z = np.array([3.0, -4.0])
        sp = _compute_significance_profile(z)
        assert sp[0] > 0
        assert sp[1] < 0

    def test_handles_inf(self):
        z = np.array([np.inf, -np.inf, 3.0])
        sp = _compute_significance_profile(z)
        assert np.all(np.isfinite(sp))


class TestGenerateConfigurationNull:
    def test_runs_and_produces_result(self):
        g = _make_test_graph()
        result = generate_configuration_null(
            g, n_random=10, motif_size=3, show_progress=False
        )
        assert result.null_type == "configuration"
        assert result.n_random == 10
        assert result.z_scores.shape == (16,)
        assert result.significance_profile.shape == (16,)
        assert result.null_counts.shape == (10, 16)

    def test_degree_preservation(self):
        g = _make_test_graph()
        g_copy = g.copy()
        g_copy.rewire(n=g.ecount() * 10)
        assert verify_degree_preservation(g, g_copy)


class TestGenerateErdosRenyiNull:
    def test_runs_and_produces_result(self):
        g = _make_test_graph()
        result = generate_erdos_renyi_null(
            g, n_random=10, motif_size=3, show_progress=False
        )
        assert result.null_type == "erdos_renyi"
        assert result.n_random == 10
        assert result.z_scores.shape == (16,)


class TestVerifyDegreePreservation:
    def test_same_graph(self):
        g = _make_test_graph()
        assert verify_degree_preservation(g, g.copy())

    def test_rewired_preserves(self):
        g = _make_test_graph()
        g2 = g.copy()
        g2.rewire(n=g.ecount() * 10)
        assert verify_degree_preservation(g, g2)

    def test_different_graph_fails(self):
        g1 = ig.Graph(directed=True)
        g1.add_vertices(5)
        g1.add_edges([(0, 1), (0, 2), (0, 3)])  # hub

        g2 = ig.Graph(directed=True)
        g2.add_vertices(5)
        g2.add_edges([(0, 1), (1, 2), (2, 3)])  # chain

        assert not verify_degree_preservation(g1, g2)
