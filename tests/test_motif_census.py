"""Tests for motif_census.py."""

import math

import igraph as ig
import numpy as np
import pytest

from src.motif_census import (
    compute_motif_census,
    motif_frequencies,
    enriched_motifs,
    TRIAD_LABELS,
    CONNECTED_TRIAD_INDICES,
    MOTIF_FAN_OUT,
    MOTIF_FAN_IN,
    MOTIF_CHAIN,
    MOTIF_FFL,
)


def _make_chain_graph() -> ig.Graph:
    """A->B->C: a pure chain (021C)."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (1, 2)])
    return g


def _make_fan_out_graph() -> ig.Graph:
    """A->B, A->C: fan-out (021D)."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2)])
    return g


def _make_fan_in_graph() -> ig.Graph:
    """A->C, B->C: fan-in (021U)."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 2), (1, 2)])
    return g


def _make_ffl_graph() -> ig.Graph:
    """A->B, A->C, B->C: feedforward loop (030T)."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.add_edges([(0, 1), (0, 2), (1, 2)])
    return g


def _make_larger_graph() -> ig.Graph:
    """A graph with multiple triads to test enumeration."""
    g = ig.Graph(directed=True)
    g.add_vertices(6)
    # Fan-out from 0: 0->1, 0->2
    # Chain: 1->3
    # Fan-in to 4: 2->4, 3->4
    # FFL: 0->1, 0->2, 1->2 (if we add 1->2)
    g.add_edges([
        (0, 1), (0, 2), (1, 3), (2, 4), (3, 4), (1, 2), (0, 5), (5, 4),
    ])
    return g


class TestComputeMotifCensus:
    def test_chain_graph(self):
        g = _make_chain_graph()
        result = compute_motif_census(g, size=3)
        assert result.size == 3
        assert result.n_classes == 16
        # Only one triad of 3 nodes: 021C (chain) at index 5
        assert result.raw_counts[MOTIF_CHAIN] == 1
        # Total connected triads should be 1
        total_connected = sum(
            result.raw_counts[i] for i in CONNECTED_TRIAD_INDICES
        )
        assert total_connected == 1

    def test_fan_out_graph(self):
        g = _make_fan_out_graph()
        result = compute_motif_census(g, size=3)
        assert result.raw_counts[MOTIF_FAN_OUT] == 1

    def test_fan_in_graph(self):
        g = _make_fan_in_graph()
        result = compute_motif_census(g, size=3)
        assert result.raw_counts[MOTIF_FAN_IN] == 1

    def test_ffl_graph(self):
        g = _make_ffl_graph()
        result = compute_motif_census(g, size=3)
        assert result.raw_counts[MOTIF_FFL] == 1

    def test_larger_graph_nonzero(self):
        g = _make_larger_graph()
        result = compute_motif_census(g, size=3)
        # Should have multiple triads
        total = sum(result.raw_counts[i] for i in CONNECTED_TRIAD_INDICES)
        assert total > 1

    def test_no_nan_in_counts(self):
        g = _make_chain_graph()
        result = compute_motif_census(g, size=3)
        for c in result.raw_counts:
            assert not (isinstance(c, float) and math.isnan(c))

    def test_labels_match(self):
        g = _make_chain_graph()
        result = compute_motif_census(g, size=3)
        assert result.labels == TRIAD_LABELS
        assert len(result.labels) == 16

    def test_undirected_raises(self):
        g = ig.Graph(directed=False)
        g.add_vertices(3)
        g.add_edges([(0, 1), (1, 2)])
        with pytest.raises(ValueError, match="directed"):
            compute_motif_census(g)

    def test_invalid_size_raises(self):
        g = _make_chain_graph()
        with pytest.raises(ValueError, match="3 or 4"):
            compute_motif_census(g, size=5)

    def test_size_4(self):
        g = _make_larger_graph()
        result = compute_motif_census(g, size=4)
        assert result.size == 4
        # Size 4 has 218 isomorphism classes
        assert result.n_classes == 218

    def test_as_array(self):
        g = _make_ffl_graph()
        result = compute_motif_census(g, size=3)
        arr = result.as_array()
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (16,)

    def test_as_dict(self):
        g = _make_ffl_graph()
        result = compute_motif_census(g, size=3)
        d = result.as_dict()
        assert "030T" in d
        assert d["030T"] == 1

    def test_connected_counts(self):
        g = _make_ffl_graph()
        result = compute_motif_census(g, size=3)
        cc = result.connected_counts()
        assert "003" not in cc
        assert "030T" in cc


class TestMotifFrequencies:
    def test_frequencies_sum_to_one(self):
        g = _make_larger_graph()
        result = compute_motif_census(g, size=3)
        freqs = motif_frequencies(result)
        assert abs(freqs.sum() - 1.0) < 1e-10

    def test_empty_graph(self):
        g = ig.Graph(directed=True)
        g.add_vertices(3)
        result = compute_motif_census(g, size=3)
        freqs = motif_frequencies(result)
        # All counts should be 0 except the empty triad
        # But frequencies should handle this
        assert freqs.sum() >= 0


class TestEnrichedMotifs:
    def test_finds_enriched(self):
        # Place enriched values at known indices:
        # index 7 = 030T (FFL), index 6 = 021D (fan-out), index 2 = 021U (fan-in)
        z = np.zeros(16)
        z[MOTIF_FFL] = 4.1       # 030T enriched
        z[MOTIF_FAN_OUT] = 3.5   # 021D enriched
        z[MOTIF_FAN_IN] = -2.5   # 021U anti-enriched
        z[1] = 0.5               # not significant
        results = enriched_motifs(z, threshold=2.0, labels=TRIAD_LABELS)
        labels_found = {r["label"] for r in results}
        assert "030T" in labels_found   # z=4.1
        assert "021D" in labels_found   # z=3.5
        assert "021U" in labels_found   # z=-2.5

    def test_direction_correct(self):
        z = np.array([0.0] * 3 + [3.0] + [0.0] * 12)
        results = enriched_motifs(z, threshold=2.0, labels=TRIAD_LABELS)
        assert results[0]["direction"] == "enriched"

        z2 = np.array([0.0] * 3 + [-3.0] + [0.0] * 12)
        results2 = enriched_motifs(z2, threshold=2.0, labels=TRIAD_LABELS)
        assert results2[0]["direction"] == "anti-enriched"

    def test_sorted_by_magnitude(self):
        z = np.array([0.0, 0.0, 0.0, 5.0, -3.0] + [0.0] * 11)
        results = enriched_motifs(z, threshold=2.0, labels=TRIAD_LABELS)
        assert abs(results[0]["z_score"]) >= abs(results[1]["z_score"])
