"""Tests for neuron_graph.py — JSON compatibility with existing pipeline."""

import json
import tempfile
from pathlib import Path

import igraph as ig
import numpy as np
import pytest

from src.graph_loader import (
    parse_attribution_graph,
    load_attribution_graph,
    DEFAULT_EXCLUDE_TYPES,
)
from src.neuron_graph import (
    NeuronGraphConfig,
    build_neuron_graph_json,
    characterize_graph,
    _gini,
)
from src.unrolled_census import fast_unrolled_counts
from src.unrolled_motifs import get_effective_layer


def _make_minimal_neuron_json(
    n_layers: int = 5,
    neurons_per_layer: int = 10,
    n_edges: int = 50,
    seed: int = 42,
) -> dict:
    """Build a small synthetic neuron graph JSON for testing.

    Creates a feedforward neuron graph with forward-only edges
    and mixed excitatory/inhibitory signs.

    Args:
        n_layers: Number of layers.
        neurons_per_layer: Neurons per layer.
        n_edges: Approximate number of edges to generate.
        seed: Random seed for reproducibility.

    Returns:
        Dict in circuit-tracer compatible JSON format.
    """
    rng = np.random.default_rng(seed)

    nodes = []
    for layer in range(n_layers):
        for neuron in range(neurons_per_layer):
            node_id = f"L{layer}_N{neuron}"
            act = float(rng.standard_normal())
            nodes.append({
                "node_id": node_id,
                "feature": neuron,
                "layer": str(layer),
                "ctx_idx": 0,
                "feature_type": "mlp_neuron",
                "clerp": f"neuron {neuron} @ L{layer}",
                "activation": act,
                "influence": 0.0,
                "is_target_logit": False,
                "token_prob": 0.0,
            })

    # Generate forward-only edges with random weights
    links = []
    for _ in range(n_edges):
        src_layer = int(rng.integers(0, n_layers - 1))
        tgt_layer = int(rng.integers(src_layer + 1, n_layers))
        src_neuron = int(rng.integers(0, neurons_per_layer))
        tgt_neuron = int(rng.integers(0, neurons_per_layer))
        weight = float(rng.standard_normal() * 2.0)
        links.append({
            "source": f"L{src_layer}_N{src_neuron}",
            "target": f"L{tgt_layer}_N{tgt_neuron}",
            "weight": weight,
        })

    return {
        "metadata": {
            "slug": "test-neuron-graph",
            "scan": "google/gemma-2-2b",
            "prompt": "5 + 3 =",
            "prompt_tokens": ["<bos>", "5", " +", " 3", " ="],
            "schema_version": 1,
            "info": {
                "graph_type": "neuron_level",
                "top_k": neurons_per_layer,
                "max_layer_gap": 5,
            },
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
        },
        "nodes": nodes,
        "links": links,
    }


@pytest.fixture
def neuron_json():
    return _make_minimal_neuron_json()


@pytest.fixture
def neuron_json_path(neuron_json):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(neuron_json, f)
        return Path(f.name)


@pytest.fixture
def neuron_graph(neuron_json):
    return parse_attribution_graph(neuron_json)


class TestJsonCompatibility:
    """Verify neuron JSON loads through the existing graph_loader pipeline."""

    def test_json_loads_via_graph_loader(self, neuron_json):
        """parse_attribution_graph() succeeds on neuron JSON."""
        g = parse_attribution_graph(neuron_json)
        assert g.is_directed()
        assert g.vcount() == 50  # 5 layers × 10 neurons
        assert g.ecount() > 0

    def test_load_from_file(self, neuron_json_path):
        """load_attribution_graph() works with saved neuron JSON."""
        g = load_attribution_graph(neuron_json_path)
        assert g.is_directed()
        assert g.vcount() == 50

    def test_feature_type_not_excluded(self):
        """mlp_neuron is not in DEFAULT_EXCLUDE_TYPES."""
        assert "mlp_neuron" not in DEFAULT_EXCLUDE_TYPES

    def test_layer_parsing(self, neuron_graph):
        """Neuron node layers parse to correct integers."""
        layers = set(neuron_graph.vs["layer"])
        assert layers == {0, 1, 2, 3, 4}

    def test_sign_derived_from_weight(self, neuron_json):
        """Positive weight → excitatory, negative → inhibitory."""
        g = parse_attribution_graph(neuron_json)
        for e in g.es:
            if e["raw_weight"] >= 0:
                assert e["sign"] == "excitatory"
            else:
                assert e["sign"] == "inhibitory"
            assert e["weight"] == abs(e["raw_weight"])

    def test_metadata_stored(self, neuron_json):
        """Graph-level metadata is accessible."""
        g = parse_attribution_graph(neuron_json)
        assert g["prompt"] == "5 + 3 ="
        assert g["slug"] == "test-neuron-graph"

    def test_node_attributes(self, neuron_graph):
        """Nodes have expected attributes."""
        v = neuron_graph.vs[0]
        assert v["feature_type"] == "mlp_neuron"
        assert isinstance(v["activation"], float)
        assert isinstance(v["layer"], int)


class TestMotifPipelineCompatibility:
    """Verify neuron graphs work with the unrolled motif analysis pipeline."""

    def test_fast_unrolled_counts_works(self, neuron_graph):
        """fast_unrolled_counts() runs on neuron graph without error."""
        counts = fast_unrolled_counts(neuron_graph)
        assert isinstance(counts, dict)
        assert "coherent_ffl" in counts
        assert "incoherent_ffl" in counts
        assert all(isinstance(v, int) for v in counts.values())
        assert all(v >= 0 for v in counts.values())

    def test_effective_layer(self, neuron_graph):
        """get_effective_layer() returns correct values for neuron nodes."""
        for v in neuron_graph.vs:
            eff = get_effective_layer(neuron_graph, v.index)
            assert eff == v["layer"]

    def test_null_model_runs(self, neuron_graph):
        """Layer-pair config null model runs on neuron graph."""
        # Import the generator function
        from scripts.unrolled_null_pilot import gen_layer_pair_config

        rng = np.random.default_rng(42)
        g_null = gen_layer_pair_config(neuron_graph, rng)

        assert g_null.is_directed()
        assert g_null.vcount() == neuron_graph.vcount()
        # Should have edges (some may be lost in rewiring but most preserved)
        assert g_null.ecount() > 0

        # Null graph should work with fast_unrolled_counts
        counts = fast_unrolled_counts(g_null)
        assert isinstance(counts, dict)
        assert "coherent_ffl" in counts


class TestBuildNeuronGraphJson:
    """Test the JSON builder function."""

    def test_builds_valid_json(self):
        """build_neuron_graph_json produces loadable output."""
        config = NeuronGraphConfig(model_name="test-model")
        top_neurons = {
            0: [(10, 2.5), (20, 1.8)],
            1: [(5, 3.1), (15, -0.9)],
            2: [(30, 1.2)],
        }
        edges = [
            {
                "source_layer": 0, "source_neuron": 10, "source_act": 2.5,
                "target_layer": 1, "target_neuron": 5, "target_act": 3.1,
                "attribution": 1.75,
            },
            {
                "source_layer": 0, "source_neuron": 20, "source_act": 1.8,
                "target_layer": 2, "target_neuron": 30, "target_act": 1.2,
                "attribution": -0.5,
            },
            {
                "source_layer": 1, "source_neuron": 5, "source_act": 3.1,
                "target_layer": 2, "target_neuron": 30, "target_act": 1.2,
                "attribution": 0.8,
            },
        ]

        result = build_neuron_graph_json(
            prompt="test prompt",
            prompt_tokens=["test", " prompt"],
            slug="test-slug",
            category="arithmetic",
            top_neurons=top_neurons,
            edges=edges,
            config=config,
        )

        # Verify it loads
        g = parse_attribution_graph(result)
        assert g.is_directed()
        assert g.vcount() == 4  # 4 unique neurons in edges
        assert g.ecount() == 3

    def test_only_active_neurons_included(self):
        """Nodes only appear if they participate in edges."""
        config = NeuronGraphConfig()
        top_neurons = {
            0: [(i, float(i)) for i in range(100)],  # 100 neurons
            1: [(i, float(i)) for i in range(100)],
        }
        edges = [
            {
                "source_layer": 0, "source_neuron": 50, "source_act": 50.0,
                "target_layer": 1, "target_neuron": 75, "target_act": 75.0,
                "attribution": 1.0,
            },
        ]

        result = build_neuron_graph_json(
            prompt="t", prompt_tokens=["t"], slug="s", category="c",
            top_neurons=top_neurons, edges=edges, config=config,
        )
        assert len(result["nodes"]) == 2  # Only 2 active neurons


class TestCharacterizeGraph:
    """Test graph characterization utility."""

    def test_characterize_graph(self, neuron_json):
        """Gini, density, excitatory fraction computed correctly."""
        props = characterize_graph(neuron_json)
        assert props["n_nodes"] == 50
        assert props["n_edges"] == 50  # _make_minimal_neuron_json default
        assert 0.0 <= props["density"] <= 1.0
        assert 0.0 <= props["degree_gini"] <= 1.0
        assert 0.0 <= props["excitatory_fraction"] <= 1.0
        assert props["mean_degree"] > 0
        assert len(props["nodes_per_layer"]) == 5

    def test_empty_graph(self):
        """Characterize an empty graph."""
        empty = {"nodes": [], "links": []}
        props = characterize_graph(empty)
        assert props["n_nodes"] == 0
        assert props["n_edges"] == 0
        assert props["density"] == 0.0


class TestGini:
    """Test Gini coefficient computation."""

    def test_perfect_equality(self):
        """All equal values → Gini = 0."""
        assert _gini([1, 1, 1, 1]) == pytest.approx(0.0, abs=1e-10)

    def test_maximum_inequality(self):
        """One nonzero, rest zero → Gini near 1."""
        # For [0, 0, 0, 100]: Gini = (2*(1*0 + 2*0 + 3*0 + 4*100) - 5*100) / (4*100)
        # = (800 - 500) / 400 = 0.75
        g = _gini([0, 0, 0, 100])
        assert g > 0.5

    def test_empty(self):
        assert _gini([]) == 0.0

    def test_all_zeros(self):
        assert _gini([0, 0, 0]) == 0.0
