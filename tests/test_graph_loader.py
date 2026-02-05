"""Tests for graph_loader.py."""

import json
import tempfile
from pathlib import Path

import igraph as ig
import pytest

from src.graph_loader import (
    load_attribution_graph,
    parse_attribution_graph,
    graph_summary,
    load_graphs_from_directory,
    FEATURE_TYPE_ERROR,
    FEATURE_TYPE_TRANSCODER,
    FEATURE_TYPE_EMBEDDING,
    FEATURE_TYPE_LOGIT,
)


def _make_sample_graph_data() -> dict:
    """Create a minimal valid attribution graph JSON structure."""
    return {
        "metadata": {
            "slug": "test-graph",
            "scan": "gemma-2-2b",
            "prompt_tokens": ["The", " capital", " of", " France", " is"],
            "prompt": "The capital of France is",
            "node_threshold": 0.8,
            "schema_version": 1,
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
        },
        "nodes": [
            {
                "node_id": "E_100_0",
                "feature": 0,
                "layer": "E",
                "ctx_idx": 0,
                "feature_type": "embedding",
                "jsNodeId": "E_100-0",
                "clerp": "The",
            },
            {
                "node_id": "5_253_1",
                "feature": 32131,
                "layer": "5",
                "ctx_idx": 1,
                "feature_type": "cross layer transcoder",
                "jsNodeId": "5_253-0",
                "clerp": "references to France",
                "activation": 9.449,
                "influence": 0.482,
            },
            {
                "node_id": "10_500_3",
                "feature": 55321,
                "layer": "10",
                "ctx_idx": 3,
                "feature_type": "cross layer transcoder",
                "jsNodeId": "10_500-0",
                "clerp": "capital cities",
                "activation": 5.2,
                "influence": 0.31,
            },
            {
                "node_id": "0_3_2",
                "feature": -1,
                "layer": "3",
                "ctx_idx": 2,
                "feature_type": "mlp reconstruction error",
                "jsNodeId": "3_2-0",
                "clerp": "",
                "influence": 0.021,
            },
            {
                "node_id": "27_1234_4",
                "feature": 1234,
                "layer": "27",
                "ctx_idx": 4,
                "feature_type": "logit",
                "jsNodeId": "L_1234-4",
                "clerp": 'Output "Paris" (p=0.892)',
                "token_prob": 0.892,
                "is_target_logit": True,
            },
        ],
        "links": [
            {"source": "E_100_0", "target": "5_253_1", "weight": 12.964},
            {"source": "5_253_1", "target": "10_500_3", "weight": 6.5},
            {"source": "5_253_1", "target": "27_1234_4", "weight": 8.341},
            {"source": "10_500_3", "target": "27_1234_4", "weight": 4.2},
            {"source": "E_100_0", "target": "10_500_3", "weight": -3.1},
        ],
    }


@pytest.fixture
def sample_data():
    return _make_sample_graph_data()


@pytest.fixture
def sample_json_path(sample_data):
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(sample_data, f)
        return Path(f.name)


class TestParseAttributionGraph:
    def test_basic_loading(self, sample_data):
        g = parse_attribution_graph(sample_data)
        assert g.is_directed()
        # 5 nodes minus 1 error node = 4 kept
        assert g.vcount() == 4
        assert g.ecount() == 5

    def test_error_nodes_excluded(self, sample_data):
        g = parse_attribution_graph(sample_data)
        feature_types = g.vs["feature_type"]
        assert FEATURE_TYPE_ERROR not in feature_types

    def test_node_attributes(self, sample_data):
        g = parse_attribution_graph(sample_data)
        # Check the France feature node
        france_node = g.vs.find(name="5_253_1")
        assert france_node["layer"] == 5
        assert france_node["feature_type"] == FEATURE_TYPE_TRANSCODER
        assert france_node["clerp"] == "references to France"
        assert france_node["activation"] == 9.449

    def test_embedding_layer(self, sample_data):
        g = parse_attribution_graph(sample_data)
        emb_node = g.vs.find(name="E_100_0")
        assert emb_node["layer"] == -1  # Embedding gets -1

    def test_edge_attributes(self, sample_data):
        g = parse_attribution_graph(sample_data)
        # Find the edge from France feature to logit
        src_idx = g.vs.find(name="5_253_1").index
        tgt_idx = g.vs.find(name="27_1234_4").index
        eid = g.get_eid(src_idx, tgt_idx)
        assert g.es[eid]["weight"] == 8.341
        assert g.es[eid]["sign"] == "excitatory"

    def test_inhibitory_edge(self, sample_data):
        g = parse_attribution_graph(sample_data)
        # The edge with weight -3.1 should be inhibitory
        src_idx = g.vs.find(name="E_100_0").index
        tgt_idx = g.vs.find(name="10_500_3").index
        eid = g.get_eid(src_idx, tgt_idx)
        assert g.es[eid]["sign"] == "inhibitory"
        assert g.es[eid]["weight"] == 3.1  # abs value
        assert g.es[eid]["raw_weight"] == -3.1

    def test_weight_threshold(self, sample_data):
        g = parse_attribution_graph(sample_data, weight_threshold=5.0)
        # Only edges with |weight| >= 5.0 should remain
        # 12.964, 6.5, 8.341 pass; 4.2 and 3.1 do not
        assert g.ecount() == 3

    def test_metadata_stored(self, sample_data):
        g = parse_attribution_graph(sample_data)
        assert g["prompt"] == "The capital of France is"
        assert g["model"] == "gemma-2-2b"
        assert g["slug"] == "test-graph"

    def test_no_metadata(self, sample_data):
        g = parse_attribution_graph(sample_data, include_metadata=False)
        # Should not raise, metadata attributes simply not set
        assert g.vcount() == 4

    def test_edges_skip_excluded_nodes(self, sample_data):
        """Edges referencing excluded (error) nodes should be dropped."""
        # Add an edge from the error node
        sample_data["links"].append(
            {"source": "0_3_2", "target": "5_253_1", "weight": 1.0}
        )
        g = parse_attribution_graph(sample_data)
        # Error node is excluded, so this edge should not appear
        assert g.ecount() == 5  # same as before

    def test_include_error_nodes(self, sample_data):
        g = parse_attribution_graph(sample_data, exclude_node_types=frozenset())
        assert g.vcount() == 5  # All nodes including error


class TestLoadAttributionGraph:
    def test_load_from_file(self, sample_json_path):
        g = load_attribution_graph(sample_json_path)
        assert g.is_directed()
        assert g.vcount() == 4

    def test_load_with_threshold(self, sample_json_path):
        g = load_attribution_graph(sample_json_path, weight_threshold=7.0)
        assert g.ecount() == 2  # only 12.964 and 8.341


class TestGraphSummary:
    def test_summary_contents(self, sample_data):
        g = parse_attribution_graph(sample_data)
        summary = graph_summary(g)
        assert summary["n_nodes"] == 4
        assert summary["n_edges"] == 5
        assert summary["prompt"] == "The capital of France is"
        assert "density" in summary
        assert FEATURE_TYPE_TRANSCODER in summary["node_type_counts"]


class TestLoadGraphsFromDirectory:
    def test_load_directory(self, sample_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                path = Path(tmpdir) / f"graph_{i}.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(sample_data, f)

            graphs = load_graphs_from_directory(tmpdir)
            assert len(graphs) == 3
            assert all(g.is_directed() for g in graphs)
