"""Tests for motif instance finding and graph drawing."""

import igraph as ig
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.motif_census import (
    MotifInstance,
    build_motif_pattern,
    find_motif_instances,
    MOTIF_FFL,
    MOTIF_CHAIN,
    MOTIF_FAN_IN,
    MOTIF_FAN_OUT,
    MOTIF_ROLES,
    TRIAD_LABELS,
)
from src.visualization import (
    _igraph_to_networkx,
    _compute_layered_layout,
    _compute_neuronpedia_layout,
    plot_graph_with_motif,
    plot_top_motif,
)

# Use non-interactive backend for tests
matplotlib.use("Agg")


# --- Helper graph builders ---

def _make_ffl_graph() -> ig.Graph:
    """A->B, A->C, B->C: feedforward loop (030T) with weights."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.vs["layer"] = [1, 5, 10]
    g.vs["ctx_idx"] = [1, 1, 1]
    g.vs["clerp"] = ["input feature", "mediator feature", "output feature"]
    g.vs["feature_type"] = ["embedding", "cross layer transcoder", "logit"]
    g.add_edges([(0, 1), (0, 2), (1, 2)])
    g.es["weight"] = [3.0, 2.0, 5.0]
    g.es["sign"] = ["excitatory", "excitatory", "excitatory"]
    return g


def _make_chain_graph() -> ig.Graph:
    """A->B->C: chain (021C) with weights."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.vs["layer"] = [1, 5, 10]
    g.vs["ctx_idx"] = [0, 1, 2]
    g.vs["clerp"] = ["start", "middle", "end"]
    g.vs["feature_type"] = ["embedding", "cross layer transcoder", "logit"]
    g.add_edges([(0, 1), (1, 2)])
    g.es["weight"] = [4.0, 6.0]
    g.es["sign"] = ["excitatory", "inhibitory"]
    return g


def _make_fan_in_graph() -> ig.Graph:
    """A->C, B->C: fan-in (021U) with weights."""
    g = ig.Graph(directed=True)
    g.add_vertices(3)
    g.vs["layer"] = [1, 1, 10]
    g.vs["ctx_idx"] = [0, 1, 2]
    g.vs["clerp"] = ["source a", "source b", "target"]
    g.vs["feature_type"] = ["embedding", "embedding", "logit"]
    g.add_edges([(0, 2), (1, 2)])
    g.es["weight"] = [2.0, 3.0]
    g.es["sign"] = ["excitatory", "excitatory"]
    return g


def _make_multi_ffl_graph() -> ig.Graph:
    """Graph with multiple FFL instances for testing dedup and ordering.

    Contains two FFL instances:
    - (0,1,2): A->B, A->C, B->C with total weight 10.0
    - (3,4,5): D->E, D->F, E->F with total weight 30.0
    """
    g = ig.Graph(directed=True)
    g.add_vertices(6)
    g.vs["layer"] = [1, 5, 10, 1, 5, 10]
    g.vs["ctx_idx"] = [0, 0, 0, 1, 1, 1]
    g.vs["clerp"] = ["A", "B", "C", "D", "E", "F"]
    g.vs["feature_type"] = ["embedding"] * 6

    # FFL 1: 0->1, 0->2, 1->2
    g.add_edges([(0, 1), (0, 2), (1, 2)])
    # FFL 2: 3->4, 3->5, 4->5
    g.add_edges([(3, 4), (3, 5), (4, 5)])

    g.es["weight"] = [3.0, 3.0, 4.0, 10.0, 10.0, 10.0]
    g.es["sign"] = ["excitatory"] * 6
    return g


# --- Tests for build_motif_pattern ---

class TestBuildMotifPattern:
    def test_ffl_pattern(self):
        p = build_motif_pattern(MOTIF_FFL, size=3)
        assert p.vcount() == 3
        assert p.ecount() == 3
        assert p.is_directed()

    def test_chain_pattern(self):
        p = build_motif_pattern(MOTIF_CHAIN, size=3)
        assert p.vcount() == 3
        assert p.ecount() == 2

    def test_fan_in_pattern(self):
        p = build_motif_pattern(MOTIF_FAN_IN, size=3)
        assert p.vcount() == 3
        assert p.ecount() == 2

    def test_size_4_pattern(self):
        p = build_motif_pattern(0, size=4)
        assert p.vcount() == 4


# --- Tests for find_motif_instances ---

class TestFindMotifInstances:
    def test_find_ffl_in_ffl_graph(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        assert len(instances) == 1
        inst = instances[0]
        assert inst.isoclass == MOTIF_FFL
        assert inst.label == "030T"
        assert set(inst.node_indices) == {0, 1, 2}
        assert len(inst.subgraph_edges) == 3
        assert inst.total_weight == pytest.approx(10.0)

    def test_find_chain_in_chain_graph(self):
        g = _make_chain_graph()
        instances = find_motif_instances(g, MOTIF_CHAIN)
        assert len(instances) == 1
        assert set(instances[0].node_indices) == {0, 1, 2}
        assert instances[0].total_weight == pytest.approx(10.0)

    def test_find_fan_in(self):
        g = _make_fan_in_graph()
        instances = find_motif_instances(g, MOTIF_FAN_IN)
        assert len(instances) == 1
        assert set(instances[0].node_indices) == {0, 1, 2}

    def test_no_ffl_in_chain_graph(self):
        g = _make_chain_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        assert len(instances) == 0

    def test_no_chain_in_fan_in_graph(self):
        g = _make_fan_in_graph()
        instances = find_motif_instances(g, MOTIF_CHAIN)
        assert len(instances) == 0

    def test_deduplication(self):
        """Symmetric mappings should be deduplicated."""
        g = _make_fan_in_graph()
        instances = find_motif_instances(g, MOTIF_FAN_IN)
        # Fan-in A->C, B->C has symmetric mappings (swap A and B),
        # but after dedup there should be exactly 1 instance
        assert len(instances) == 1

    def test_sort_by_weight(self):
        g = _make_multi_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL, sort_by="weight")
        assert len(instances) == 2
        # Higher weight instance (D,E,F) should come first
        assert instances[0].total_weight > instances[1].total_weight
        assert instances[0].total_weight == pytest.approx(30.0)
        assert instances[1].total_weight == pytest.approx(10.0)

    def test_sort_by_none(self):
        g = _make_multi_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL, sort_by="none")
        assert len(instances) == 2

    def test_max_instances(self):
        g = _make_multi_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL, max_instances=1)
        assert len(instances) == 1
        # Should be the highest-weight one
        assert instances[0].total_weight == pytest.approx(30.0)

    def test_node_roles_assigned(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        inst = instances[0]
        # FFL roles: regulator, mediator, target
        role_values = set(inst.node_roles.values())
        expected_roles = set(MOTIF_ROLES[MOTIF_FFL])
        assert role_values == expected_roles

    def test_undirected_raises(self):
        g = ig.Graph(directed=False)
        g.add_vertices(3)
        g.add_edges([(0, 1), (1, 2)])
        with pytest.raises(ValueError, match="directed"):
            find_motif_instances(g, MOTIF_CHAIN)

    def test_empty_graph(self):
        g = ig.Graph(directed=True)
        g.add_vertices(3)
        instances = find_motif_instances(g, MOTIF_FFL)
        assert len(instances) == 0

    def test_instance_is_dataclass(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        inst = instances[0]
        assert isinstance(inst, MotifInstance)
        assert isinstance(inst.node_indices, tuple)
        assert isinstance(inst.node_roles, dict)
        assert isinstance(inst.subgraph_edges, list)

    def test_subgraph_edges_are_valid(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        inst = instances[0]
        node_set = set(inst.node_indices)
        for u, v in inst.subgraph_edges:
            assert u in node_set
            assert v in node_set


# --- Tests for igraph-to-networkx conversion ---

class TestIgraphToNetworkx:
    def test_node_count_preserved(self):
        g = _make_ffl_graph()
        nxg = _igraph_to_networkx(g)
        assert nxg.number_of_nodes() == g.vcount()

    def test_edge_count_preserved(self):
        g = _make_ffl_graph()
        nxg = _igraph_to_networkx(g)
        assert nxg.number_of_edges() == g.ecount()

    def test_attributes_preserved(self):
        g = _make_ffl_graph()
        nxg = _igraph_to_networkx(g)
        assert nxg.nodes[0]["clerp"] == "input feature"
        assert nxg.nodes[0]["layer"] == 1

    def test_edge_attributes_preserved(self):
        g = _make_ffl_graph()
        nxg = _igraph_to_networkx(g)
        edge_data = nxg.get_edge_data(0, 1)
        assert edge_data["weight"] == 3.0
        assert edge_data["sign"] == "excitatory"


# --- Tests for layered layout ---

class TestComputeLayeredLayout:
    def test_all_nodes_have_positions(self):
        g = _make_ffl_graph()
        pos = _compute_layered_layout(g)
        assert len(pos) == g.vcount()
        for idx in range(g.vcount()):
            assert idx in pos

    def test_layers_separated_vertically(self):
        g = _make_ffl_graph()
        pos = _compute_layered_layout(g)
        # Node 0 is layer 1, node 1 is layer 5, node 2 is layer 10
        # Earlier layers should have higher y (less negative)
        assert pos[0][1] > pos[1][1]
        assert pos[1][1] > pos[2][1]

    def test_same_layer_same_y(self):
        g = _make_fan_in_graph()
        pos = _compute_layered_layout(g)
        # Nodes 0 and 1 are both layer 1
        assert pos[0][1] == pos[1][1]


# --- Tests for Neuronpedia-style layout ---

class TestComputeNeuronpediaLayout:
    def test_all_nodes_have_positions(self):
        g = _make_ffl_graph()
        pos, layers, labels = _compute_neuronpedia_layout(g)
        assert len(pos) == g.vcount()
        for idx in range(g.vcount()):
            assert idx in pos

    def test_embedding_at_bottom(self):
        g = _make_ffl_graph()
        pos, layers, labels = _compute_neuronpedia_layout(g)
        # Embedding node (layer=-1 mapped to Emb) should have lowest y
        emb_idx = 0  # layer=1 (embedding feature_type)
        logit_idx = 2  # layer=10 (logit feature_type)
        # The embedding node is at layer 1 (not -1) in this test graph
        # but the logit should be at the logit pseudo-layer (highest y)
        assert pos[logit_idx][1] > pos[emb_idx][1]

    def test_logit_at_top(self):
        g = _make_ffl_graph()
        pos, layers, labels = _compute_neuronpedia_layout(g)
        logit_idx = 2  # feature_type="logit"
        # Logit should have the highest y value
        max_y = max(p[1] for p in pos.values())
        assert pos[logit_idx][1] == max_y

    def test_layer_labels_include_emb_and_lgt(self):
        g = _make_ffl_graph()
        pos, layers, labels = _compute_neuronpedia_layout(g)
        assert "Emb" in labels or "L1" in labels  # depends on whether layer=-1 nodes exist
        assert "Lgt" in labels

    def test_returns_sorted_layers(self):
        g = _make_ffl_graph()
        pos, layers, labels = _compute_neuronpedia_layout(g)
        assert layers == sorted(layers)
        assert len(layers) == len(labels)

    def test_same_ctx_idx_same_base_x(self):
        """Nodes at the same ctx_idx should be near the same x."""
        g = _make_ffl_graph()  # all ctx_idx=1
        pos, _, _ = _compute_neuronpedia_layout(g)
        x_values = [pos[i][0] for i in range(3)]
        # All at ctx_idx=1, so base x should be 1.0 (with small jitter)
        for x in x_values:
            assert abs(x - 1.0) < 0.5  # within jitter range


# --- Smoke tests for plotting ---

class TestPlotGraphWithMotif:
    def test_returns_figure(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        fig = plot_graph_with_motif(g, instances[0])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_title(self):
        g = _make_ffl_graph()
        instances = find_motif_instances(g, MOTIF_FFL)
        fig = plot_graph_with_motif(g, instances[0], title="Test Title")
        ax = fig.axes[0]
        assert ax.get_title() == "Test Title"
        plt.close(fig)

    def test_chain_graph_plot(self):
        g = _make_chain_graph()
        instances = find_motif_instances(g, MOTIF_CHAIN)
        fig = plot_graph_with_motif(g, instances[0])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTopMotif:
    def test_returns_figure_and_instance(self):
        g = _make_ffl_graph()
        fig, inst = plot_top_motif(g, MOTIF_FFL)
        assert isinstance(fig, plt.Figure)
        assert isinstance(inst, MotifInstance)
        plt.close(fig)

    def test_rank_selection(self):
        g = _make_multi_ffl_graph()
        _, inst0 = plot_top_motif(g, MOTIF_FFL, rank=0)
        _, inst1 = plot_top_motif(g, MOTIF_FFL, rank=1)
        assert inst0.total_weight > inst1.total_weight
        plt.close("all")

    def test_no_instances_raises(self):
        g = _make_chain_graph()
        with pytest.raises(ValueError, match="No instances"):
            plot_top_motif(g, MOTIF_FFL)

    def test_rank_out_of_range_raises(self):
        g = _make_ffl_graph()
        with pytest.raises(ValueError, match="Rank"):
            plot_top_motif(g, MOTIF_FFL, rank=5)
