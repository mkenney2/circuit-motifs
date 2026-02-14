"""Unrolled regulatory motif templates for feedforward-compatible analysis.

Defines motif templates that represent feedforward-compatible analogues of
classic recurrent regulatory motifs (Alon, 2007). Each template is a small
directed graph pattern with layer-ordering and edge-sign constraints.

The key insight: transformers are feedforward, so recurrent motifs like
mutual inhibition and feedback loops cannot exist as literal cycles. But
their computational functions may be achieved by "unrolled" wiring patterns
spread across layers.

Templates:
    1. Cross-chain inhibition (unrolled mutual inhibition) — 4 nodes
    2. Feedforward damping (unrolled negative feedback) — 3 nodes
    3. Feedforward amplification (unrolled positive feedback) — 3 nodes
    4. Residual self-loop positive (unrolled positive autoregulation) — 2 nodes
    5. Residual self-loop negative (unrolled negative autoregulation) — 2 nodes
    6. Coherent FFL (sign-coherent feedforward loop) — 3 nodes
    7. Incoherent FFL (sign-incoherent feedforward loop) — 3 nodes
    8. Cross-chain toggle (unrolled bistable switch) — 5 nodes
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np


# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------

def get_layer_index(graph: ig.Graph, node_id: int) -> int:
    """Extract the transformer layer index for a node.

    Args:
        graph: An igraph DiGraph with a "layer" vertex attribute.
        node_id: The integer vertex index.

    Returns:
        Integer layer index. -1 for embedding nodes, -2 for unparseable.
    """
    return graph.vs[node_id]["layer"]


def get_effective_layer(graph: ig.Graph, node_id: int) -> int:
    """Get the effective layer for ordering, mapping logit nodes above all real layers.

    Embedding nodes get layer -1, logit nodes get max_layer + 1,
    and transcoder features keep their actual layer.

    Args:
        graph: An igraph DiGraph with "layer" and "feature_type" vertex attributes.
        node_id: The integer vertex index.

    Returns:
        Effective layer index for ordering comparisons.
    """
    v = graph.vs[node_id]
    ft = v["feature_type"] if "feature_type" in graph.vs.attributes() else ""
    layer = v["layer"] if "layer" in graph.vs.attributes() else 0

    if ft == "logit":
        # Place logit nodes above the highest real layer
        max_layer = max(
            (v2["layer"] for v2 in graph.vs if v2["layer"] >= 0),
            default=0,
        )
        return max_layer + 1
    return layer


def validate_layer_ordering(graph: ig.Graph) -> tuple[bool, list[tuple[int, int, int, int]]]:
    """Confirm all edges go forward in layer index.

    Args:
        graph: An igraph DiGraph with "layer" vertex attribute.

    Returns:
        Tuple of (is_valid, list of backward edges as (src, tgt, src_layer, tgt_layer)).
    """
    backward_edges = []
    for edge in graph.es:
        src_layer = get_effective_layer(graph, edge.source)
        tgt_layer = get_effective_layer(graph, edge.target)
        if src_layer >= tgt_layer:
            backward_edges.append((edge.source, edge.target, src_layer, tgt_layer))
    return len(backward_edges) == 0, backward_edges


def layer_gap_distribution(graph: ig.Graph) -> dict[int, int]:
    """Compute histogram of (target_layer - source_layer) for all edges.

    Args:
        graph: An igraph DiGraph with "layer" vertex attribute.

    Returns:
        Dict mapping layer gap to count.
    """
    gaps: list[int] = []
    for edge in graph.es:
        src_layer = get_effective_layer(graph, edge.source)
        tgt_layer = get_effective_layer(graph, edge.target)
        gaps.append(tgt_layer - src_layer)
    return dict(Counter(gaps))


def layer_stats(graph: ig.Graph) -> dict[str, Any]:
    """Compute layer structure statistics for an attribution graph.

    Args:
        graph: An igraph DiGraph with "layer" vertex attribute.

    Returns:
        Dict with n_layers, nodes_per_layer, gap_distribution, all_forward.
    """
    nodes_per_layer: dict[int, int] = Counter()
    for v in graph.vs:
        nodes_per_layer[get_effective_layer(graph, v.index)] += 1

    gap_dist = layer_gap_distribution(graph)
    is_valid, backward = validate_layer_ordering(graph)

    return {
        "n_layers": len(nodes_per_layer),
        "nodes_per_layer": dict(nodes_per_layer),
        "gap_distribution": gap_dist,
        "all_forward": is_valid,
        "n_backward_edges": len(backward),
    }


# ---------------------------------------------------------------------------
# Template definition
# ---------------------------------------------------------------------------

@dataclass
class UnrolledMotifTemplate:
    """A feedforward-compatible analogue of a classic regulatory motif.

    Each template defines a small directed graph pattern with:
    - Labeled nodes with chain membership and relative ordering
    - Signed edges (+1 excitatory, -1 inhibitory, 0 any)
    - Layer-gap constraints for matching

    Attributes:
        name: Machine-readable identifier (e.g. "cross_chain_inhibition").
        classic_analogue: Name of the classic recurrent motif this unrolls.
        description: Human-readable description of the motif's function.
        nodes: List of node dicts with keys: id, chain, relative_order.
        edges: List of edge dicts with keys: src, tgt, sign.
        min_layer_gap: Minimum layer distance for each edge during matching.
        max_layer_gap: Maximum layer distance for each edge during matching.
        roles: Dict mapping node id to semantic role name.
    """
    name: str
    classic_analogue: str
    description: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    min_layer_gap: int = 1
    max_layer_gap: int = 5
    roles: dict[str, str] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    @property
    def node_ids(self) -> list[str]:
        return [n["id"] for n in self.nodes]

    def to_template_graph(self) -> ig.Graph:
        """Convert to an igraph DiGraph for VF2 matching.

        The template graph has:
        - Vertex attribute "template_id": the node's id string
        - Vertex attribute "chain": the chain membership
        - Vertex attribute "relative_order": ordering within chain
        - Edge attribute "sign": +1, -1, or 0 (any)

        Returns:
            A directed igraph.Graph representing the template.
        """
        g = ig.Graph(directed=True)
        id_to_idx: dict[str, int] = {}

        for i, node in enumerate(self.nodes):
            g.add_vertex(
                template_id=node["id"],
                chain=node.get("chain", 0),
                relative_order=node.get("relative_order", i),
            )
            id_to_idx[node["id"]] = i

        for edge in self.edges:
            src_idx = id_to_idx[edge["src"]]
            tgt_idx = id_to_idx[edge["tgt"]]
            g.add_edge(src_idx, tgt_idx, sign=edge.get("sign", 0))

        return g


# ---------------------------------------------------------------------------
# Motif catalog
# ---------------------------------------------------------------------------

def _build_cross_chain_inhibition() -> UnrolledMotifTemplate:
    """Cross-chain inhibition: unrolled mutual inhibition (4 nodes).

    Classic: A -| B, B -| A (mutual inhibition)
    Unrolled:
        A_early --(+)--> A_late    (chain A continues)
        B_early --(+)--> B_late    (chain B continues)
        A_early --(−)--> B_late    (A suppresses B)
        B_early --(−)--> A_late    (B suppresses A)
    """
    return UnrolledMotifTemplate(
        name="cross_chain_inhibition",
        classic_analogue="mutual_inhibition",
        description=(
            "Two processing chains mutually suppress each other across layers. "
            "The feedforward analogue of mutual inhibition — implements "
            "competition between alternative interpretations."
        ),
        nodes=[
            {"id": "A_early", "chain": 0, "relative_order": 0},
            {"id": "B_early", "chain": 1, "relative_order": 0},
            {"id": "A_late", "chain": 0, "relative_order": 1},
            {"id": "B_late", "chain": 1, "relative_order": 1},
        ],
        edges=[
            {"src": "A_early", "tgt": "A_late", "sign": +1},
            {"src": "B_early", "tgt": "B_late", "sign": +1},
            {"src": "A_early", "tgt": "B_late", "sign": -1},
            {"src": "B_early", "tgt": "A_late", "sign": -1},
        ],
        roles={
            "A_early": "chain_a_source",
            "B_early": "chain_b_source",
            "A_late": "chain_a_target",
            "B_late": "chain_b_target",
        },
    )


def _build_feedforward_damping() -> UnrolledMotifTemplate:
    """Feedforward damping: unrolled negative feedback (3 nodes).

    Classic: A → B -| A (negative feedback)
    Unrolled:
        A_early --(+)--> B_mid --(−)--> A_late
    """
    return UnrolledMotifTemplate(
        name="feedforward_damping",
        classic_analogue="negative_feedback",
        description=(
            "A feature activates an intermediary that then suppresses a "
            "downstream instance of the same logical role. Implements "
            "self-regulation / gain control."
        ),
        nodes=[
            {"id": "A_early", "chain": 0, "relative_order": 0},
            {"id": "B_mid", "chain": 1, "relative_order": 1},
            {"id": "A_late", "chain": 0, "relative_order": 2},
        ],
        edges=[
            {"src": "A_early", "tgt": "B_mid", "sign": +1},
            {"src": "B_mid", "tgt": "A_late", "sign": -1},
        ],
        roles={
            "A_early": "source",
            "B_mid": "damper",
            "A_late": "target",
        },
    )


def _build_feedforward_amplification() -> UnrolledMotifTemplate:
    """Feedforward amplification: unrolled positive feedback (3 nodes).

    Classic: A → B → A (positive feedback)
    Unrolled:
        A_early --(+)--> B_mid --(+)--> A_late
    """
    return UnrolledMotifTemplate(
        name="feedforward_amplification",
        classic_analogue="positive_feedback",
        description=(
            "A feature activates an intermediary that reinforces a "
            "downstream instance of the same logical role. Implements "
            "signal amplification / persistence."
        ),
        nodes=[
            {"id": "A_early", "chain": 0, "relative_order": 0},
            {"id": "B_mid", "chain": 1, "relative_order": 1},
            {"id": "A_late", "chain": 0, "relative_order": 2},
        ],
        edges=[
            {"src": "A_early", "tgt": "B_mid", "sign": +1},
            {"src": "B_mid", "tgt": "A_late", "sign": +1},
        ],
        roles={
            "A_early": "source",
            "B_mid": "amplifier",
            "A_late": "target",
        },
    )


def _build_residual_self_loop_positive() -> UnrolledMotifTemplate:
    """Residual self-loop (positive): unrolled positive autoregulation (2 nodes).

    Classic: A → A (positive autoregulation)
    Unrolled:
        A_early --(+)--> A_late
    """
    return UnrolledMotifTemplate(
        name="residual_self_loop_positive",
        classic_analogue="positive_autoregulation",
        description=(
            "A feature at an early layer excites the same logical role at a "
            "later layer. Self-reinforcement through the residual stream."
        ),
        nodes=[
            {"id": "A_early", "chain": 0, "relative_order": 0},
            {"id": "A_late", "chain": 0, "relative_order": 1},
        ],
        edges=[
            {"src": "A_early", "tgt": "A_late", "sign": +1},
        ],
        roles={
            "A_early": "source",
            "A_late": "target",
        },
    )


def _build_residual_self_loop_negative() -> UnrolledMotifTemplate:
    """Residual self-loop (negative): unrolled negative autoregulation (2 nodes).

    Classic: A -| A (negative autoregulation)
    Unrolled:
        A_early --(−)--> A_late
    """
    return UnrolledMotifTemplate(
        name="residual_self_loop_negative",
        classic_analogue="negative_autoregulation",
        description=(
            "A feature at an early layer suppresses the same logical role at a "
            "later layer. Self-dampening / gain control."
        ),
        nodes=[
            {"id": "A_early", "chain": 0, "relative_order": 0},
            {"id": "A_late", "chain": 0, "relative_order": 1},
        ],
        edges=[
            {"src": "A_early", "tgt": "A_late", "sign": -1},
        ],
        roles={
            "A_early": "source",
            "A_late": "target",
        },
    )


def _build_coherent_ffl() -> UnrolledMotifTemplate:
    """Coherent FFL: 030T with all-excitatory edges (3 nodes).

    A --(+)--> B --(+)--> C
    A --(+)-----------> C
    Both paths have the same sign → coherent signal integration.
    """
    return UnrolledMotifTemplate(
        name="coherent_ffl",
        classic_analogue="coherent_feedforward_loop",
        description=(
            "A feedforward loop where the direct and indirect paths have "
            "the same sign. Implements AND-gate logic or persistent signal "
            "detection (Mangan & Alon, 2003)."
        ),
        nodes=[
            {"id": "A", "chain": 0, "relative_order": 0},
            {"id": "B", "chain": 1, "relative_order": 1},
            {"id": "C", "chain": 2, "relative_order": 2},
        ],
        edges=[
            {"src": "A", "tgt": "B", "sign": +1},
            {"src": "B", "tgt": "C", "sign": +1},
            {"src": "A", "tgt": "C", "sign": +1},
        ],
        roles={
            "A": "regulator",
            "B": "mediator",
            "C": "target",
        },
    )


def _build_incoherent_ffl() -> UnrolledMotifTemplate:
    """Incoherent FFL: 030T with mixed-sign edges (3 nodes).

    A --(+)--> B --(+)--> C
    A --(−)-----------> C
    The direct path inhibits while the indirect path excites (or vice versa).
    Implements pulse generation or fold-change detection.
    """
    return UnrolledMotifTemplate(
        name="incoherent_ffl",
        classic_analogue="incoherent_feedforward_loop",
        description=(
            "A feedforward loop where the direct and indirect paths have "
            "opposing signs. Implements output competition — one path "
            "excites the target while the other suppresses it."
        ),
        nodes=[
            {"id": "A", "chain": 0, "relative_order": 0},
            {"id": "B", "chain": 1, "relative_order": 1},
            {"id": "C", "chain": 2, "relative_order": 2},
        ],
        edges=[
            {"src": "A", "tgt": "B", "sign": +1},
            {"src": "B", "tgt": "C", "sign": +1},
            {"src": "A", "tgt": "C", "sign": -1},
        ],
        roles={
            "A": "regulator",
            "B": "mediator",
            "C": "target",
        },
    )


def _build_cross_chain_toggle() -> UnrolledMotifTemplate:
    """Cross-chain toggle: unrolled bistable switch (5 nodes).

    Classic: A -| B, B -| A with upstream bias
    Unrolled:
        Bias --(+)--> A_mid --(−)--> B_late
        Bias --(+)--> B_mid --(−)--> A_late
    """
    return UnrolledMotifTemplate(
        name="cross_chain_toggle",
        classic_analogue="bistable_switch",
        description=(
            "An upstream bias node activates two competing chains that "
            "mutually suppress each other's downstream targets. Implements "
            "winner-take-all competition or attention-based routing."
        ),
        nodes=[
            {"id": "Bias", "chain": 2, "relative_order": 0},
            {"id": "A_mid", "chain": 0, "relative_order": 1},
            {"id": "B_mid", "chain": 1, "relative_order": 1},
            {"id": "A_late", "chain": 0, "relative_order": 2},
            {"id": "B_late", "chain": 1, "relative_order": 2},
        ],
        edges=[
            {"src": "Bias", "tgt": "A_mid", "sign": +1},
            {"src": "Bias", "tgt": "B_mid", "sign": +1},
            {"src": "A_mid", "tgt": "B_late", "sign": -1},
            {"src": "B_mid", "tgt": "A_late", "sign": -1},
        ],
        roles={
            "Bias": "bias",
            "A_mid": "chain_a_mid",
            "B_mid": "chain_b_mid",
            "A_late": "chain_a_target",
            "B_late": "chain_b_target",
        },
    )


# ---------------------------------------------------------------------------
# Catalog registry
# ---------------------------------------------------------------------------

def build_catalog() -> list[UnrolledMotifTemplate]:
    """Build the full catalog of unrolled motif templates.

    Returns:
        List of all 8 UnrolledMotifTemplate objects.
    """
    return [
        _build_cross_chain_inhibition(),
        _build_feedforward_damping(),
        _build_feedforward_amplification(),
        _build_residual_self_loop_positive(),
        _build_residual_self_loop_negative(),
        _build_coherent_ffl(),
        _build_incoherent_ffl(),
        _build_cross_chain_toggle(),
    ]


# Convenience: name → template mapping
CATALOG: dict[str, UnrolledMotifTemplate] = {t.name: t for t in build_catalog()}

# Quick access constants
MOTIF_CROSS_CHAIN_INHIBITION = "cross_chain_inhibition"
MOTIF_FEEDFORWARD_DAMPING = "feedforward_damping"
MOTIF_FEEDFORWARD_AMPLIFICATION = "feedforward_amplification"
MOTIF_SELF_LOOP_POSITIVE = "residual_self_loop_positive"
MOTIF_SELF_LOOP_NEGATIVE = "residual_self_loop_negative"
MOTIF_COHERENT_FFL = "coherent_ffl"
MOTIF_INCOHERENT_FFL = "incoherent_ffl"
MOTIF_CROSS_CHAIN_TOGGLE = "cross_chain_toggle"
