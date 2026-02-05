"""Motif enumeration via igraph's FANMOD implementation.

Wraps igraph.motifs_randesu() for directed graphs (size 3 and 4),
handles NaN replacement, and provides triad class labeling.
Also provides motif instance finding via VF2 subgraph isomorphism.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np


# MAN triad labels for the 16 directed triad isomorphism classes (size 3).
# Index corresponds to igraph's isoclass ID for size-3 directed motifs.
# WARNING: igraph's isoclass ordering differs from the standard MAN census ordering.
# This mapping was empirically verified against igraph's Graph.isoclass() method.
TRIAD_LABELS: list[str] = [
    "003",   # 0:  no edges (empty triad)
    "012",   # 1:  A->B (single edge)
    "021U",  # 2:  A->C, B->C (fan-in)
    "102",   # 3:  A<->B (mutual)
    "021C",  # 4:  A->B->C (chain)
    "111U",  # 5:  A<->B, C->A (mutual + in)
    "021D",  # 6:  A->B, A->C (fan-out)
    "030T",  # 7:  A->B, A->C, B->C (feedforward loop)
    "120U",  # 8:  A->B, A->C, B<->C (mutual + fan-in)
    "111D",  # 9:  A<->B, A->C (mutual + out)
    "201",   # 10: A<->B, A<->C (double mutual)
    "030C",  # 11: A->B->C->A (cycle)
    "120C",  # 12: A<->B, A->C, C->B (regulated mutual)
    "120D",  # 13: A<->B, A->C, B->C (mutual + FFL)
    "210",   # 14: A<->B, A<->C, B->C (dense partial)
    "300",   # 15: A<->B, A<->C, B<->C (complete)
]

# Key motif indices for quick reference (igraph isoclass IDs)
MOTIF_FAN_IN = 2        # 021U
MOTIF_CHAIN = 4         # 021C
MOTIF_FAN_OUT = 6       # 021D
MOTIF_FFL = 7           # 030T (feedforward loop)
MOTIF_CYCLE = 11        # 030C
MOTIF_COMPLETE = 15     # 300

# Connected triad indices (skip 003 which is the empty triad)
CONNECTED_TRIAD_INDICES: list[int] = list(range(1, 16))


@dataclass
class MotifCensusResult:
    """Result of a motif census computation.

    Attributes:
        size: Motif size (3 or 4).
        raw_counts: Raw motif counts indexed by isomorphism class ID.
        labels: Human-readable labels for each class (only for size 3).
        graph_nodes: Number of nodes in the graph.
        graph_edges: Number of edges in the graph.
    """
    size: int
    raw_counts: list[int]
    labels: list[str] = field(default_factory=list)
    graph_nodes: int = 0
    graph_edges: int = 0

    @property
    def n_classes(self) -> int:
        """Number of isomorphism classes."""
        return len(self.raw_counts)

    def connected_counts(self) -> dict[str, int]:
        """Return counts for connected triads only (size 3).

        Returns:
            Dict mapping triad label to count.
        """
        if self.size != 3:
            raise ValueError("connected_counts() only supported for size 3")
        return {
            self.labels[i]: self.raw_counts[i]
            for i in CONNECTED_TRIAD_INDICES
        }

    def as_array(self) -> np.ndarray:
        """Return counts as a numpy array."""
        return np.array(self.raw_counts, dtype=np.float64)

    def as_dict(self) -> dict[str, int]:
        """Return counts as a label->count dict (size 3 only)."""
        if self.labels:
            return {label: count for label, count in zip(self.labels, self.raw_counts)}
        return {str(i): count for i, count in enumerate(self.raw_counts)}


def compute_motif_census(graph: ig.Graph, size: int = 3) -> MotifCensusResult:
    """Compute the motif census for a directed graph.

    Uses igraph's motifs_randesu() which implements the FANMOD algorithm.
    NaN values (for inapplicable motif classes) are replaced with 0.

    Args:
        graph: A directed igraph.Graph.
        size: Motif size (3 or 4). Default is 3 (16 triad classes).

    Returns:
        MotifCensusResult with raw counts and labels.

    Raises:
        ValueError: If graph is not directed or size is not 3 or 4.
    """
    if not graph.is_directed():
        raise ValueError("Motif census requires a directed graph")
    if size not in (3, 4):
        raise ValueError(f"Motif size must be 3 or 4, got {size}")

    raw = graph.motifs_randesu(size=size)

    # Replace NaN with 0
    counts = [0 if (c is None or (isinstance(c, float) and math.isnan(c))) else int(c)
              for c in raw]

    labels = TRIAD_LABELS if size == 3 else [str(i) for i in range(len(counts))]

    return MotifCensusResult(
        size=size,
        raw_counts=counts,
        labels=labels,
        graph_nodes=graph.vcount(),
        graph_edges=graph.ecount(),
    )


def compute_triad_census_networkx(graph: ig.Graph) -> dict[str, int]:
    """Cross-validate triad census using NetworkX.

    Converts the igraph graph to NetworkX and runs triadic_census().
    Note: NetworkX and igraph may use different isomorphism class orderings.

    Args:
        graph: A directed igraph.Graph.

    Returns:
        Dict mapping NetworkX triad names to counts.
    """
    import networkx as nx

    # Convert igraph to networkx
    nx_graph = nx.DiGraph()
    for v in graph.vs:
        nx_graph.add_node(v.index, **{attr: v[attr] for attr in graph.vs.attributes()})
    for e in graph.es:
        nx_graph.add_edge(
            e.source, e.target,
            **{attr: e[attr] for attr in graph.es.attributes()},
        )

    return nx.triadic_census(nx_graph)


def motif_frequencies(result: MotifCensusResult) -> np.ndarray:
    """Compute relative frequencies of each motif class.

    Args:
        result: A MotifCensusResult.

    Returns:
        Array of frequencies (each count / total count). Zero if total is 0.
    """
    counts = result.as_array()
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts)
    return counts / total


def enriched_motifs(
    z_scores: np.ndarray,
    threshold: float = 2.0,
    labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Identify enriched and anti-enriched motifs from Z-scores.

    Args:
        z_scores: Array of Z-scores per motif class.
        threshold: Absolute Z-score threshold for significance.
        labels: Optional labels for each class.

    Returns:
        List of dicts with keys: index, label, z_score, direction.
    """
    results = []
    for i, z in enumerate(z_scores):
        if abs(z) >= threshold:
            label = labels[i] if labels and i < len(labels) else str(i)
            results.append({
                "index": i,
                "label": label,
                "z_score": float(z),
                "direction": "enriched" if z > 0 else "anti-enriched",
            })
    results.sort(key=lambda x: abs(x["z_score"]), reverse=True)
    return results


# --- Motif instance finding ---

# Semantic role names for key size-3 motif types.
# Keys are igraph isoclass IDs; values are ordered role lists matching the
# canonical node ordering returned by get_subisomorphisms_vf2().
MOTIF_ROLES: dict[int, list[str]] = {
    2:  ["source_a", "source_b", "target"],       # 021U fan-in
    4:  ["source", "mediator", "target"],          # 021C chain
    6:  ["source", "target_a", "target_b"],        # 021D fan-out
    7:  ["regulator", "mediator", "target"],        # 030T feedforward loop
    11: ["node_a", "node_b", "node_c"],            # 030C cycle
    15: ["node_a", "node_b", "node_c"],            # 300 complete
}


@dataclass
class MotifInstance:
    """A specific instance of a motif found in a graph.

    Attributes:
        isoclass: igraph isomorphism class ID.
        label: Human-readable motif label (e.g., "030T").
        node_indices: Tuple of node indices in the original graph.
        node_roles: Dict mapping node index to its semantic role name.
        subgraph_edges: List of (source, target) tuples for edges within the motif.
        total_weight: Sum of absolute edge weights within the motif instance.
    """
    isoclass: int
    label: str
    node_indices: tuple[int, ...]
    node_roles: dict[int, str]
    subgraph_edges: list[tuple[int, int]]
    total_weight: float


def build_motif_pattern(isoclass: int, size: int = 3) -> ig.Graph:
    """Build a canonical pattern graph for a given isomorphism class.

    Args:
        isoclass: igraph isomorphism class ID.
        size: Number of nodes (3 or 4).

    Returns:
        A directed igraph.Graph representing the canonical motif pattern.
    """
    return ig.Graph.Isoclass(n=size, cls=isoclass, directed=True)


def find_motif_instances(
    graph: ig.Graph,
    motif_isoclass: int,
    size: int = 3,
    max_instances: int | None = None,
    sort_by: str = "weight",
) -> list[MotifInstance]:
    """Find all instances of a specific motif in the graph.

    Uses VF2 subgraph isomorphism to enumerate all occurrences of the
    given motif pattern. Deduplicates symmetric mappings and optionally
    sorts by total edge weight.

    Args:
        graph: A directed igraph.Graph.
        motif_isoclass: igraph isomorphism class ID of the motif to find.
        size: Motif size (3 or 4).
        max_instances: If set, return at most this many instances.
        sort_by: Sort criterion — "weight" (descending total weight) or "none".

    Returns:
        List of MotifInstance objects, sorted by total_weight descending
        if sort_by="weight".

    Raises:
        ValueError: If graph is not directed.
    """
    if not graph.is_directed():
        raise ValueError("Motif instance finding requires a directed graph")

    pattern = build_motif_pattern(isoclass=motif_isoclass, size=size)

    # get_subisomorphisms_vf2 returns list of lists:
    # each inner list maps pattern node i → graph node index
    raw_mappings = graph.get_subisomorphisms_vf2(pattern)

    # Deduplicate: different mappings can produce the same set of nodes
    # for symmetric motifs. Use frozenset of node indices as dedup key.
    seen: set[frozenset[int]] = set()
    unique_mappings: list[list[int]] = []
    for mapping in raw_mappings:
        key = frozenset(mapping)
        if key not in seen:
            seen.add(key)
            unique_mappings.append(mapping)

    # Determine label
    label = TRIAD_LABELS[motif_isoclass] if size == 3 and motif_isoclass < 16 else str(motif_isoclass)

    # Get role names
    roles_template = MOTIF_ROLES.get(motif_isoclass, [f"node_{i}" for i in range(size)])

    # Check if graph has weight attribute on edges
    has_weights = "weight" in graph.es.attributes() if graph.ecount() > 0 else False

    instances: list[MotifInstance] = []
    for mapping in unique_mappings:
        node_set = set(mapping)
        node_indices = tuple(mapping)

        # Assign roles based on pattern node ordering
        node_roles = {}
        for pattern_idx, graph_node in enumerate(mapping):
            if pattern_idx < len(roles_template):
                node_roles[graph_node] = roles_template[pattern_idx]
            else:
                node_roles[graph_node] = f"node_{pattern_idx}"

        # Find edges within the motif subgraph
        subgraph_edges: list[tuple[int, int]] = []
        total_weight = 0.0
        for e in graph.es:
            if e.source in node_set and e.target in node_set:
                subgraph_edges.append((e.source, e.target))
                if has_weights:
                    total_weight += abs(e["weight"])

        instances.append(MotifInstance(
            isoclass=motif_isoclass,
            label=label,
            node_indices=node_indices,
            node_roles=node_roles,
            subgraph_edges=subgraph_edges,
            total_weight=total_weight,
        ))

    if sort_by == "weight":
        instances.sort(key=lambda inst: inst.total_weight, reverse=True)

    if max_instances is not None:
        instances = instances[:max_instances]

    return instances
