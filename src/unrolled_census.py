"""Unrolled motif enumeration via VF2 subgraph isomorphism.

Finds instances of unrolled motif templates in attribution graphs,
enforcing layer-ordering and edge-sign constraints. This is the
sign-aware, layer-aware analogue of motif_census.py's instance finding.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import igraph as ig
import numpy as np

from src.unrolled_motifs import (
    UnrolledMotifTemplate,
    get_effective_layer,
    build_catalog,
)


@dataclass
class UnrolledMotifInstance:
    """A specific instance of an unrolled motif found in a graph.

    Attributes:
        template_name: Name of the matched template.
        node_indices: Tuple of node indices in the original graph,
            ordered to match the template's node list.
        node_roles: Dict mapping node index to its semantic role.
        layers: List of effective layer indices for each node.
        edge_weights: List of raw (signed) weights for each template edge.
        edge_signs: List of signs ("excitatory"/"inhibitory") for each edge.
        total_weight: Sum of absolute edge weights in the instance.
        max_layer_gap: Maximum layer gap across all edges in the instance.
    """
    template_name: str
    node_indices: tuple[int, ...]
    node_roles: dict[int, str]
    layers: list[int]
    edge_weights: list[float]
    edge_signs: list[str]
    total_weight: float
    max_layer_gap: int


def _edge_sign_matches(
    graph_raw_weight: float,
    graph_sign: str,
    template_sign: int,
) -> bool:
    """Check whether a graph edge's sign matches the template requirement.

    Args:
        graph_raw_weight: The raw (signed) weight of the graph edge.
        graph_sign: The sign attribute ("excitatory" or "inhibitory").
        template_sign: Required sign (+1, -1, or 0 for any).

    Returns:
        True if the edge satisfies the sign constraint.
    """
    if template_sign == 0:
        return True
    if template_sign == +1:
        return graph_sign == "excitatory"
    if template_sign == -1:
        return graph_sign == "inhibitory"
    return True


def find_unrolled_instances(
    graph: ig.Graph,
    template: UnrolledMotifTemplate,
    weight_threshold: float = 0.0,
    max_layer_gap: int | None = None,
) -> list[UnrolledMotifInstance]:
    """Find all instances of an unrolled motif template in an attribution graph.

    Uses igraph's VF2 subgraph isomorphism with post-filtering for:
    1. Layer ordering: all edges must go forward in layer index
    2. Layer gap: no edge spans more than max_layer_gap layers
    3. Sign matching: edge signs must match template specification
    4. Weight threshold: all edges must have |weight| >= threshold

    Args:
        graph: A directed igraph.Graph with layer, sign, weight attributes.
        template: The unrolled motif template to search for.
        weight_threshold: Minimum absolute edge weight for all motif edges.
        max_layer_gap: Maximum layer gap per edge. Defaults to template's max_layer_gap.

    Returns:
        List of UnrolledMotifInstance, sorted by total_weight descending.
    """
    if not graph.is_directed():
        raise ValueError("Unrolled motif finding requires a directed graph")

    if max_layer_gap is None:
        max_layer_gap = template.max_layer_gap

    template_graph = template.to_template_graph()

    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False
    has_weight = "weight" in graph.es.attributes() if graph.ecount() > 0 else False
    has_raw_weight = "raw_weight" in graph.es.attributes() if graph.ecount() > 0 else False

    # Pre-prune: remove edges below weight threshold BEFORE VF2.
    # This dramatically reduces the combinatorial search space.
    if weight_threshold > 0 and has_weight:
        keep = [e.index for e in graph.es if abs(e["weight"]) >= weight_threshold]
        search_graph = graph.subgraph_edges(keep, delete_vertices=False)
    else:
        search_graph = graph

    # VF2 subgraph isomorphism: find all structural matches
    # Each mapping is a list where mapping[template_node_i] = graph_node_j
    raw_mappings = search_graph.get_subisomorphisms_vf2(template_graph)

    # Deduplicate: same set of graph nodes, same edge pattern
    seen: set[frozenset[int]] = set()
    instances: list[UnrolledMotifInstance] = []

    for mapping in raw_mappings:
        # Dedup by frozenset of graph node indices
        key = frozenset(mapping)
        if key in seen:
            continue

        # Validate all constraints for this mapping
        valid = True
        edge_weights: list[float] = []
        edge_signs: list[str] = []
        max_gap = 0

        for tmpl_edge in template.edges:
            src_tmpl_idx = template.node_ids.index(tmpl_edge["src"])
            tgt_tmpl_idx = template.node_ids.index(tmpl_edge["tgt"])

            graph_src = mapping[src_tmpl_idx]
            graph_tgt = mapping[tgt_tmpl_idx]

            # Check edge exists in pruned graph
            eid = search_graph.get_eid(graph_src, graph_tgt, error=False)
            if eid == -1:
                valid = False
                break

            # Weight threshold already enforced by pre-pruning;
            # still read weight for instance metadata.
            if has_weight:
                w = search_graph.es[eid]["weight"]
            else:
                w = 1.0

            # Check sign constraint
            sign_str = search_graph.es[eid]["sign"] if has_sign else "excitatory"
            raw_w = search_graph.es[eid]["raw_weight"] if has_raw_weight else w
            tmpl_sign = tmpl_edge.get("sign", 0)

            if not _edge_sign_matches(raw_w, sign_str, tmpl_sign):
                valid = False
                break

            # Check layer ordering and gap
            src_layer = get_effective_layer(search_graph, graph_src)
            tgt_layer = get_effective_layer(search_graph, graph_tgt)

            if src_layer >= tgt_layer:
                valid = False
                break

            gap = tgt_layer - src_layer
            if gap > max_layer_gap:
                valid = False
                break
            if gap < template.min_layer_gap:
                valid = False
                break

            max_gap = max(max_gap, gap)
            edge_weights.append(raw_w)
            edge_signs.append(sign_str)

        if not valid:
            continue

        seen.add(key)

        # Build instance
        node_indices = tuple(mapping)
        layers = [get_effective_layer(search_graph, n) for n in mapping]
        total_weight = sum(abs(w) for w in edge_weights)

        node_roles: dict[int, str] = {}
        for tmpl_idx, graph_node in enumerate(mapping):
            node_id = template.node_ids[tmpl_idx]
            role = template.roles.get(node_id, node_id)
            node_roles[graph_node] = role

        instances.append(UnrolledMotifInstance(
            template_name=template.name,
            node_indices=node_indices,
            node_roles=node_roles,
            layers=layers,
            edge_weights=edge_weights,
            edge_signs=edge_signs,
            total_weight=total_weight,
            max_layer_gap=max_gap,
        ))

    # Sort by total weight descending
    instances.sort(key=lambda x: x.total_weight, reverse=True)
    return instances


def run_unrolled_census(
    graph: ig.Graph,
    templates: list[UnrolledMotifTemplate] | None = None,
    weight_threshold: float = 0.0,
    max_layer_gap: int | None = None,
    max_instances_per_motif: int | None = None,
) -> dict[str, list[UnrolledMotifInstance]]:
    """Run a full unrolled motif census on an attribution graph.

    Finds instances of all templates and returns them grouped by motif name.

    Args:
        graph: A directed igraph.Graph.
        templates: List of templates to search for. Defaults to full catalog.
        weight_threshold: Minimum absolute edge weight.
        max_layer_gap: Maximum layer gap per edge.
        max_instances_per_motif: If set, keep only top-K instances per motif.

    Returns:
        Dict mapping template name to list of instances.
    """
    if templates is None:
        templates = build_catalog()

    results: dict[str, list[UnrolledMotifInstance]] = {}
    for template in templates:
        instances = find_unrolled_instances(
            graph,
            template,
            weight_threshold=weight_threshold,
            max_layer_gap=max_layer_gap,
        )
        if max_instances_per_motif is not None:
            instances = instances[:max_instances_per_motif]
        results[template.name] = instances

    return results


def unrolled_census_counts(
    census_results: dict[str, list[UnrolledMotifInstance]],
) -> dict[str, int]:
    """Extract instance counts from census results.

    Args:
        census_results: Output of run_unrolled_census().

    Returns:
        Dict mapping template name to instance count.
    """
    return {name: len(instances) for name, instances in census_results.items()}


def unrolled_census_summary(
    census_results: dict[str, list[UnrolledMotifInstance]],
) -> dict[str, dict[str, Any]]:
    """Build a summary dict of the unrolled census results.

    Args:
        census_results: Output of run_unrolled_census().

    Returns:
        Dict mapping template name to summary stats (count, mean_weight,
        max_weight, mean_layer_gap).
    """
    summary: dict[str, dict[str, Any]] = {}
    for name, instances in census_results.items():
        if not instances:
            summary[name] = {
                "count": 0,
                "mean_weight": 0.0,
                "max_weight": 0.0,
                "mean_layer_gap": 0.0,
            }
            continue

        weights = [inst.total_weight for inst in instances]
        gaps = [inst.max_layer_gap for inst in instances]
        summary[name] = {
            "count": len(instances),
            "mean_weight": float(np.mean(weights)),
            "max_weight": float(np.max(weights)),
            "mean_layer_gap": float(np.mean(gaps)),
        }

    return summary


def fast_unrolled_counts(
    graph: ig.Graph,
    max_layer_gap: int = 5,
    min_layer_gap: int = 1,
) -> dict[str, int]:
    """Count unrolled motif instances using direct adjacency lookups.

    Orders of magnitude faster than VF2 for large/dense graphs because it
    avoids enumerating all subgraph isomorphisms.  Uses O(V*D) memory
    instead of O(instances) which can be millions for dense graphs.

    Args:
        graph: A directed igraph.Graph with layer, sign, weight attributes.
        max_layer_gap: Maximum layer gap per edge (default 5).
        min_layer_gap: Minimum layer gap per edge (default 1).

    Returns:
        Dict mapping template name to instance count.
    """
    V = graph.vcount()
    if V == 0 or graph.ecount() == 0:
        return {
            "cross_chain_inhibition": 0,
            "feedforward_damping": 0,
            "feedforward_amplification": 0,
            "residual_self_loop_positive": 0,
            "residual_self_loop_negative": 0,
            "coherent_ffl": 0,
            "incoherent_ffl": 0,
            "cross_chain_toggle": 0,
        }

    # Precompute effective layers
    layers = [get_effective_layer(graph, v) for v in range(V)]
    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False

    # Build adjacency sets filtered by sign and layer gap.
    # Only edges with min_layer_gap <= gap <= max_layer_gap are included.
    exc_succ: dict[int, set[int]] = defaultdict(set)
    inh_succ: dict[int, set[int]] = defaultdict(set)
    exc_pred: dict[int, set[int]] = defaultdict(set)
    inh_pred: dict[int, set[int]] = defaultdict(set)

    for e in graph.es:
        s, t = e.source, e.target
        gap = layers[t] - layers[s]
        if gap < min_layer_gap or gap > max_layer_gap:
            continue
        sign = e["sign"] if has_sign else "excitatory"
        if sign == "excitatory":
            exc_succ[s].add(t)
            exc_pred[t].add(s)
        else:
            inh_succ[s].add(t)
            inh_pred[t].add(s)

    counts: dict[str, int] = {}

    # ── 2-node templates ──────────────────────────────────────────────
    # residual_self_loop_positive: A→B(+), forward layer ordering
    counts["residual_self_loop_positive"] = sum(
        len(targets) for targets in exc_succ.values()
    )
    # residual_self_loop_negative: A→B(-), forward layer ordering
    counts["residual_self_loop_negative"] = sum(
        len(targets) for targets in inh_succ.values()
    )

    # ── 3-node chain templates ────────────────────────────────────────
    # feedforward_damping: A→B(+), B→C(-), layers A < B < C
    # All nodes distinct (guaranteed by gap ≥ 1 → strict layer ordering)
    count_damp = 0
    for b in range(V):
        count_damp += len(exc_pred[b]) * len(inh_succ[b])
    counts["feedforward_damping"] = count_damp

    # feedforward_amplification: A→B(+), B→C(+), layers A < B < C
    count_amp = 0
    for b in range(V):
        count_amp += len(exc_pred[b]) * len(exc_succ[b])
    counts["feedforward_amplification"] = count_amp

    # ── 3-node triangle templates (FFLs) ──────────────────────────────
    # coherent_ffl: A→B(+), B→C(+), A→C(+), layers A < B < C
    # A→C edge gap already enforced by adjacency structure.
    count_cffl = 0
    for b in range(V):
        for a in exc_pred[b]:
            a_exc = exc_succ[a]
            for c in exc_succ[b]:
                if c in a_exc:
                    count_cffl += 1
    counts["coherent_ffl"] = count_cffl

    # incoherent_ffl: A→B(+), B→C(+), A→C(-), layers A < B < C
    count_iffl = 0
    for b in range(V):
        for a in exc_pred[b]:
            a_inh = inh_succ[a]
            for c in exc_succ[b]:
                if c in a_inh:
                    count_iffl += 1
    counts["incoherent_ffl"] = count_iffl

    # ── 4-node template ───────────────────────────────────────────────
    # cross_chain_inhibition:
    #   A_e→A_l(+), B_e→B_l(+), A_e→B_l(-), B_e→A_l(-)
    #   All 4 nodes distinct.  Z2 symmetry (swap chains) handled by
    #   iterating ordered pairs (a_e < b_e) — each instance counted once.
    count_xci = 0
    for a_e in range(V):
        a_exc = exc_succ[a_e]
        a_inh = inh_succ[a_e]
        if not a_exc or not a_inh:
            continue
        for b_e in range(a_e + 1, V):
            b_exc = exc_succ[b_e]
            b_inh = inh_succ[b_e]
            if not b_exc or not b_inh:
                continue
            # A_l: excitatory from A_e AND inhibitory from B_e
            a_l_cands = (a_exc & b_inh) - {a_e, b_e}
            # B_l: excitatory from B_e AND inhibitory from A_e
            b_l_cands = (b_exc & a_inh) - {a_e, b_e}
            if not a_l_cands or not b_l_cands:
                continue
            # Subtract diagonal (A_l == B_l)
            overlap = len(a_l_cands & b_l_cands)
            count_xci += len(a_l_cands) * len(b_l_cands) - overlap
    counts["cross_chain_inhibition"] = count_xci

    # ── 5-node template ───────────────────────────────────────────────
    # cross_chain_toggle:
    #   Bias→A_m(+), Bias→B_m(+), A_m→B_l(-), B_m→A_l(-)
    #   All 5 nodes distinct.  Z2 symmetry (swap A/B chains) handled by
    #   iterating ordered pairs (a_m < b_m index in targets list).
    count_xct = 0
    for bias in range(V):
        targets = list(exc_succ[bias])
        k = len(targets)
        if k < 2:
            continue
        for i in range(k):
            a_m = targets[i]
            a_m_inh = inh_succ[a_m]  # B_l candidates
            if not a_m_inh:
                continue
            for j in range(i + 1, k):
                b_m = targets[j]
                b_m_inh = inh_succ[b_m]  # A_l candidates
                if not b_m_inh:
                    continue
                excluded = {bias, a_m, b_m}
                b_l_cands = a_m_inh - excluded
                a_l_cands = b_m_inh - excluded
                if not a_l_cands or not b_l_cands:
                    continue
                overlap = len(a_l_cands & b_l_cands)
                count_xct += len(a_l_cands) * len(b_l_cands) - overlap
    counts["cross_chain_toggle"] = count_xct

    return counts
