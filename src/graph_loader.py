"""Parse circuit-tracer / Neuronpedia JSON attribution graphs into igraph DiGraphs.

Handles the standard JSON format with top-level keys: metadata, qParams, nodes, links.
Node types: "cross layer transcoder", "mlp reconstruction error", "embedding", "logit".
Error nodes are excluded by default. Edges can be thresholded by absolute weight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import igraph as ig


# Feature types in the circuit-tracer / Neuronpedia JSON schema
FEATURE_TYPE_TRANSCODER = "cross layer transcoder"
FEATURE_TYPE_ERROR = "mlp reconstruction error"
FEATURE_TYPE_EMBEDDING = "embedding"
FEATURE_TYPE_LOGIT = "logit"

# Node types to exclude from motif analysis by default
DEFAULT_EXCLUDE_TYPES = frozenset({FEATURE_TYPE_ERROR})


def load_attribution_graph(
    json_path: str | Path,
    weight_threshold: float = 0.0,
    exclude_node_types: frozenset[str] | None = None,
    include_metadata: bool = True,
) -> ig.Graph:
    """Load a circuit-tracer/Neuronpedia JSON attribution graph into an igraph DiGraph.

    Args:
        json_path: Path to the JSON file.
        weight_threshold: Minimum absolute edge weight to include. Edges with
            |weight| < threshold are dropped.
        exclude_node_types: Set of feature_type strings to exclude. Defaults to
            excluding error nodes only.
        include_metadata: Whether to store graph-level metadata as graph attributes.

    Returns:
        A directed igraph.Graph with node and edge attributes.
    """
    if exclude_node_types is None:
        exclude_node_types = DEFAULT_EXCLUDE_TYPES

    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    return parse_attribution_graph(
        data,
        weight_threshold=weight_threshold,
        exclude_node_types=exclude_node_types,
        include_metadata=include_metadata,
        source_path=str(json_path),
    )


def parse_attribution_graph(
    data: dict[str, Any],
    weight_threshold: float = 0.0,
    exclude_node_types: frozenset[str] | None = None,
    include_metadata: bool = True,
    source_path: str | None = None,
) -> ig.Graph:
    """Parse an attribution graph dict into an igraph DiGraph.

    Args:
        data: Parsed JSON dict with keys: metadata, qParams, nodes, links.
        weight_threshold: Minimum absolute edge weight to include.
        exclude_node_types: Set of feature_type strings to exclude.
        include_metadata: Whether to store graph-level metadata as graph attributes.
        source_path: Optional source file path for metadata.

    Returns:
        A directed igraph.Graph with node and edge attributes.
    """
    if exclude_node_types is None:
        exclude_node_types = DEFAULT_EXCLUDE_TYPES

    g = ig.Graph(directed=True)

    # Parse metadata
    metadata = data.get("metadata", {})
    if include_metadata:
        g["prompt"] = metadata.get("prompt", "")
        g["prompt_tokens"] = metadata.get("prompt_tokens", [])
        g["model"] = metadata.get("scan", "")
        g["slug"] = metadata.get("slug", "")
        g["node_threshold"] = metadata.get("node_threshold", None)
        g["schema_version"] = metadata.get("schema_version", 1)
        if source_path:
            g["source_path"] = source_path

    # Add nodes, tracking which node_ids we keep
    kept_node_ids: set[str] = set()
    node_id_to_idx: dict[str, int] = {}

    for node in data.get("nodes", []):
        feature_type = node.get("feature_type", "")
        if feature_type in exclude_node_types:
            continue

        node_id = node["node_id"]
        kept_node_ids.add(node_id)

        layer = _parse_layer(node.get("layer", ""))
        idx = g.vcount()
        node_id_to_idx[node_id] = idx

        g.add_vertex(
            name=node_id,
            layer=layer,
            ctx_idx=node.get("ctx_idx", -1),
            feature=node.get("feature", None),
            feature_type=feature_type,
            clerp=node.get("clerp", ""),
            activation=node.get("activation", None),
            influence=node.get("influence", None),
            is_target_logit=node.get("is_target_logit", False),
            token_prob=node.get("token_prob", 0.0),
        )

    # Add edges
    for link in data.get("links", []):
        source_id = link["source"]
        target_id = link["target"]

        if source_id not in kept_node_ids or target_id not in kept_node_ids:
            continue

        weight = link.get("weight", 0.0)
        if weight is None:
            weight = 0.0

        abs_weight = abs(weight)
        if abs_weight < weight_threshold:
            continue

        sign = "excitatory" if weight >= 0 else "inhibitory"

        g.add_edge(
            node_id_to_idx[source_id],
            node_id_to_idx[target_id],
            weight=abs_weight,
            raw_weight=weight,
            sign=sign,
        )

    return g


def _parse_layer(layer_val: Any) -> int:
    """Parse the layer field which can be 'E' (embedding), a numeric string, or int.

    Args:
        layer_val: Raw layer value from JSON.

    Returns:
        Integer layer index. -1 for embedding nodes, -2 for unparseable values.
    """
    if isinstance(layer_val, int):
        return layer_val
    if isinstance(layer_val, str):
        if layer_val.upper() == "E":
            return -1
        try:
            return int(layer_val)
        except ValueError:
            return -2
    return -2


def graph_summary(g: ig.Graph) -> dict[str, Any]:
    """Return a summary dict of graph properties useful for sanity checking.

    Args:
        g: An igraph DiGraph loaded from an attribution graph.

    Returns:
        Dict with node count, edge count, density, degree stats, and node type breakdown.
    """
    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = g.density() if n_nodes > 1 else 0.0

    in_degrees = g.indegree()
    out_degrees = g.outdegree()

    # Node type breakdown
    type_counts: dict[str, int] = {}
    if "feature_type" in g.vs.attributes():
        for ft in g.vs["feature_type"]:
            type_counts[ft] = type_counts.get(ft, 0) + 1

    # Layer distribution
    layer_counts: dict[int, int] = {}
    if "layer" in g.vs.attributes():
        for lyr in g.vs["layer"]:
            layer_counts[lyr] = layer_counts.get(lyr, 0) + 1

    summary = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "mean_in_degree": sum(in_degrees) / n_nodes if n_nodes > 0 else 0,
        "max_in_degree": max(in_degrees) if in_degrees else 0,
        "mean_out_degree": sum(out_degrees) / n_nodes if n_nodes > 0 else 0,
        "max_out_degree": max(out_degrees) if out_degrees else 0,
        "node_type_counts": type_counts,
        "layer_counts": layer_counts,
        "prompt": g["prompt"] if "prompt" in g.attributes() else "",
        "model": g["model"] if "model" in g.attributes() else "",
    }
    return summary


def load_graphs_from_directory(
    directory: str | Path,
    weight_threshold: float = 0.0,
    exclude_node_types: frozenset[str] | None = None,
) -> list[ig.Graph]:
    """Load all JSON attribution graphs from a directory.

    Args:
        directory: Path to directory containing .json graph files.
        weight_threshold: Minimum absolute edge weight to include.
        exclude_node_types: Set of feature_type strings to exclude.

    Returns:
        List of igraph DiGraphs.
    """
    directory = Path(directory)
    graphs = []
    for json_file in sorted(directory.glob("*.json")):
        try:
            g = load_attribution_graph(
                json_file,
                weight_threshold=weight_threshold,
                exclude_node_types=exclude_node_types,
            )
            graphs.append(g)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    return graphs
