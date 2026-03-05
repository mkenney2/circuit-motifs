"""Unrolled motif cascade analysis for attribution graphs.

Shows how sign-aware unrolled motif instances chain together from
embedding inputs to the output logit, forming organized processing
cascades. Produces two figures per graph:

  1. Cascade schematic: abstract diagram of chained motif instances
  2. Graph overlay: real attribution graph with cascade highlighted

Usage:
    # Single graph (default: Dallas multihop)
    python scripts/unrolled_cascade.py

    # All 9 task categories
    python scripts/unrolled_cascade.py --all-categories
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_loader import load_attribution_graph
from src.unrolled_census import find_unrolled_instances, UnrolledMotifInstance
from src.unrolled_motifs import (
    CATALOG,
    build_catalog,
    get_effective_layer,
)
from src.visualization import _compute_neuronpedia_layout, _igraph_to_networkx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_PATH = PROJECT_ROOT / "data" / "raw" / "multihop" / "capital-state-dallas.json"
FIGURES_DIR = PROJECT_ROOT / "figures"
WEIGHT_THRESHOLD = 1.0

# Representative graph for each task category
CATEGORY_GRAPHS: dict[str, str] = {
    "arithmetic": "arithmetic/count-by-sevens-plt-clean.json",
    "code": "code/str-indexing-pos-0-plt-clean.json",
    "creative": "creative/rabbit-poem.json",
    "factual_recall": "factual_recall/ndag-plt-clean.json",
    "multihop": "multihop/capital-state-dallas.json",
    "multilingual": "multilingual/opposite_of_petit.json",
    "reasoning": "reasoning/medical-diagnosis.json",
    "safety": "safety/bomb-baseline.json",
    "uncategorized": "uncategorized/pmb-29-plt.json",
}

# Motif types to include in cascade analysis (skip 2-node templates)
CASCADE_MOTIF_TYPES = [
    "coherent_ffl",
    "incoherent_ffl",
    "feedforward_amplification",
    "feedforward_damping",
    "cross_chain_inhibition",
]

# Colors per motif type
MOTIF_COLORS = {
    "coherent_ffl": "#2ca02c",           # green
    "incoherent_ffl": "#d62728",         # red
    "feedforward_amplification": "#1f77b4",  # blue
    "feedforward_damping": "#ff7f0e",    # orange
    "cross_chain_inhibition": "#9467bd", # purple
}

MOTIF_SHORT_NAMES = {
    "coherent_ffl": "Coherent FFL",
    "incoherent_ffl": "Incoherent FFL",
    "feedforward_amplification": "FF Amplification",
    "feedforward_damping": "FF Damping",
    "cross_chain_inhibition": "Cross-chain Inhib.",
}

# Cascade tier boundaries (effective layer index)
TIER_EARLY = (-1, 4)    # embedding through L4
TIER_MIDDLE = (5, 12)   # L5 through L12
TIER_LATE = (13, 99)    # L13+

TIER_NAMES = {
    "early": "Early (Emb\u2013L4)",
    "middle": "Middle (L5\u2013L12)",
    "late": "Late (L13+)",
}

TIER_COLORS = {
    "early": "#d1e5f0",
    "middle": "#d9f0d3",
    "late": "#e8d5f0",
}


# ---------------------------------------------------------------------------
# Step 1 & 2: Load graph and enumerate instances
# ---------------------------------------------------------------------------

def load_and_enumerate(
    graph_path: Path = GRAPH_PATH,
    weight_threshold: float = WEIGHT_THRESHOLD,
) -> tuple[ig.Graph, dict[str, list[UnrolledMotifInstance]]]:
    """Load a graph and find all unrolled motif instances."""
    g = load_attribution_graph(str(graph_path), weight_threshold=weight_threshold)
    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

    all_instances: dict[str, list[UnrolledMotifInstance]] = {}
    for motif_name in CASCADE_MOTIF_TYPES:
        template = CATALOG[motif_name]
        instances = find_unrolled_instances(
            g, template, weight_threshold=weight_threshold,
        )
        all_instances[motif_name] = instances
        print(f"  {motif_name}: {len(instances)} instances")

    return g, all_instances


# ---------------------------------------------------------------------------
# Step 3: Annotate cascade position
# ---------------------------------------------------------------------------

def classify_tier(layer: int) -> str:
    """Classify an effective layer into a cascade tier."""
    if layer <= TIER_EARLY[1]:
        return "early"
    elif layer <= TIER_MIDDLE[1]:
        return "middle"
    else:
        return "late"


def annotate_instance(
    g: ig.Graph,
    inst: UnrolledMotifInstance,
) -> dict:
    """Compute cascade metadata for an instance."""
    layers = inst.layers
    min_layer = min(layers)
    max_layer = max(layers)

    has_ft = "feature_type" in g.vs.attributes()
    touches_embedding = any(
        g.vs[nid]["feature_type"] == "embedding"
        for nid in inst.node_indices
    ) if has_ft else False

    # "Touches output" = contains a logit node OR reaches the top
    # feature layers (within 3 layers of the highest non-logit layer)
    touches_logit = any(
        g.vs[nid]["feature_type"] == "logit"
        for nid in inst.node_indices
    ) if has_ft else False

    non_logit_layers = [
        get_effective_layer(g, v) for v in range(g.vcount())
        if not has_ft or g.vs[v]["feature_type"] != "logit"
    ]
    graph_max_feature_layer = max(non_logit_layers) if non_logit_layers else 0
    near_output = max_layer >= graph_max_feature_layer - 2
    touches_output = touches_logit or near_output

    return {
        "instance": inst,
        "min_layer": min_layer,
        "max_layer": max_layer,
        "tier": classify_tier((min_layer + max_layer) // 2),
        "touches_embedding": touches_embedding,
        "touches_output": touches_output or near_output,
        "node_set": frozenset(inst.node_indices),
    }


# ---------------------------------------------------------------------------
# Step 4: Build cascade graph
# ---------------------------------------------------------------------------

def _mean_layer(ann: dict) -> float:
    """Mean effective layer of an instance's nodes."""
    return sum(ann["instance"].layers) / len(ann["instance"].layers)


def build_cascade_graph(
    g: ig.Graph,
    annotated: list[dict],
) -> tuple[ig.Graph, list[dict]]:
    """Build a DAG where nodes = motif instances and edges = shared graph nodes.

    Two instances i -> j are connected if:
      (a) they share at least one graph node, AND
      (b) instance i's mean layer < instance j's mean layer (forward flow)

    We also add edges between non-overlapping instances that are connected
    by a direct graph edge (a node in instance i feeds a node in instance j).

    Returns:
        cascade_g: Directed igraph.Graph of instance nodes.
        annotated: The input list (ordered by instance index).
    """
    n = len(annotated)
    cascade_g = ig.Graph(n, directed=True)

    # Store metadata on cascade graph nodes
    for i, ann in enumerate(annotated):
        inst = ann["instance"]
        cascade_g.vs[i]["template_name"] = inst.template_name
        cascade_g.vs[i]["min_layer"] = ann["min_layer"]
        cascade_g.vs[i]["max_layer"] = ann["max_layer"]
        cascade_g.vs[i]["tier"] = ann["tier"]
        cascade_g.vs[i]["touches_embedding"] = ann["touches_embedding"]
        cascade_g.vs[i]["touches_output"] = ann["touches_output"]
        cascade_g.vs[i]["total_weight"] = inst.total_weight

    # Precompute mean layers for ordering
    mean_layers = [_mean_layer(ann) for ann in annotated]

    # Build edges: two connection types
    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Must flow forward: mean layer of i < mean layer of j
            if mean_layers[i] >= mean_layers[j]:
                continue

            nodes_i = annotated[i]["node_set"]
            nodes_j = annotated[j]["node_set"]

            # Type 1: shared graph nodes (instances overlap at a handoff node)
            shared = nodes_i & nodes_j
            if shared:
                weight = len(shared) * 2.0 + 0.5 * (
                    annotated[i]["instance"].total_weight +
                    annotated[j]["instance"].total_weight
                )
                cascade_g.add_edge(i, j, weight=weight,
                                   shared_nodes=len(shared), link_type="shared")
                continue

            # Type 2: direct graph edge from a node in i to a node in j
            for ni in nodes_i:
                for nj in nodes_j:
                    eid = g.get_eid(ni, nj, error=False)
                    if eid >= 0:
                        w = g.es[eid]["weight"] if "weight" in g.es.attributes() else 1.0
                        weight = w + 0.25 * (
                            annotated[i]["instance"].total_weight +
                            annotated[j]["instance"].total_weight
                        )
                        cascade_g.add_edge(i, j, weight=weight,
                                           shared_nodes=0, link_type="edge")
                        break
                else:
                    continue
                break

    return cascade_g, annotated


# ---------------------------------------------------------------------------
# Step 5: Extract main cascade (heaviest path via DP)
# ---------------------------------------------------------------------------

def _dp_heaviest_path(
    cascade_g: ig.Graph,
    annotated: list[dict],
    start_filter=None,
    end_filter=None,
) -> list[int]:
    """Find the heaviest path using DP on the cascade DAG.

    Args:
        cascade_g: The cascade DAG.
        annotated: List of instance annotations.
        start_filter: If set, only these indices can be path starts (bonus weight).
        end_filter: If set, only these indices can be path ends.

    Returns:
        List of instance indices along the heaviest path.
    """
    n = cascade_g.vcount()
    if n == 0:
        return []

    topo_order = cascade_g.topological_sorting(mode="out")

    best_weight = np.full(n, -np.inf)
    predecessor = np.full(n, -1, dtype=int)

    # Initialize path starts
    for i in range(n):
        if start_filter is not None and i not in start_filter:
            # Non-start nodes can only be reached via edges, not as path starts
            best_weight[i] = -np.inf
        else:
            best_weight[i] = annotated[i]["instance"].total_weight

    # Forward pass
    for u in topo_order:
        for eid in cascade_g.incident(u, mode="out"):
            e = cascade_g.es[eid]
            v = e.target
            new_weight = best_weight[u] + e["weight"] + annotated[v]["instance"].total_weight
            if new_weight > best_weight[v]:
                best_weight[v] = new_weight
                predecessor[v] = u

    # Choose endpoint: prefer end_filter or output nodes, but only if reachable
    if end_filter:
        candidates = [i for i in end_filter if best_weight[i] > -np.inf]
    else:
        candidates = [i for i in range(n)
                      if annotated[i]["touches_output"] and best_weight[i] > -np.inf]
    if not candidates:
        candidates = [i for i in range(n)
                      if annotated[i]["tier"] == "late" and best_weight[i] > -np.inf]
    if not candidates:
        # Fall back to any reachable node
        candidates = [i for i in range(n) if best_weight[i] > -np.inf]
    if not candidates:
        return []

    best_end = max(candidates, key=lambda i: best_weight[i])

    path = []
    node = best_end
    while node >= 0:
        path.append(node)
        node = predecessor[node]
    path.reverse()
    return path


def find_heaviest_path(
    cascade_g: ig.Graph,
    annotated: list[dict],
) -> list[int]:
    """Find the heaviest path, preferring embedding-to-output coverage.

    If a direct connected path from embedding to output exists, use it.
    Otherwise, find the best early segment (from embedding) and best late
    segment (to output) and concatenate them with a gap marker (-1).
    """
    n = cascade_g.vcount()
    if n == 0:
        return []

    emb_set = {i for i in range(n) if annotated[i]["touches_embedding"]}
    out_set = {i for i in range(n) if annotated[i]["touches_output"]}

    # Try direct embedding-to-output path
    if emb_set and out_set:
        path = _dp_heaviest_path(cascade_g, annotated,
                                  start_filter=emb_set, end_filter=out_set)
        if path and len(path) >= 2 and path[0] in emb_set and path[-1] in out_set:
            return path

    # No direct path — find best early + best late segments separately
    combined = []

    if emb_set:
        early_path = _dp_heaviest_path(cascade_g, annotated,
                                        start_filter=emb_set, end_filter=None)
        if early_path and early_path[0] in emb_set:
            combined.extend(early_path)

    # Add gap marker
    if combined:
        combined.append(-1)

    if out_set:
        late_path = _dp_heaviest_path(cascade_g, annotated,
                                       start_filter=None, end_filter=out_set)
        if late_path:
            combined.extend(late_path)

    if combined:
        return combined

    # Ultimate fallback
    return _dp_heaviest_path(cascade_g, annotated)


def find_top_paths(
    cascade_g: ig.Graph,
    annotated: list[dict],
    n_paths: int = 3,
) -> list[list[int]]:
    """Find the top-N heaviest paths by iteratively penalizing used edges."""
    paths = []
    # Work on a copy for penalization
    weights_orig = list(cascade_g.es["weight"]) if cascade_g.ecount() > 0 else []

    for _ in range(n_paths):
        path = find_heaviest_path(cascade_g, annotated)
        if not path or len(path) < 2:
            break
        paths.append(path)

        # Penalize edges used in this path (skip gap markers)
        for k in range(len(path) - 1):
            if path[k] < 0 or path[k + 1] < 0:
                continue
            eid = cascade_g.get_eid(path[k], path[k + 1], error=False)
            if eid >= 0:
                cascade_g.es[eid]["weight"] *= 0.1

    # Restore original weights
    if weights_orig:
        for i, w in enumerate(weights_orig):
            cascade_g.es[i]["weight"] = w

    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_dark(hex_color: str) -> bool:
    """Check if a hex color is dark (for choosing white vs black text)."""
    hex_color = hex_color.lstrip("#")
    r, gg, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (0.299 * r + 0.587 * gg + 0.114 * b) < 140


def _get_clerp_label(g: ig.Graph, node_idx: int, max_len: int = 25) -> str:
    """Get a short clerp label for a node."""
    has_clerp = "clerp" in g.vs.attributes()
    has_ft = "feature_type" in g.vs.attributes()
    if not has_clerp:
        return f"node {node_idx}"
    clerp = g.vs[node_idx]["clerp"] or ""
    if not clerp and has_ft:
        ft = g.vs[node_idx]["feature_type"]
        if ft == "embedding":
            clerp = "[embedding]"
        elif ft == "logit":
            clerp = "[logit]"
    if not clerp:
        clerp = f"[node {node_idx}]"
    if len(clerp) > max_len:
        clerp = clerp[:max_len - 1] + "\u2026"
    return clerp


def _get_layer_label(g: ig.Graph, node_idx: int) -> str:
    """Get a short layer label for a node."""
    layer = g.vs[node_idx]["layer"] if "layer" in g.vs.attributes() else 0
    ft = g.vs[node_idx]["feature_type"] if "feature_type" in g.vs.attributes() else ""
    if ft == "embedding":
        return "Emb"
    if ft == "logit":
        return "Lgt"
    return f"L{layer}"


# ---------------------------------------------------------------------------
# Figure 1: Cascade Schematic
# ---------------------------------------------------------------------------

def generate_schematic(
    g: ig.Graph,
    annotated: list[dict],
    main_path: list[int],
    alt_paths: list[list[int]],
    output_path: Path | None = None,
    weight_threshold: float = WEIGHT_THRESHOLD,
):
    """Generate the cascade schematic figure."""
    if output_path is None:
        output_path = FIGURES_DIR / "fig_unrolled_cascade_schematic.png"

    if not main_path:
        print("  WARNING: No cascade path found, skipping schematic.")
        return

    # Filter out gap markers (-1), track where gaps are
    real_indices = [i for i in main_path if i >= 0]
    gap_positions = set()
    for k, idx in enumerate(main_path):
        if idx == -1:
            gap_positions.add(k)

    # Build step list: pairs of (step_display_index, annotation)
    path_instances = [annotated[i] for i in real_indices]

    # Track which display steps have a gap before them
    gaps_before_step: set[int] = set()
    display_idx = 0
    for k, idx in enumerate(main_path):
        if idx == -1:
            gaps_before_step.add(display_idx)
        else:
            display_idx += 1

    # Layout: vertical, bottom = early, top = late
    n_steps = len(path_instances)
    n_gaps = len(gaps_before_step)
    fig, ax = plt.subplots(
        figsize=(16, max(12, 2.5 * (n_steps + n_gaps))),
        facecolor="white",
    )

    # Y range per step — account for gap spacing
    step_height = 2.5
    gap_height = 1.5
    total_height = n_steps * step_height + n_gaps * gap_height + 2.0
    ax.set_xlim(-1, 15)
    ax.set_ylim(-0.5, total_height + 0.5)
    ax.set_aspect("auto")
    ax.axis("off")

    # Compute y-center for each step, inserting gap spacing
    step_y_centers: list[float] = []
    y_offset = 0.0
    for step_idx in range(n_steps):
        if step_idx in gaps_before_step:
            y_offset += gap_height
        y_center = step_idx * step_height + step_height / 2 + y_offset
        step_y_centers.append(y_center)

    # Draw tier background bands
    tier_instances = defaultdict(list)
    for step_idx, ann in enumerate(path_instances):
        tier_instances[ann["tier"]].append(step_idx)

    for tier_name, step_indices in tier_instances.items():
        y_min = step_y_centers[min(step_indices)] - step_height / 2 - 0.3
        y_max = step_y_centers[max(step_indices)] + step_height / 2 + 0.3
        ax.axhspan(y_min, y_max, facecolor=TIER_COLORS[tier_name],
                    alpha=0.20, zorder=0)
        ax.text(-0.5, y_max - 0.1, TIER_NAMES[tier_name],
                fontsize=11, fontweight="bold",
                color="#555555", va="top", ha="left", zorder=8)

    # Draw gap markers
    for step_idx in gaps_before_step:
        if step_idx > 0 and step_idx <= n_steps:
            gap_y = (step_y_centers[step_idx - 1] + step_y_centers[step_idx]) / 2
        elif step_idx == 0:
            gap_y = step_y_centers[0] - step_height / 2 - gap_height / 2
        else:
            continue
        ax.text(7.0, gap_y, "\u22ee  structural gap  \u22ee",
                fontsize=11, color="#999999", fontstyle="italic",
                ha="center", va="center", zorder=3)

    # Draw each motif instance as a colored box
    for step_idx, ann in enumerate(path_instances):
        inst = ann["instance"]
        y_center = step_y_centers[step_idx]
        color = MOTIF_COLORS.get(inst.template_name, "#888888")
        short_name = MOTIF_SHORT_NAMES.get(inst.template_name, inst.template_name)

        # Collect clerp labels for nodes in this instance
        node_labels = []
        for nid in inst.node_indices:
            clerp = _get_clerp_label(g, nid, max_len=30)
            layer = _get_layer_label(g, nid)
            role = inst.node_roles.get(nid, "")
            node_labels.append(f"{role}: {clerp} ({layer})")

        label_text = "\n".join(node_labels)

        # Main box
        box_width = 10.0
        box_height = step_height * 0.8
        box = FancyBboxPatch(
            (2.0, y_center - box_height / 2), box_width, box_height,
            boxstyle="round,pad=0.15",
            facecolor=color, edgecolor="black",
            linewidth=2.5, alpha=0.85, zorder=5,
        )
        ax.add_patch(box)

        # Motif type label (left side)
        text_color = "white" if _is_dark(color) else "#111111"
        ax.text(2.3, y_center + box_height / 2 - 0.25, short_name,
                fontsize=11, fontweight="bold", color=text_color,
                va="top", ha="left", zorder=6)

        # Node details (centered, smaller)
        ax.text(7.0, y_center - 0.1, label_text,
                fontsize=8, color=text_color,
                va="center", ha="center", zorder=6,
                linespacing=1.4)

        # Weight annotation (right side)
        ax.text(12.3, y_center, f"w={inst.total_weight:.1f}",
                fontsize=9, color="#555555", fontweight="bold",
                va="center", ha="left", zorder=6)

        # Layer range annotation
        ax.text(12.3, y_center - 0.4,
                f"L{ann['min_layer']}\u2013L{ann['max_layer']}",
                fontsize=8, color="#777777",
                va="center", ha="left", zorder=6)

    # Draw cascade arrows between consecutive steps (skip across gaps)
    for i in range(n_steps - 1):
        # Skip if there's a gap between step i and step i+1
        if (i + 1) in gaps_before_step:
            continue

        y_from = step_y_centers[i] + step_height * 0.4
        y_to = step_y_centers[i + 1] - step_height * 0.4

        # Count shared nodes
        ann_i = annotated[real_indices[i]]
        ann_j = annotated[real_indices[i + 1]]
        shared = ann_i["node_set"] & ann_j["node_set"]
        n_shared = len(shared)

        # Get clerp labels for shared nodes
        shared_labels = [_get_clerp_label(g, nid, max_len=20) for nid in shared]
        shared_text = ", ".join(shared_labels[:3])
        if len(shared_labels) > 3:
            shared_text += f" +{len(shared_labels) - 3}"

        arrow = FancyArrowPatch(
            (7.0, y_from), (7.0, y_to),
            arrowstyle="->,head_length=10,head_width=6",
            connectionstyle="arc3,rad=0",
            color="#333333", linewidth=2.0 + n_shared,
            zorder=4, alpha=0.8,
        )
        ax.add_patch(arrow)

        # Shared node label beside arrow
        ax.text(13.2, (y_from + y_to) / 2,
                f"shared: {shared_text}",
                fontsize=7, color="#666666", fontstyle="italic",
                va="center", ha="left", zorder=6)

    # Title
    prompt = g["prompt"] if "prompt" in g.attributes() else ""
    prompt_short = prompt if len(prompt) <= 55 else prompt[:52] + "..."
    fig.suptitle(
        f"Unrolled Motif Cascade\n\"{prompt_short}\"",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor=MOTIF_COLORS[mt], edgecolor="black",
                       linewidth=1.0, label=MOTIF_SHORT_NAMES[mt])
        for mt in CASCADE_MOTIF_TYPES
        if any(annotated[i]["instance"].template_name == mt for i in real_indices)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                  framealpha=0.92, edgecolor="#cccccc", fancybox=True,
                  bbox_to_anchor=(1.0, 1.0))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved schematic to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Graph overlay with cascade highlighted
# ---------------------------------------------------------------------------

def generate_graph_overlay(
    g: ig.Graph,
    annotated: list[dict],
    main_path: list[int],
    output_path: Path | None = None,
    weight_threshold: float = WEIGHT_THRESHOLD,
):
    """Generate the attribution graph with cascade instances highlighted."""
    if output_path is None:
        output_path = FIGURES_DIR / "fig_unrolled_cascade_graph.png"

    if not main_path:
        print("  WARNING: No cascade path found, skipping graph overlay.")
        return

    nxg = _igraph_to_networkx(g)
    pos, sorted_layers, layer_labels = _compute_neuronpedia_layout(g)

    has_ft = "feature_type" in g.vs.attributes()
    has_sign = "sign" in g.es.attributes() if g.ecount() > 0 else False
    has_weight = "weight" in g.es.attributes() if g.ecount() > 0 else False
    has_ctx = "ctx_idx" in g.vs.attributes()
    has_clerp = "clerp" in g.vs.attributes()

    # Collect cascade nodes and edges (skip gap markers)
    cascade_nodes: dict[int, dict] = {}  # graph_node -> annotation info
    cascade_edges: set[tuple[int, int]] = set()
    real_path = [i for i in main_path if i >= 0]

    for path_idx in real_path:
        ann = annotated[path_idx]
        inst = ann["instance"]
        color = MOTIF_COLORS.get(inst.template_name, "#888888")

        for nid in inst.node_indices:
            if nid not in cascade_nodes:
                cascade_nodes[nid] = {
                    "color": color,
                    "template": inst.template_name,
                    "count": 0,
                }
            cascade_nodes[nid]["count"] += 1

        # Add edges that belong to this instance
        template = CATALOG[inst.template_name]
        for tmpl_edge in template.edges:
            src_tmpl_idx = template.node_ids.index(tmpl_edge["src"])
            tgt_tmpl_idx = template.node_ids.index(tmpl_edge["tgt"])
            graph_src = inst.node_indices[src_tmpl_idx]
            graph_tgt = inst.node_indices[tgt_tmpl_idx]
            eid = g.get_eid(graph_src, graph_tgt, error=False)
            if eid >= 0:
                cascade_edges.add((graph_src, graph_tgt))

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    ax.set_facecolor("white")

    # --- Context edges (faint) ---
    all_edges = list(nxg.edges)
    non_cascade = [e for e in all_edges if e not in cascade_edges]
    if non_cascade:
        nx.draw_networkx_edges(
            nxg, pos, edgelist=non_cascade, alpha=0.15,
            edge_color="#aaaaaa", arrows=True, arrowsize=3,
            connectionstyle="arc3,rad=0.05", ax=ax, node_size=18,
        )

    # --- Context nodes (faint) ---
    emb_nodes = [v.index for v in g.vs if has_ft and v["feature_type"] == "embedding"]
    logit_nodes = [v.index for v in g.vs if has_ft and v["feature_type"] == "logit"]
    feature_nodes = [v.index for v in g.vs
                     if v.index not in set(emb_nodes + logit_nodes)]

    ctx_emb = [n for n in emb_nodes if n not in cascade_nodes]
    ctx_logit = [n for n in logit_nodes if n not in cascade_nodes]
    ctx_feature = [n for n in feature_nodes if n not in cascade_nodes]

    if ctx_emb:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_emb, node_size=25,
                               node_color="#b0d4e8", edgecolors="#6699bb",
                               linewidths=0.5, node_shape="s", alpha=0.4, ax=ax)
    if ctx_feature:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_feature, node_size=18,
                               node_color="#bbbbbb", edgecolors="#888888",
                               linewidths=0.4, node_shape="o", alpha=0.35, ax=ax)
    if ctx_logit:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_logit, node_size=30,
                               node_color="#a8ddb5", edgecolors="#4daf4a",
                               linewidths=0.5, node_shape="p", alpha=0.4, ax=ax)

    # --- Cascade edges (thick, colored) ---
    for u, v in cascade_edges:
        eid = g.get_eid(u, v, error=False)
        if eid < 0:
            continue

        # Color by the source node's motif type
        edge_color = cascade_nodes.get(u, {}).get("color", "#333333")
        ls = "-"
        if has_sign and g.es[eid]["sign"] == "inhibitory":
            ls = "--"
            edge_color = "#d62728"

        width = 2.0
        if has_weight:
            width = 1.5 + min(3.0, g.es[eid]["weight"] / 5.0)

        nx.draw_networkx_edges(
            nxg, pos, edgelist=[(u, v)], edge_color=edge_color,
            width=width, arrows=True, arrowsize=10, alpha=0.85,
            connectionstyle="arc3,rad=0.08", ax=ax, node_size=120,
            style=ls,
        )

    # --- Cascade nodes (colored, larger) ---
    for nid, info in cascade_nodes.items():
        ft = g.vs[nid]["feature_type"] if has_ft else ""
        shape = "s" if ft == "embedding" else ("p" if ft == "logit" else "o")
        n_count = info["count"]
        node_size = 120 + 40 * n_count
        lw = 2.5 if n_count > 1 else 1.5

        coll = nx.draw_networkx_nodes(
            nxg, pos, nodelist=[nid], node_size=node_size,
            node_color=info["color"], edgecolors="black", linewidths=lw,
            node_shape=shape, ax=ax,
        )
        if coll:
            coll.set_zorder(6)

    # --- Clerp labels on cascade nodes ---
    labeled_positions: list[tuple[float, float]] = []
    offset_options = [(22, 18), (-22, 18), (22, -18), (-22, -18),
                      (26, 0), (-26, 0), (0, 22), (0, -22)]

    for nid in cascade_nodes:
        if not has_clerp:
            continue
        clerp = g.vs[nid]["clerp"]
        if not clerp:
            continue

        clerp_short = clerp if len(clerp) <= 30 else clerp[:27] + "\u2026"
        layer_str = _get_layer_label(g, nid)
        display = f"{clerp_short}\n({layer_str})"

        x, y = pos[nid]
        color = cascade_nodes[nid]["color"]

        # Pick offset furthest from existing labels
        best_ox, best_oy = offset_options[0]
        best_dist = -1
        for ox, oy in offset_options:
            target_x, target_y = x + ox / 50, y + oy / 50
            if labeled_positions:
                min_d = min((target_x - lx) ** 2 + (target_y - ly) ** 2
                            for lx, ly in labeled_positions)
            else:
                min_d = 999
            if min_d > best_dist:
                best_dist = min_d
                best_ox, best_oy = ox, oy

        labeled_positions.append((x + best_ox / 50, y + best_oy / 50))
        ha = "left" if best_ox > 0 else ("right" if best_ox < 0 else "center")

        ax.annotate(
            display, xy=(x, y),
            xytext=(best_ox, best_oy), textcoords="offset points",
            fontsize=9, ha=ha,
            va="bottom" if best_oy > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=color, alpha=0.92, linewidth=1.2),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8),
            zorder=7,
        )

    # --- Axis setup ---
    if has_ctx:
        all_ctx = [v["ctx_idx"] for v in g.vs]
        min_ctx, max_ctx = min(all_ctx), max(all_ctx)
    else:
        min_ctx, max_ctx = 0, 0

    layer_y_positions = [i * 1.0 for i in range(len(sorted_layers))]
    ax.set_yticks(layer_y_positions)
    ax.set_yticklabels(layer_labels, fontsize=11, color="#333333",
                       fontfamily="monospace", fontweight="bold")
    ax.yaxis.set_ticks_position("left")

    prompt_tokens = g["prompt_tokens"] if "prompt_tokens" in g.attributes() else []
    if prompt_tokens:
        token_x_positions = list(range(min_ctx, max_ctx + 1))
        token_tick_labels = []
        for ctx in token_x_positions:
            if ctx < len(prompt_tokens):
                tok = prompt_tokens[ctx].replace("\u2191", "^")
                token_tick_labels.append(tok)
            else:
                token_tick_labels.append(f"[{ctx}]")
        ax.set_xticks([x * 1.0 for x in token_x_positions])
        ax.set_xticklabels(token_tick_labels, fontsize=11, rotation=45,
                           ha="right", color="#333333", fontstyle="italic")
        ax.xaxis.set_ticks_position("bottom")

    for y_val in layer_y_positions:
        ax.axhline(y=y_val, color="#eeeeee", linewidth=0.4, zorder=0)
    if has_ctx:
        for ctx in range(min_ctx, max_ctx + 1):
            ax.axvline(x=ctx * 1.0, color="#eeeeee", linewidth=0.4, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#cccccc")
    ax.spines["left"].set_color("#cccccc")
    ax.tick_params(axis="both", which="both", length=3, color="#cccccc")

    ax.set_xlim((min_ctx - 0.8), (max_ctx + 1.5))
    ax.set_ylim(-0.8, max(layer_y_positions) + 0.8)

    # --- Legend ---
    legend_handles = []
    seen_types = set()
    for path_idx in real_path:
        tname = annotated[path_idx]["instance"].template_name
        if tname not in seen_types:
            seen_types.add(tname)
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=MOTIF_COLORS.get(tname, "#888"),
                           markeredgecolor="black", markersize=8, linewidth=0,
                           label=MOTIF_SHORT_NAMES.get(tname, tname))
            )
    legend_handles.append(
        plt.Line2D([0], [0], color="#d62728", linewidth=2.5, linestyle="--",
                   label="Inhibitory edge")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10,
              framealpha=0.92, edgecolor="#cccccc", fancybox=True)

    prompt = g["prompt"] if "prompt" in g.attributes() else ""
    prompt_display = prompt if len(prompt) <= 55 else prompt[:52] + "..."
    ax.set_title(
        f"Unrolled Motif Cascade on Attribution Graph\n"
        f"\"{prompt_display}\"    "
        f"(weight threshold = {weight_threshold})",
        fontsize=14, fontweight="bold", loc="left", pad=10,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved graph overlay to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Single-category cascade analysis
# ---------------------------------------------------------------------------

def run_cascade_analysis(
    graph_path: Path,
    category: str,
    output_prefix: str | None = None,
    weight_threshold: float = WEIGHT_THRESHOLD,
) -> dict:
    """Run full cascade analysis on a single graph.

    Args:
        graph_path: Path to the attribution graph JSON.
        category: Task category name (used in output filenames and summary).
        output_prefix: Prefix for output figures. If None, uses default naming.
        weight_threshold: Minimum edge weight to include.

    Returns:
        Summary dict with cascade statistics for this graph.
    """
    print(f"\n{'=' * 60}")
    print(f"Cascade Analysis: {category}")
    print(f"{'=' * 60}")

    # Determine output paths
    if output_prefix:
        schematic_path = FIGURES_DIR / f"{output_prefix}_schematic.png"
        graph_path_out = FIGURES_DIR / f"{output_prefix}_graph.png"
    else:
        schematic_path = FIGURES_DIR / "fig_unrolled_cascade_schematic.png"
        graph_path_out = FIGURES_DIR / "fig_unrolled_cascade_graph.png"

    # Step 1-2: Load and enumerate
    print("\n[1/6] Loading graph and enumerating instances...")
    g, all_instances = load_and_enumerate(graph_path, weight_threshold)

    # Flatten all instances into a single list with annotations
    print("\n[2/6] Annotating cascade positions...")
    all_annotated: list[dict] = []
    for motif_name, instances in all_instances.items():
        for inst in instances:
            ann = annotate_instance(g, inst)
            all_annotated.append(ann)

    total_instances = len(all_annotated)
    print(f"  Total annotated instances: {total_instances}")
    tier_counts = defaultdict(int)
    for ann in all_annotated:
        tier_counts[ann["tier"]] += 1
    for tier, count in sorted(tier_counts.items()):
        print(f"    {tier}: {count}")

    emb_count = sum(1 for a in all_annotated if a["touches_embedding"])
    out_count = sum(1 for a in all_annotated if a["touches_output"])
    print(f"  Touches embedding: {emb_count}")
    print(f"  Touches output: {out_count}")

    # Instance counts per motif type
    instance_counts = {mt: len(insts) for mt, insts in all_instances.items()}

    # Step 3: Build cascade graph
    print("\n[3/6] Building cascade graph...")
    cascade_g, annotated = build_cascade_graph(g, all_annotated)
    print(f"  Cascade graph: {cascade_g.vcount()} nodes, {cascade_g.ecount()} edges")

    # Connected components analysis
    components = cascade_g.connected_components(mode="weak")
    print(f"  Connected components: {len(components)}")
    comp_sizes = sorted([len(c) for c in components], reverse=True)
    print(f"  Largest components: {comp_sizes[:5]}")

    # Step 4: Find main cascade path
    print("\n[4/6] Finding heaviest cascade path...")
    main_path = find_heaviest_path(cascade_g, annotated)
    real_path = [i for i in main_path if i >= 0]
    has_gap = -1 in main_path
    path_length = len(real_path)
    print(f"  Main path length: {path_length} steps" +
          (" [with gap]" if has_gap else ""))

    # Compute path statistics
    path_types: list[str] = []
    total_path_weight = 0.0
    layer_min = 999
    layer_max = -1
    if main_path:
        for i, idx in enumerate(main_path):
            if idx == -1:
                print(f"    [{i}] --- structural gap ---")
                continue
            ann = annotated[idx]
            inst = ann["instance"]
            path_types.append(inst.template_name)
            total_path_weight += inst.total_weight
            layer_min = min(layer_min, ann["min_layer"])
            layer_max = max(layer_max, ann["max_layer"])
            name = MOTIF_SHORT_NAMES.get(inst.template_name, inst.template_name)
            labels = [_get_clerp_label(g, nid, 25) for nid in inst.node_indices]
            print(f"    [{i}] {name} (L{ann['min_layer']}-L{ann['max_layer']}, "
                  f"w={inst.total_weight:.1f}): {', '.join(labels)}")

    # Dominant motif type in path
    type_counts = Counter(path_types)
    dominant_type = type_counts.most_common(1)[0][0] if type_counts else ""

    # Find alternative paths
    print("\n[5/6] Finding alternative paths...")
    alt_paths = find_top_paths(cascade_g, annotated, n_paths=3)
    for pi, path in enumerate(alt_paths):
        real = [i for i in path if i >= 0]
        tiers = [annotated[i]["tier"] for i in real]
        types = [annotated[i]["instance"].template_name for i in real]
        gap = -1 in path
        gap_str = " [with gap]" if gap else ""
        print(f"  Path {pi + 1}: {len(real)} steps{gap_str}, "
              f"tiers={tiers}, types={[MOTIF_SHORT_NAMES.get(t, t) for t in types]}")

    # Step 5-6: Generate figures
    print("\n[6/6] Generating figures...")
    generate_schematic(g, annotated, main_path, alt_paths,
                       output_path=schematic_path,
                       weight_threshold=weight_threshold)
    generate_graph_overlay(g, annotated, main_path,
                           output_path=graph_path_out,
                           weight_threshold=weight_threshold)

    # Build summary
    prompt = g["prompt"] if "prompt" in g.attributes() else ""
    summary = {
        "category": category,
        "graph_file": str(graph_path.name),
        "prompt": prompt,
        "n_nodes": g.vcount(),
        "n_edges": g.ecount(),
        "total_instances": total_instances,
        "instance_counts": instance_counts,
        "path_length": path_length,
        "has_gap": has_gap,
        "path_types": path_types,
        "dominant_type": dominant_type,
        "type_composition": dict(type_counts),
        "total_path_weight": total_path_weight,
        "layer_range": (layer_min, layer_max) if path_types else (0, 0),
        "n_components": len(components),
        "largest_component": comp_sizes[0] if comp_sizes else 0,
    }

    print(f"\nDone with {category}!")
    return summary


# ---------------------------------------------------------------------------
# Comparison figure (all categories)
# ---------------------------------------------------------------------------

def generate_comparison_figure(
    summaries: list[dict],
    output_path: Path | None = None,
):
    """Generate a summary comparison figure across all categories."""
    if output_path is None:
        output_path = FIGURES_DIR / "fig_unrolled_cascade_comparison.png"

    if not summaries:
        print("  WARNING: No summaries to compare.")
        return

    # Sort by path length (descending) for visual clarity
    summaries = sorted(summaries, key=lambda s: s["path_length"], reverse=True)

    n_cats = len(summaries)
    categories = [s["category"] for s in summaries]

    fig, axes = plt.subplots(1, 3, figsize=(20, max(6, 0.8 * n_cats)),
                              facecolor="white", gridspec_kw={"wspace": 0.35})

    # --- Panel 1: Cascade path length ---
    ax1 = axes[0]
    bar_colors = []
    for s in summaries:
        if s["dominant_type"]:
            bar_colors.append(MOTIF_COLORS.get(s["dominant_type"], "#888888"))
        else:
            bar_colors.append("#cccccc")

    y_pos = np.arange(n_cats)
    bars = ax1.barh(y_pos, [s["path_length"] for s in summaries],
                    color=bar_colors, edgecolor="black", linewidth=0.8, alpha=0.85)

    # Mark gaps with a hatched overlay
    for i, s in enumerate(summaries):
        if s["has_gap"]:
            ax1.barh(i, s["path_length"], color="none",
                     edgecolor="#d62728", linewidth=2.0, linestyle="--")
            ax1.text(s["path_length"] + 0.15, i, "gap",
                     fontsize=8, color="#d62728", va="center", fontstyle="italic")

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(categories, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cascade Path Length (steps)", fontsize=11)
    ax1.set_title("Path Length", fontsize=13, fontweight="bold")
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- Panel 2: Motif type composition (stacked bar) ---
    ax2 = axes[1]
    all_types = CASCADE_MOTIF_TYPES
    # Build stacked data
    left_offsets = np.zeros(n_cats)
    for mt in all_types:
        counts = []
        for s in summaries:
            counts.append(s["type_composition"].get(mt, 0))
        counts_arr = np.array(counts, dtype=float)
        if counts_arr.sum() > 0:
            ax2.barh(y_pos, counts_arr, left=left_offsets,
                     color=MOTIF_COLORS[mt], edgecolor="white", linewidth=0.5,
                     label=MOTIF_SHORT_NAMES[mt], alpha=0.85)
            left_offsets += counts_arr

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([""] * n_cats)  # shared y-axis labels from panel 1
    ax2.set_xlabel("Motif Instances in Path", fontsize=11)
    ax2.set_title("Path Composition", fontsize=13, fontweight="bold")
    ax2.invert_yaxis()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=8, loc="lower right", framealpha=0.9)

    # --- Panel 3: Total path weight + layer range ---
    ax3 = axes[2]
    weights = [s["total_path_weight"] for s in summaries]
    layer_spans = [s["layer_range"][1] - s["layer_range"][0]
                   for s in summaries]

    # Dual metric: weight as bars, layer span as scatter overlay
    bars3 = ax3.barh(y_pos, weights, color="#4daf4a", edgecolor="black",
                     linewidth=0.8, alpha=0.6, label="Total weight")

    # Add layer range annotations
    for i, s in enumerate(summaries):
        lr = s["layer_range"]
        ax3.text(weights[i] + 0.3, i,
                 f"L{lr[0]}\u2013L{lr[1]}",
                 fontsize=9, color="#555555", va="center")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([""] * n_cats)
    ax3.set_xlabel("Total Path Weight", fontsize=11)
    ax3.set_title("Weight & Layer Range", fontsize=13, fontweight="bold")
    ax3.invert_yaxis()
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle(
        "Unrolled Motif Cascade Comparison Across Task Categories",
        fontsize=15, fontweight="bold", y=1.02,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved comparison figure to {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Unrolled motif cascade analysis for attribution graphs."
    )
    parser.add_argument(
        "--all-categories", action="store_true",
        help="Run cascade analysis on one representative graph from each "
             "of the 9 task categories and generate a comparison figure.",
    )
    parser.add_argument(
        "--weight-threshold", type=float, default=WEIGHT_THRESHOLD,
        help=f"Minimum edge weight to include (default: {WEIGHT_THRESHOLD}).",
    )
    args = parser.parse_args()

    if args.all_categories:
        # Multi-graph mode: loop over all 9 categories
        data_dir = PROJECT_ROOT / "data" / "raw"
        summaries: list[dict] = []

        for category, rel_path in sorted(CATEGORY_GRAPHS.items()):
            graph_path = data_dir / rel_path
            if not graph_path.exists():
                print(f"\nWARNING: Graph not found for {category}: {graph_path}")
                continue

            prefix = f"fig_unrolled_cascade_{category}"
            summary = run_cascade_analysis(
                graph_path=graph_path,
                category=category,
                output_prefix=prefix,
                weight_threshold=args.weight_threshold,
            )
            summaries.append(summary)

        # Generate comparison figure
        if summaries:
            print(f"\n{'=' * 60}")
            print("Generating cross-category comparison figure...")
            print(f"{'=' * 60}")
            generate_comparison_figure(summaries)

            # Print summary table
            print(f"\n{'=' * 60}")
            print("Summary Table")
            print(f"{'=' * 60}")
            print(f"{'Category':<16} {'Steps':>5} {'Gap':>4} {'Weight':>7} "
                  f"{'Layers':>10} {'Dominant Type':<22}")
            print("-" * 70)
            for s in sorted(summaries, key=lambda x: x["path_length"],
                            reverse=True):
                lr = s["layer_range"]
                gap_str = "yes" if s["has_gap"] else "no"
                dom = MOTIF_SHORT_NAMES.get(s["dominant_type"],
                                             s["dominant_type"] or "none")
                print(f"{s['category']:<16} {s['path_length']:>5} {gap_str:>4} "
                      f"{s['total_path_weight']:>7.1f} "
                      f"{'L' + str(lr[0]) + '-L' + str(lr[1]):>10} "
                      f"{dom:<22}")

        print("\nAll categories done!")

    else:
        # Single-graph mode (default: Dallas multihop)
        summary = run_cascade_analysis(
            graph_path=GRAPH_PATH,
            category="multihop",
            weight_threshold=args.weight_threshold,
        )
        print("\nDone!")


if __name__ == "__main__":
    main()
