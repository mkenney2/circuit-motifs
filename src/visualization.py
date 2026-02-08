"""Visualization for motif profiles, heatmaps, dendrograms, and graph drawing.

Produces publication-ready figures for the blog post:
- Motif Z-score bar charts (single graph and cross-task)
- Z-score heatmap across tasks and motif classes
- Task similarity dendrogram
- Attribution graph drawing with highlighted motif instances
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import igraph as ig
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram

from src.motif_census import (
    TRIAD_LABELS,
    CONNECTED_TRIAD_INDICES,
    MotifInstance,
    find_motif_instances,
)
from src.null_model import NullModelResult
from src.comparison import TaskProfile
from src.scale_comparison import ModelProfile, ScaleTrend


# Color palette for task categories
TASK_COLORS: dict[str, str] = {
    "factual_recall": "#1f77b4",
    "multihop": "#ff7f0e",
    "arithmetic": "#2ca02c",
    "creative": "#d62728",
    "multilingual": "#9467bd",
    "safety": "#8c564b",
    "reasoning": "#e377c2",
    "code": "#17becf",
    "uncategorized": "#7f7f7f",
}


def plot_zscore_bar(
    null_result: NullModelResult,
    title: str = "Motif Z-Score Profile",
    threshold: float = 2.0,
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Z-score bar chart for a single graph's motif profile.

    Args:
        null_result: NullModelResult from null model computation.
        title: Plot title.
        threshold: Z-score threshold lines for significance.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    z_scores = null_result.z_scores
    labels = TRIAD_LABELS if len(z_scores) == 16 else [str(i) for i in range(len(z_scores))]

    # Use connected triads only for size 3
    if len(z_scores) == 16:
        indices = CONNECTED_TRIAD_INDICES
        z_plot = z_scores[indices]
        label_plot = [labels[i] for i in indices]
    else:
        indices = list(range(len(z_scores)))
        z_plot = z_scores
        label_plot = labels

    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#d62728" if z > threshold else "#1f77b4" if z < -threshold else "#7f7f7f"
              for z in z_plot]

    bars = ax.bar(range(len(z_plot)), z_plot, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5, label=f"Z = {threshold}")
    ax.axhline(y=-threshold, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(range(len(label_plot)))
    ax.set_xticklabels(label_plot, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.set_title(title, fontsize=14)

    # Legend
    enriched_patch = mpatches.Patch(color="#d62728", label="Enriched")
    depleted_patch = mpatches.Patch(color="#1f77b4", label="Anti-enriched")
    ns_patch = mpatches.Patch(color="#7f7f7f", label="Not significant")
    ax.legend(handles=[enriched_patch, depleted_patch, ns_patch], loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_zscore_heatmap(
    profiles: dict[str, TaskProfile],
    title: str = "Motif Z-Score Heatmap Across Tasks",
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a heatmap of mean Z-scores across tasks and motif classes.

    Args:
        profiles: Dict mapping task name to TaskProfile.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    tasks = sorted(profiles.keys())
    labels = TRIAD_LABELS

    # Use connected triads only
    indices = CONNECTED_TRIAD_INDICES
    col_labels = [labels[i] for i in indices]

    # Build data matrix
    data = np.array([profiles[t].mean_z[indices] for t in tasks])

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=tasks,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean Z-score"},
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Triad Class", fontsize=12)
    ax.set_ylabel("Task Category", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sp_heatmap(
    profiles: dict[str, TaskProfile],
    title: str = "Significance Profile Heatmap",
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of mean significance profiles across tasks.

    Args:
        profiles: Dict mapping task name to TaskProfile.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    tasks = sorted(profiles.keys())
    labels = TRIAD_LABELS
    indices = CONNECTED_TRIAD_INDICES
    col_labels = [labels[i] for i in indices]

    data = np.array([profiles[t].mean_sp[indices] for t in tasks])

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=tasks,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Mean SP"},
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Triad Class", fontsize=12)
    ax.set_ylabel("Task Category", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_task_dendrogram(
    linkage_matrix: np.ndarray,
    task_names: list[str],
    title: str = "Task Similarity Dendrogram",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a dendrogram of task category similarity based on motif profiles.

    Args:
        linkage_matrix: Linkage matrix from scipy hierarchical clustering.
        task_names: Ordered task names matching the linkage matrix.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=task_names,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0,
    )

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cosine Distance", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cosine_similarity_matrix(
    sim_matrix: np.ndarray,
    task_names: list[str],
    title: str = "Task Cosine Similarity",
    figsize: tuple[float, float] = (8, 7),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot pairwise cosine similarity matrix as a heatmap.

    Args:
        sim_matrix: Square similarity matrix.
        task_names: Task names for labels.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        sim_matrix,
        xticklabels=task_names,
        yticklabels=task_names,
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"label": "Cosine Similarity"},
    )

    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_grouped_bar(
    profiles: dict[str, TaskProfile],
    title: str = "Motif Z-Scores by Task Category",
    figsize: tuple[float, float] = (16, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot grouped bar chart of mean Z-scores across tasks.

    Args:
        profiles: Dict mapping task name to TaskProfile.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    tasks = sorted(profiles.keys())
    labels = TRIAD_LABELS
    indices = CONNECTED_TRIAD_INDICES
    col_labels = [labels[i] for i in indices]
    n_motifs = len(indices)
    n_tasks = len(tasks)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.8 / n_tasks
    x = np.arange(n_motifs)

    for i, task in enumerate(tasks):
        z_vals = profiles[task].mean_z[indices]
        z_err = profiles[task].std_sp[indices] if profiles[task].n_graphs > 1 else None
        color = TASK_COLORS.get(task, f"C{i}")
        offset = (i - n_tasks / 2 + 0.5) * bar_width
        ax.bar(x + offset, z_vals, bar_width, label=task, color=color,
               edgecolor="black", linewidth=0.3, yerr=z_err, capsize=2)

    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.4)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Z-score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


# --- Attribution graph drawing with motif highlighting ---
# Visual style modeled after the Neuronpedia circuit-tracer graph viewer:
#   - Grid layout: X = token position (ctx_idx), Y = layer
#   - Embeddings at bottom, logits at top
#   - Layer labels on the left axis, prompt tokens along the bottom
#   - Node shapes: squares for embeddings, circles for features, pentagons for logits
#   - Green edges for excitatory, red for inhibitory
#   - Motif nodes highlighted in cyan (Neuronpedia selection color)

# Role-based node colors for motif highlighting
ROLE_COLORS: dict[str, str] = {
    "regulator": "#e74c8a",   # magenta-pink
    "source": "#e74c8a",      # magenta-pink
    "source_a": "#e74c8a",    # magenta-pink
    "source_b": "#ff7f0e",    # orange
    "mediator": "#ff7f0e",    # orange
    "target": "#00d4aa",      # teal-green (Neuronpedia accent)
    "target_a": "#00d4aa",    # teal-green
    "target_b": "#17becf",    # cyan
    "node_a": "#e74c8a",      # magenta-pink
    "node_b": "#ff7f0e",      # orange
    "node_c": "#00d4aa",      # teal-green
}

# Node type colors (context / non-motif)
_EMB_COLOR = "#a8d8ea"        # light blue
_FEATURE_COLOR = "#d5d5d5"    # light gray
_LOGIT_COLOR = "#c3e6cb"      # light green
_MOTIF_HIGHLIGHT = "#00d4aa"  # Neuronpedia teal


def _igraph_to_networkx(graph: ig.Graph) -> nx.DiGraph:
    """Convert an igraph DiGraph to a NetworkX DiGraph, preserving attributes.

    Args:
        graph: A directed igraph.Graph.

    Returns:
        A networkx.DiGraph with all node and edge attributes preserved.
    """
    nxg = nx.DiGraph()
    node_attrs = graph.vs.attributes() if graph.vcount() > 0 else []
    edge_attrs = graph.es.attributes() if graph.ecount() > 0 else []

    for v in graph.vs:
        attrs = {attr: v[attr] for attr in node_attrs}
        attrs["ig_index"] = v.index
        nxg.add_node(v.index, **attrs)

    for e in graph.es:
        attrs = {attr: e[attr] for attr in edge_attrs}
        nxg.add_edge(e.source, e.target, **attrs)

    return nxg


def _compute_layered_layout(
    graph: ig.Graph,
    horizontal_spacing: float = 1.5,
    vertical_spacing: float = 2.0,
) -> dict[int, tuple[float, float]]:
    """Compute a layered layout positioning nodes by transformer layer.

    Nodes are arranged vertically by their layer attribute (early layers at
    top, late layers at bottom). Within each layer, nodes are spread
    horizontally.

    Args:
        graph: A directed igraph.Graph with "layer" vertex attribute.
        horizontal_spacing: Horizontal distance between nodes in the same layer.
        vertical_spacing: Vertical distance between layers.

    Returns:
        Dict mapping node index to (x, y) position.
    """
    layer_groups: dict[int, list[int]] = defaultdict(list)
    for v in graph.vs:
        layer = v["layer"] if "layer" in graph.vs.attributes() else 0
        layer_groups[layer].append(v.index)

    sorted_layers = sorted(layer_groups.keys())

    pos: dict[int, tuple[float, float]] = {}
    for layer_rank, layer_id in enumerate(sorted_layers):
        nodes_in_layer = layer_groups[layer_id]
        n = len(nodes_in_layer)
        y = -layer_rank * vertical_spacing

        for i, node_idx in enumerate(nodes_in_layer):
            x = (i - (n - 1) / 2.0) * horizontal_spacing
            pos[node_idx] = (x, y)

    return pos


def _compute_neuronpedia_layout(
    graph: ig.Graph,
    token_spacing: float = 1.0,
    layer_spacing: float = 1.0,
    jitter_spacing: float = 0.25,
) -> tuple[dict[int, tuple[float, float]], list[int], list[str]]:
    """Compute a Neuronpedia-style grid layout: X=token position, Y=layer.

    Nodes are placed on a grid where the x-axis is the token position
    (ctx_idx) and the y-axis is the transformer layer. Multiple nodes
    at the same (layer, ctx_idx) are spread with small horizontal offsets.

    Embeddings are at the bottom (y=0), layers increase upward, and a
    special logit row is placed above the highest layer.

    Args:
        graph: A directed igraph.Graph with "layer" and "ctx_idx" attributes.
        token_spacing: Horizontal distance between adjacent token positions.
        layer_spacing: Vertical distance between adjacent layers.
        jitter_spacing: Sub-offset for multiple nodes at the same grid cell.

    Returns:
        Tuple of:
        - pos: Dict mapping node index to (x, y) position.
        - sorted_layers: List of layer IDs in bottom-to-top order.
        - layer_labels: Corresponding human-readable labels ("Emb", "L1", ..., "Lgt").
    """
    has_layer = "layer" in graph.vs.attributes()
    has_ctx = "ctx_idx" in graph.vs.attributes()
    has_ft = "feature_type" in graph.vs.attributes()

    # Determine the set of real transformer layers (excluding embedding = -1)
    real_layers: set[int] = set()
    max_layer = 0
    for v in graph.vs:
        layer = v["layer"] if has_layer else 0
        if layer >= 0:
            real_layers.add(layer)
            max_layer = max(max_layer, layer)

    # Build ordered layer list: embedding (-1), then 0..max, then logit pseudo-layer
    logit_layer_id = max_layer + 1
    sorted_layers: list[int] = [-1] + sorted(real_layers) + [logit_layer_id]

    layer_to_y: dict[int, float] = {}
    for rank, layer_id in enumerate(sorted_layers):
        layer_to_y[layer_id] = rank * layer_spacing

    # Build human-readable labels
    layer_labels: list[str] = []
    for layer_id in sorted_layers:
        if layer_id == -1:
            layer_labels.append("Emb")
        elif layer_id == logit_layer_id:
            layer_labels.append("Lgt")
        else:
            layer_labels.append(f"L{layer_id}")

    # Group nodes by (layer_bucket, ctx_idx) to handle collisions
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for v in graph.vs:
        layer = v["layer"] if has_layer else 0
        ctx = v["ctx_idx"] if has_ctx else 0
        ft = v["feature_type"] if has_ft else ""

        # Map logit nodes to the logit pseudo-layer
        if ft == "logit":
            layer_bucket = logit_layer_id
        elif layer == -1:
            layer_bucket = -1
        else:
            layer_bucket = layer

        grid[(layer_bucket, ctx)].append(v.index)

    # Assign positions
    pos: dict[int, tuple[float, float]] = {}
    for (layer_bucket, ctx), node_list in grid.items():
        base_x = ctx * token_spacing
        y = layer_to_y.get(layer_bucket, 0)
        n = len(node_list)
        for i, node_idx in enumerate(node_list):
            offset = (i - (n - 1) / 2.0) * jitter_spacing
            pos[node_idx] = (base_x + offset, y)

    return pos, sorted_layers, layer_labels


def plot_graph_with_motif(
    graph: ig.Graph,
    motif_instance: MotifInstance,
    title: str | None = None,
    figsize: tuple[float, float] = (16, 12),
    context_node_size: float = 25,
    motif_node_size: float = 200,
    context_alpha: float = 0.45,
    context_edge_alpha: float = 0.15,
    motif_edge_width: float = 2.5,
    label_fontsize: int = 8,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Draw the attribution graph in Neuronpedia style with a motif highlighted.

    Layout matches the Neuronpedia circuit-tracer viewer:
    - X-axis: token positions with prompt tokens labeled at the bottom
    - Y-axis: transformer layers with labels on the left (Emb at bottom, Lgt at top)
    - Node shapes: squares for embeddings, circles for features, pentagons for logits
    - Motif nodes are highlighted with role colors and clerp label annotations
    - Edges colored green (excitatory) or red (inhibitory) within the motif

    Args:
        graph: A directed igraph.Graph (attribution graph).
        motif_instance: A MotifInstance to highlight.
        title: Plot title. Defaults to "Attribution Graph — {motif label}".
        figsize: Figure size.
        context_node_size: Size of non-motif nodes.
        motif_node_size: Size of motif nodes.
        context_alpha: Alpha for context nodes.
        context_edge_alpha: Alpha for context edges.
        motif_edge_width: Line width for motif edges.
        label_fontsize: Font size for clerp labels.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    nxg = _igraph_to_networkx(graph)
    pos, sorted_layers, layer_labels = _compute_neuronpedia_layout(graph)

    motif_nodes = set(motif_instance.node_indices)
    motif_edges = set(motif_instance.subgraph_edges)

    if title is None:
        title = f"Attribution Graph — {motif_instance.label}"

    has_ft = "feature_type" in graph.vs.attributes()
    has_clerp = "clerp" in graph.vs.attributes()
    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False
    has_weight = "weight" in graph.es.attributes() if graph.ecount() > 0 else False

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Precompute max edge weight for width scaling
    if has_weight and graph.ecount() > 0:
        max_w = max(abs(graph.es[e]["weight"]) for e in range(graph.ecount()))
    else:
        max_w = 1.0

    # --- Classify nodes by type ---
    emb_nodes: list[int] = []
    logit_nodes: list[int] = []
    feature_nodes: list[int] = []
    for v in graph.vs:
        ft = v["feature_type"] if has_ft else ""
        if ft == "embedding":
            emb_nodes.append(v.index)
        elif ft == "logit":
            logit_nodes.append(v.index)
        else:
            feature_nodes.append(v.index)

    # --- Draw context edges (non-motif) — zorder 1 ---
    context_edges = [(u, v) for u, v in nxg.edges if (u, v) not in motif_edges]
    if context_edges:
        nx.draw_networkx_edges(
            nxg, pos, edgelist=context_edges, alpha=context_edge_alpha,
            edge_color="#888888", arrows=True, arrowsize=4,
            connectionstyle="arc3,rad=0.05", ax=ax, node_size=context_node_size,
        )

    # --- Draw context nodes by type — zorder 2 ---
    ctx_emb = [n for n in emb_nodes if n not in motif_nodes]
    ctx_logit = [n for n in logit_nodes if n not in motif_nodes]
    ctx_feature = [n for n in feature_nodes if n not in motif_nodes]

    if ctx_emb:
        nx.draw_networkx_nodes(
            nxg, pos, nodelist=ctx_emb, node_size=context_node_size * 1.2,
            node_color="white", edgecolors="#3182bd", linewidths=1.0,
            node_shape="s", alpha=max(context_alpha, 0.6), ax=ax,
        )
    if ctx_feature:
        nx.draw_networkx_nodes(
            nxg, pos, nodelist=ctx_feature, node_size=context_node_size,
            node_color="white", edgecolors="#666666", linewidths=0.8,
            node_shape="o", alpha=context_alpha, ax=ax,
        )
    if ctx_logit:
        nx.draw_networkx_nodes(
            nxg, pos, nodelist=ctx_logit, node_size=context_node_size * 1.8,
            node_color="white", edgecolors="#41ab5d", linewidths=1.2,
            node_shape="p", alpha=max(context_alpha, 0.65), ax=ax,
        )

    # --- Draw motif edges — zorder 5 (on top of context) ---
    for u, v in motif_instance.subgraph_edges:
        eid = graph.get_eid(u, v, error=False)
        if eid == -1:
            edge_color = "#333333"
            width = motif_edge_width
        else:
            sign = graph.es[eid]["sign"] if has_sign else "excitatory"
            edge_color = "#2ca02c" if sign == "excitatory" else "#d62728"
            if has_weight:
                w = graph.es[eid]["weight"]
                width = motif_edge_width * (0.5 + 0.5 * abs(w) / max_w) if max_w > 0 else motif_edge_width
            else:
                width = motif_edge_width

        edges_coll = nx.draw_networkx_edges(
            nxg, pos, edgelist=[(u, v)], edge_color=edge_color,
            width=width, arrows=True, arrowsize=12, alpha=0.9,
            connectionstyle="arc3,rad=0.08", ax=ax, node_size=motif_node_size,
        )
        if edges_coll:
            for artist in (edges_coll if hasattr(edges_coll, '__iter__') else [edges_coll]):
                artist.set_zorder(5)

    # --- Draw motif nodes — zorder 6 (on top of edges) ---
    for node_idx in motif_instance.node_indices:
        ft = graph.vs[node_idx]["feature_type"] if has_ft else ""
        role = motif_instance.node_roles.get(node_idx, "node_a")
        color = ROLE_COLORS.get(role, _MOTIF_HIGHLIGHT)

        if ft == "embedding":
            shape = "s"
        elif ft == "logit":
            shape = "p"
        else:
            shape = "o"

        coll = nx.draw_networkx_nodes(
            nxg, pos, nodelist=[node_idx], node_size=motif_node_size,
            node_color=color, edgecolors="black", linewidths=1.5,
            node_shape=shape, ax=ax,
        )
        if coll:
            coll.set_zorder(6)

    # --- Clerp labels on motif nodes — zorder 7 ---
    label_offsets = [(16, 16), (-16, 16), (16, -16), (-16, -16), (20, 0)]
    for i, node_idx in enumerate(motif_instance.node_indices):
        if not has_clerp:
            continue
        clerp = graph.vs[node_idx]["clerp"]
        if not clerp:
            continue

        if len(clerp) > 45:
            clerp = clerp[:42] + "..."

        role = motif_instance.node_roles.get(node_idx, "node_a")
        role_label = role.replace("_", " ")
        display = f"{clerp}\n({role_label})"

        x, y = pos[node_idx]
        ox, oy = label_offsets[i % len(label_offsets)]
        ha = "left" if ox > 0 else "right"

        ann = ax.annotate(
            display,
            xy=(x, y),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=label_fontsize,
            ha=ha,
            va="bottom" if oy > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=ROLE_COLORS.get(role, "gray"), alpha=0.95,
                      linewidth=1.5),
            arrowprops=dict(arrowstyle="-|>", color=ROLE_COLORS.get(role, "gray"),
                            lw=1.0),
            zorder=7,
        )

    # --- Label logit nodes (always — they're the output) ---
    for v_idx in logit_nodes:
        if v_idx in motif_nodes:
            continue  # Already labeled above
        if not has_clerp:
            continue
        clerp = graph.vs[v_idx]["clerp"]
        if not clerp:
            continue
        display = clerp
        if len(display) > 45:
            display = display[:42] + "..."

        x, y = pos[v_idx]
        ax.annotate(
            display,
            xy=(x, y),
            xytext=(0, 14),
            textcoords="offset points",
            fontsize=label_fontsize,
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9",
                      edgecolor="#74c476", alpha=0.95, linewidth=1.0),
            zorder=7,
        )

    # --- Label embedding nodes with their token text ---
    prompt_tokens = graph["prompt_tokens"] if "prompt_tokens" in graph.attributes() else []
    has_ctx = "ctx_idx" in graph.vs.attributes()

    if prompt_tokens:
        for v_idx in emb_nodes:
            if v_idx in motif_nodes:
                continue  # Already labeled
            ctx = graph.vs[v_idx]["ctx_idx"] if has_ctx else -1
            if 0 <= ctx < len(prompt_tokens):
                tok = prompt_tokens[ctx].replace("\u2191", "^").strip()
                if not tok:
                    continue
                display = f'Emb: "{tok}"'
            else:
                continue

            x, y = pos[v_idx]
            ax.annotate(
                display,
                xy=(x, y),
                xytext=(0, -10),
                textcoords="offset points",
                fontsize=label_fontsize - 1,
                ha="center",
                va="top",
                color="#2171b5",
                alpha=0.9,
                zorder=3,
            )

    # --- Axis setup: layer labels on left, token labels on bottom ---
    if has_ctx:
        all_ctx = [v["ctx_idx"] for v in graph.vs]
        min_ctx, max_ctx = min(all_ctx), max(all_ctx)
    else:
        min_ctx, max_ctx = 0, 0

    # Y-axis: layer labels
    layer_y_positions = [i * 1.0 for i in range(len(sorted_layers))]
    ax.set_yticks(layer_y_positions)
    ax.set_yticklabels(layer_labels, fontsize=9, color="#222222",
                       fontfamily="monospace", fontweight="bold")
    ax.yaxis.set_ticks_position("left")

    # X-axis: token labels
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
        ax.set_xticklabels(token_tick_labels, fontsize=9, rotation=45,
                           ha="right", color="#222222", fontstyle="italic",
                           fontweight="medium")
        ax.xaxis.set_ticks_position("bottom")

    # Light horizontal grid lines for layers
    for y_val in layer_y_positions:
        ax.axhline(y=y_val, color="#e0e0e0", linewidth=0.5, zorder=0)

    # Light vertical grid lines for token positions
    if has_ctx:
        for ctx in range(min_ctx, max_ctx + 1):
            ax.axvline(x=ctx * 1.0, color="#e0e0e0", linewidth=0.5, zorder=0)

    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.tick_params(axis="both", which="both", length=4, color="#999999")

    # Padding
    x_min = (min_ctx - 0.8) * 1.0
    x_max = (max_ctx + 1.5) * 1.0  # extra room for logit labels
    y_min = -1.0
    y_max = (len(sorted_layers) - 0.3) * 1.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # --- Legend ---
    legend_handles = []
    # Node type shapes
    legend_handles.append(plt.Line2D([0], [0], marker="s", color="w",
                          markeredgecolor="#3182bd", markerfacecolor="white",
                          markersize=8, label="Embedding", linewidth=0))
    legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                          markeredgecolor="#666666", markerfacecolor="white",
                          markersize=8, label="Feature", linewidth=0))
    legend_handles.append(plt.Line2D([0], [0], marker="p", color="w",
                          markeredgecolor="#41ab5d", markerfacecolor="white",
                          markersize=8, label="Logit", linewidth=0))
    legend_handles.append(plt.Line2D([0], [0], color="w", linewidth=0))  # spacer

    # Motif role colors
    seen_roles: set[str] = set()
    for node_idx in motif_instance.node_indices:
        role = motif_instance.node_roles.get(node_idx, "node_a")
        if role not in seen_roles:
            seen_roles.add(role)
            color = ROLE_COLORS.get(role, "#7f7f7f")
            legend_handles.append(plt.Line2D([0], [0], marker="o", color="w",
                                  markerfacecolor=color, markeredgecolor="black",
                                  markersize=10, linewidth=0,
                                  label=f"Motif: {role.replace('_', ' ')}"))

    # Edge types
    legend_handles.append(plt.Line2D([0], [1], color="#2ca02c", linewidth=2.5,
                          label="Excitatory"))
    legend_handles.append(plt.Line2D([0], [1], color="#d62728", linewidth=2.5,
                          label="Inhibitory"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.95, edgecolor="#cccccc", fancybox=True)

    ax.set_title(title, fontsize=13, pad=12, color="#111111", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_top_motif(
    graph: ig.Graph,
    motif_isoclass: int,
    rank: int = 0,
    size: int = 3,
    **kwargs: Any,
) -> tuple[plt.Figure, MotifInstance]:
    """Find motif instances and plot the one at the given rank.

    Convenience function combining find_motif_instances() and
    plot_graph_with_motif().

    Args:
        graph: A directed igraph.Graph.
        motif_isoclass: igraph isomorphism class ID of the motif.
        rank: Which instance to plot (0 = highest weight, 1 = second, etc.).
        size: Motif size (3 or 4).
        **kwargs: Additional keyword arguments passed to plot_graph_with_motif().

    Returns:
        Tuple of (matplotlib Figure, the MotifInstance that was plotted).

    Raises:
        ValueError: If no instances found or rank exceeds available instances.
    """
    instances = find_motif_instances(
        graph, motif_isoclass=motif_isoclass, size=size, sort_by="weight",
    )

    if not instances:
        raise ValueError(
            f"No instances of isoclass {motif_isoclass} found in graph"
        )
    if rank >= len(instances):
        raise ValueError(
            f"Rank {rank} requested but only {len(instances)} instances found"
        )

    instance = instances[rank]
    fig = plot_graph_with_motif(graph, instance, **kwargs)
    return fig, instance


# --- Cross-scale visualization ---

MODEL_SCALE_COLORS: dict[str, str] = {
    "gemma-3-270m-it": "#4e79a7",
    "gemma-3-1b-it":   "#f28e2b",
    "gemma-3-4b-it":   "#e15759",
    "gemma-3-12b-it":  "#76b7b2",
    "gemma-3-27b-it":  "#59a14f",
}


def plot_scale_trend(
    scale_trends: list[ScaleTrend],
    motif_indices: list[int] | None = None,
    title: str = "Motif Z-Score Scaling Trends",
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Z-score vs log(params) line plot with error bands.

    Args:
        scale_trends: List of ScaleTrend objects from compute_scale_trends().
        motif_indices: Which motif indices to plot. Defaults to connected triads.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    if motif_indices is None:
        motif_indices = CONNECTED_TRIAD_INDICES

    trends = [t for t in scale_trends if t.motif_index in motif_indices]

    fig, ax = plt.subplots(figsize=figsize)

    for trend in trends:
        if not trend.param_counts:
            continue
        x = [np.log10(p * 1e6) for p in trend.param_counts]
        y = trend.values
        y_std = trend.std_values

        color = f"C{motif_indices.index(trend.motif_index) % 10}"
        label = trend.motif_label
        if trend.is_significant:
            label += " *"

        ax.plot(x, y, "o-", label=label, color=color, linewidth=2, markersize=6)

        if any(s > 0 for s in y_std):
            y_arr = np.array(y)
            s_arr = np.array(y_std)
            ax.fill_between(x, y_arr - s_arr, y_arr + s_arr, alpha=0.15, color=color)

    # X-axis labels: model names at tick positions
    if trends and trends[0].param_counts:
        tick_positions = [np.log10(p * 1e6) for p in trends[0].param_counts]
        tick_labels = [f"{p}M" if p < 1000 else f"{p // 1000}B"
                       for p in trends[0].param_counts]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=10)

    ax.axhline(y=2.0, color="red", linestyle="--", alpha=0.3, label="Z = 2.0")
    ax.axhline(y=-2.0, color="red", linestyle="--", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xlabel("Model Size", fontsize=12)
    ax.set_ylabel("Mean Z-score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=8, ncol=2, loc="best")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_scale_heatmap(
    model_profiles: dict[str, ModelProfile],
    metric: str = "z_score",
    title: str = "Motif Profile Across Model Scales",
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of motif profiles across model scales.

    Y-axis: models sorted smallest→largest. X-axis: motif classes.
    Same colormap as plot_zscore_heatmap.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        metric: "z_score" or "sp".
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    sorted_models = sorted(
        model_profiles.items(),
        key=lambda kv: kv[1].model_spec.n_params,
    )

    model_names = []
    for mid, mp in sorted_models:
        p = mp.model_spec.n_params
        label = f"{p}M" if p < 1000 else f"{p // 1000}B"
        model_names.append(f"{mid} ({label})")

    indices = CONNECTED_TRIAD_INDICES
    col_labels = [TRIAD_LABELS[i] for i in indices]

    if metric == "z_score":
        data = np.array([mp.overall_mean_z[indices] for _, mp in sorted_models])
        cbar_label = "Mean Z-score"
    else:
        data = np.array([mp.overall_mean_sp[indices] for _, mp in sorted_models])
        cbar_label = "Mean SP"

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        data,
        xticklabels=col_labels,
        yticklabels=model_names,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Triad Class", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sp_overlay(
    model_profiles: dict[str, ModelProfile],
    title: str = "Significance Profiles Across Scales",
    figsize: tuple[float, float] = (16, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot grouped bar chart of SP vectors, one color per model.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    sorted_models = sorted(
        model_profiles.items(),
        key=lambda kv: kv[1].model_spec.n_params,
    )

    indices = CONNECTED_TRIAD_INDICES
    col_labels = [TRIAD_LABELS[i] for i in indices]
    n_motifs = len(indices)
    n_models = len(sorted_models)

    fig, ax = plt.subplots(figsize=figsize)

    bar_width = 0.8 / n_models
    x = np.arange(n_motifs)

    for i, (model_id, mp) in enumerate(sorted_models):
        sp_vals = mp.overall_mean_sp[indices]
        color = MODEL_SCALE_COLORS.get(model_id, f"C{i}")
        p = mp.model_spec.n_params
        label = f"{p}M" if p < 1000 else f"{p // 1000}B"
        offset = (i - n_models / 2 + 0.5) * bar_width
        ax.bar(x + offset, sp_vals, bar_width, label=label, color=color,
               edgecolor="black", linewidth=0.3)

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean SP", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=9, title="Model Size")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_per_task_scaling(
    model_profiles: dict[str, ModelProfile],
    task_name: str,
    motif_indices: list[int] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot scaling curves for a specific task category.

    Args:
        model_profiles: Dict mapping model_id to ModelProfile.
        task_name: Task category to plot.
        motif_indices: Which motif indices to include.
        title: Plot title. Defaults to "Scaling: {task_name}".
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    from src.scale_comparison import compare_task_across_scales

    if title is None:
        title = f"Motif Scaling: {task_name}"

    trends = compare_task_across_scales(model_profiles, task_name, metric="z_score")

    if not trends:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f"No data for task '{task_name}'",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title)
        return fig

    return plot_scale_trend(
        trends,
        motif_indices=motif_indices,
        title=title,
        figsize=figsize,
        save_path=save_path,
    )


def plot_scale_dendrogram(
    linkage_matrix: np.ndarray,
    model_names: list[str],
    title: str = "Model Similarity Dendrogram",
    figsize: tuple[float, float] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot a dendrogram of model similarity based on motif profiles.

    Reuses the visual style from plot_task_dendrogram.

    Args:
        linkage_matrix: Linkage matrix from scipy hierarchical clustering.
        model_names: Ordered model names matching the linkage matrix.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    dendrogram(
        linkage_matrix,
        labels=model_names,
        ax=ax,
        leaf_rotation=45,
        leaf_font_size=10,
        color_threshold=0,
    )

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Cosine Distance", fontsize=12)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
