"""Visualization for unrolled motif analysis results.

Produces figures for:
- Individual unrolled motif instance diagrams (layered layout)
- Z-score spectrum across all unrolled motif types
- Cross-motif co-occurrence heatmap
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

from src.unrolled_motifs import (
    UnrolledMotifTemplate,
    get_effective_layer,
    CATALOG,
)
from src.unrolled_census import UnrolledMotifInstance
from src.unrolled_null_model import UnrolledNullResult
from src.visualization import (
    _igraph_to_networkx,
    _compute_neuronpedia_layout,
    ROLE_COLORS,
)


# Chain-based colors for unrolled motif visualization
CHAIN_COLORS: list[str] = [
    "#e74c8a",  # magenta-pink (chain 0)
    "#17becf",  # cyan (chain 1)
    "#ff7f0e",  # orange (chain 2 / bias)
    "#2ca02c",  # green (chain 3)
    "#9467bd",  # purple (chain 4)
]


def plot_unrolled_instance(
    graph: ig.Graph,
    instance: UnrolledMotifInstance,
    template: UnrolledMotifTemplate | None = None,
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
    """Draw an attribution graph with an unrolled motif instance highlighted.

    Uses the Neuronpedia-style layout from visualization.py with chain-colored
    nodes and sign-colored edges for the motif instance.

    Args:
        graph: A directed igraph.Graph (attribution graph).
        instance: An UnrolledMotifInstance to highlight.
        template: Optional template for chain membership info.
        title: Plot title. Defaults to template name.
        figsize: Figure size.
        context_node_size: Size of non-motif nodes.
        motif_node_size: Size of motif nodes.
        context_alpha: Alpha for context nodes/edges.
        context_edge_alpha: Alpha for context edges.
        motif_edge_width: Line width for motif edges.
        label_fontsize: Font size for clerp labels.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    nxg = _igraph_to_networkx(graph)
    pos, sorted_layers, layer_labels = _compute_neuronpedia_layout(graph)

    motif_nodes = set(instance.node_indices)
    has_ft = "feature_type" in graph.vs.attributes()
    has_clerp = "clerp" in graph.vs.attributes()
    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False
    has_weight = "weight" in graph.es.attributes() if graph.ecount() > 0 else False

    if title is None:
        pretty_name = instance.template_name.replace("_", " ").title()
        title = f"Unrolled Motif — {pretty_name}"

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Precompute max edge weight for width scaling
    if has_weight and graph.ecount() > 0:
        max_w = max(abs(graph.es[e]["weight"]) for e in range(graph.ecount()))
    else:
        max_w = 1.0

    # Build set of motif edges
    motif_edge_set: set[tuple[int, int]] = set()
    if template is not None:
        for tmpl_edge in template.edges:
            src_idx = template.node_ids.index(tmpl_edge["src"])
            tgt_idx = template.node_ids.index(tmpl_edge["tgt"])
            graph_src = instance.node_indices[src_idx]
            graph_tgt = instance.node_indices[tgt_idx]
            motif_edge_set.add((graph_src, graph_tgt))
    else:
        # Fallback: all edges between motif nodes
        for e in graph.es:
            if e.source in motif_nodes and e.target in motif_nodes:
                motif_edge_set.add((e.source, e.target))

    # --- Draw context edges ---
    context_edges = [(u, v) for u, v in nxg.edges if (u, v) not in motif_edge_set]
    if context_edges:
        nx.draw_networkx_edges(
            nxg, pos, edgelist=context_edges, alpha=context_edge_alpha,
            edge_color="#888888", arrows=True, arrowsize=4,
            connectionstyle="arc3,rad=0.05", ax=ax, node_size=context_node_size,
        )

    # --- Draw context nodes ---
    ctx_nodes = [v.index for v in graph.vs if v.index not in motif_nodes]
    if ctx_nodes:
        nx.draw_networkx_nodes(
            nxg, pos, nodelist=ctx_nodes, node_size=context_node_size,
            node_color="white", edgecolors="#666666", linewidths=0.8,
            node_shape="o", alpha=context_alpha, ax=ax,
        )

    # --- Draw motif edges with sign colors ---
    for u, v in motif_edge_set:
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

    # --- Draw motif nodes colored by role ---
    for i, node_idx in enumerate(instance.node_indices):
        role = instance.node_roles.get(node_idx, f"node_{i}")

        # Try to get chain color from template
        color = ROLE_COLORS.get(role, CHAIN_COLORS[i % len(CHAIN_COLORS)])

        ft = graph.vs[node_idx]["feature_type"] if has_ft else ""
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

    # --- Clerp labels on motif nodes ---
    label_offsets = [(16, 16), (-16, 16), (16, -16), (-16, -16), (20, 0)]
    for i, node_idx in enumerate(instance.node_indices):
        if not has_clerp:
            continue
        clerp = graph.vs[node_idx]["clerp"]
        if not clerp:
            continue
        if len(clerp) > 45:
            clerp = clerp[:42] + "..."

        role = instance.node_roles.get(node_idx, f"node_{i}")
        display = f"{clerp}\n({role.replace('_', ' ')})"

        x, y = pos[node_idx]
        ox, oy = label_offsets[i % len(label_offsets)]
        ha = "left" if ox > 0 else "right"

        ax.annotate(
            display, xy=(x, y), xytext=(ox, oy),
            textcoords="offset points", fontsize=label_fontsize,
            ha=ha, va="bottom" if oy > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=CHAIN_COLORS[i % len(CHAIN_COLORS)],
                      alpha=0.95, linewidth=1.5),
            arrowprops=dict(arrowstyle="-|>",
                            color=CHAIN_COLORS[i % len(CHAIN_COLORS)], lw=1.0),
            zorder=7,
        )

    # --- Axis setup ---
    layer_y_positions = [i * 1.0 for i in range(len(sorted_layers))]
    ax.set_yticks(layer_y_positions)
    ax.set_yticklabels(layer_labels, fontsize=9, color="#222222",
                       fontfamily="monospace", fontweight="bold")

    for y_val in layer_y_positions:
        ax.axhline(y=y_val, color="#e0e0e0", linewidth=0.5, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["left"].set_color("#999999")

    # --- Legend ---
    legend_handles = [
        plt.Line2D([0], [1], color="#2ca02c", linewidth=2.5, label="Excitatory"),
        plt.Line2D([0], [1], color="#d62728", linewidth=2.5, label="Inhibitory"),
    ]
    seen_roles: set[str] = set()
    for i, node_idx in enumerate(instance.node_indices):
        role = instance.node_roles.get(node_idx, f"node_{i}")
        if role not in seen_roles:
            seen_roles.add(role)
            color = CHAIN_COLORS[i % len(CHAIN_COLORS)]
            legend_handles.append(plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor=color,
                markeredgecolor="black", markersize=10, linewidth=0,
                label=role.replace("_", " "),
            ))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=8,
              framealpha=0.95, edgecolor="#cccccc")
    ax.set_title(title, fontsize=13, pad=12, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_unrolled_spectrum(
    null_result: UnrolledNullResult,
    title: str = "Unrolled Motif Z-Score Spectrum",
    threshold: float = 2.0,
    figsize: tuple[float, float] = (12, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Z-score bar chart across all unrolled motif types.

    Args:
        null_result: UnrolledNullResult from compute_unrolled_zscores().
        title: Plot title.
        threshold: Z-score threshold for enrichment significance.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    names = list(null_result.z_scores.keys())
    z_values = [null_result.z_scores[n] for n in names]
    real_counts = [null_result.real_counts[n] for n in names]

    # Pretty labels
    pretty_names = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(figsize=figsize)

    colors = [
        "#d62728" if z > threshold else "#1f77b4" if z < -threshold else "#7f7f7f"
        for z in z_values
    ]

    bars = ax.bar(range(len(z_values)), z_values, color=colors,
                  edgecolor="black", linewidth=0.5)

    ax.axhline(y=threshold, color="red", linestyle="--", alpha=0.5,
               label=f"Z = +/-{threshold}")
    ax.axhline(y=-threshold, color="red", linestyle="--", alpha=0.5)
    ax.axhline(y=0, color="black", linewidth=0.5)

    # Annotate bars with real counts
    for i, (bar, count) in enumerate(zip(bars, real_counts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"n={count}", ha="center", va="bottom" if z_values[i] >= 0 else "top",
                fontsize=8, color="#333333")

    ax.set_xticks(range(len(pretty_names)))
    ax.set_xticklabels(pretty_names, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("Z-score", fontsize=12)
    ax.set_title(title, fontsize=14)

    enriched_patch = mpatches.Patch(color="#d62728", label="Enriched")
    depleted_patch = mpatches.Patch(color="#1f77b4", label="Anti-enriched")
    ns_patch = mpatches.Patch(color="#7f7f7f", label="Not significant")
    ax.legend(handles=[enriched_patch, depleted_patch, ns_patch], loc="upper right")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cooccurrence_heatmap(
    census_results: dict[str, list[UnrolledMotifInstance]],
    title: str = "Unrolled Motif Node Co-occurrence",
    figsize: tuple[float, float] = (9, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of node co-occurrence between unrolled motif types.

    For each pair of motif types, counts how many node indices are shared
    across their instances. Diagonal shows self-overlap (total unique nodes).

    Args:
        census_results: Dict mapping motif name to list of instances.
        title: Plot title.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    names = list(census_results.keys())
    n = len(names)

    # Collect unique node sets per motif type
    node_sets: dict[str, set[int]] = {}
    for name, instances in census_results.items():
        nodes: set[int] = set()
        for inst in instances:
            nodes.update(inst.node_indices)
        node_sets[name] = nodes

    # Build co-occurrence matrix
    matrix = np.zeros((n, n))
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            shared = len(node_sets[name_i] & node_sets[name_j])
            matrix[i, j] = shared

    pretty_names = [n.replace("_", " ") for n in names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        xticklabels=pretty_names,
        yticklabels=pretty_names,
        cmap="YlOrRd",
        annot=True,
        fmt=".0f",
        linewidths=0.5,
        square=True,
        ax=ax,
        cbar_kws={"label": "Shared nodes"},
    )

    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cross_task_unrolled_heatmap(
    task_values: dict[str, dict[str, float]],
    title: str = "Unrolled Motif Z-Scores by Task",
    cbar_label: str = "Mean Z-score",
    figsize: tuple[float, float] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot heatmap of per-motif values across task categories.

    Works for both Z-scores and SP values — just change the title and
    cbar_label accordingly.

    Args:
        task_values: Dict mapping task name to dict of motif name → value.
        title: Plot title.
        cbar_label: Label for the colorbar.
        figsize: Figure size.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure.
    """
    tasks = sorted(task_values.keys())
    if not tasks:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        return fig

    motif_names = list(next(iter(task_values.values())).keys())
    pretty_motifs = [n.replace("_", " ") for n in motif_names]

    data = np.array([
        [task_values[t].get(m, 0.0) for m in motif_names]
        for t in tasks
    ])

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        data,
        xticklabels=pretty_motifs,
        yticklabels=tasks,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": cbar_label},
    )

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Unrolled Motif", fontsize=12)
    ax.set_ylabel("Task Category", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
