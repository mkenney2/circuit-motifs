"""Generate the FFL Cascade figures for the blog post.

Two separate figures showing how feedforward loop (FFL) motifs cascade across
layers in the Dallas multihop reasoning graph:
  Figure 1 (schematic): Abstract diagram of 3 processing stages with clear flow
  Figure 2 (graph):     Real attribution graph with 6 key FFLs overlaid

Usage:
    python figures/generate_ffl_cascade.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_loader import load_attribution_graph
from src.motif_census import find_motif_instances, MOTIF_FFL
from src.visualization import _compute_neuronpedia_layout, _igraph_to_networkx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRAPH_PATH = PROJECT_ROOT / "data" / "raw" / "multihop" / "capital-state-dallas.json"
OUTPUT_SCHEMATIC = PROJECT_ROOT / "figures" / "fig_ffl_cascade_schematic.png"
OUTPUT_GRAPH = PROJECT_ROOT / "figures" / "fig_ffl_cascade_graph.png"
WEIGHT_THRESHOLD = 1.0

# Stage colors — blues for grounding, green for entity, purples for output
STAGE_COLORS = {
    1: {  # Grounding (input processing)
        "bg": "#d1e5f0",
        "rank0": "#2166ac",
        "rank1": "#4393c3",
        "rank7": "#92c5de",
    },
    2: {  # Entity resolution
        "bg": "#d9f0d3",
        "rank11": "#4daf4a",
    },
    3: {  # Output competition
        "bg": "#e8d5f0",
        "rank3": "#984ea3",
        "rank4": "#e7298a",
    },
}

# Ordered list of (rank, stage, color_key) for the 6 FFLs
KEY_FFLS = [
    (0, 1, "rank0"),
    (1, 1, "rank1"),
    (7, 1, "rank7"),
    (11, 2, "rank11"),
    (3, 3, "rank3"),
    (4, 3, "rank4"),
]

# Color lookup for each rank
FFL_COLORS: dict[int, str] = {}
for rank, stage, ckey in KEY_FFLS:
    FFL_COLORS[rank] = STAGE_COLORS[stage][ckey]

STAGE_NAMES = {
    1: "Stage 1: Grounding",
    2: "Stage 2: Entity Resolution",
    3: "Stage 3: Output Competition",
}
STAGE_TITLE_COLORS = {1: "#2166ac", 2: "#2a7e19", 3: "#984ea3"}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load graph and find FFL instances."""
    g = load_attribution_graph(str(GRAPH_PATH), weight_threshold=WEIGHT_THRESHOLD)
    instances = find_motif_instances(g, MOTIF_FFL, size=3, max_instances=50)
    return g, instances


def select_key_ffls(g, instances):
    """Select the 6 key FFL instances and annotate with stage info."""
    selected = []
    for rank, stage, ckey in KEY_FFLS:
        if rank >= len(instances):
            print(f"Warning: rank {rank} not found (only {len(instances)} instances)")
            continue
        inst = instances[rank]
        color = STAGE_COLORS[stage][ckey]
        selected.append({
            "rank": rank,
            "stage": stage,
            "color": color,
            "instance": inst,
        })
    return selected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_dark(hex_color: str) -> bool:
    """Check if a hex color is dark (for choosing white vs black text)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 140


def _draw_node_box(ax, x, y, label, color, width=2.4, height=0.7,
                   fontsize=10, bold=False, linewidth=1.5, zorder=5):
    """Draw a rounded-rectangle node with centered text."""
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2), width, height,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="black", linewidth=linewidth,
        alpha=0.92, zorder=zorder,
    )
    ax.add_patch(box)
    fw = "bold" if bold else "normal"
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight=fw, color="white" if _is_dark(color) else "#111111",
            zorder=zorder + 1)


def _draw_ffl_arrow(ax, x1, y1, x2, y2, color, linestyle="-",
                    linewidth=2.0, rad=0.12, zorder=4):
    """Draw a curved arrow between two points."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->,head_length=8,head_width=5",
        connectionstyle=f"arc3,rad={rad}",
        color=color, linewidth=linewidth, linestyle=linestyle,
        zorder=zorder, alpha=0.9,
    )
    ax.add_patch(arrow)


# ---------------------------------------------------------------------------
# Figure 1: Schematic
# ---------------------------------------------------------------------------

def _draw_flow_arrow(ax, x, y_bot, y_top, label):
    """Draw a large vertical flow arrow with a label alongside."""
    ax.annotate(
        "", xy=(x, y_top), xytext=(x, y_bot),
        arrowprops=dict(
            arrowstyle="-|>,head_length=0.5,head_width=0.35",
            color="#888888", lw=2.5, connectionstyle="arc3,rad=0",
        ),
        zorder=1,
    )
    ax.text(x + 0.3, (y_bot + y_top) / 2, label,
            fontsize=9, color="#888888", rotation=90,
            ha="left", va="center", fontstyle="italic", zorder=1)


def generate_schematic():
    """Generate the FFL cascade schematic as a standalone figure."""

    fig, ax = plt.subplots(figsize=(16, 14), facecolor="white")
    ax.set_xlim(-1, 16)
    ax.set_ylim(-0.5, 15)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Y bands for stages (bottom-to-top = early-to-late layers) ---
    # Stage 1: y = 0.2 .. 4.5
    # Stage 2: y = 5.5 .. 9.0
    # Stage 3: y = 10.0 .. 14.5
    s1_bot, s1_top = 0.2, 4.5
    s2_bot, s2_top = 5.5, 9.0
    s3_bot, s3_top = 10.0, 14.5

    ax.axhspan(s1_bot, s1_top, facecolor=STAGE_COLORS[1]["bg"], alpha=0.30, zorder=0)
    ax.axhspan(s2_bot, s2_top, facecolor=STAGE_COLORS[2]["bg"], alpha=0.30, zorder=0)
    ax.axhspan(s3_bot, s3_top, facecolor=STAGE_COLORS[3]["bg"], alpha=0.30, zorder=0)

    # Stage labels — top-left of each band, above the nodes
    for stage, (bot, top) in [(1, (s1_bot, s1_top)),
                               (2, (s2_bot, s2_top)),
                               (3, (s3_bot, s3_top))]:
        ax.text(-0.5, top - 0.1, STAGE_NAMES[stage], fontsize=12,
                fontweight="bold", color=STAGE_TITLE_COLORS[stage],
                va="top", ha="left", zorder=8)

    # --- Large flow arrow on the right side ---
    _draw_flow_arrow(ax, 15.2, s1_bot + 0.5, s3_top - 0.5,
                     "Information flow (layers 1 \u2192 16)")

    # --- Inter-stage flow arrows (centered) ---
    for y_gap_bot, y_gap_top in [(s1_top, s2_bot), (s2_top, s3_bot)]:
        mid = (y_gap_bot + y_gap_top) / 2
        ax.annotate(
            "", xy=(7.5, y_gap_top - 0.1), xytext=(7.5, y_gap_bot + 0.1),
            arrowprops=dict(
                arrowstyle="-|>,head_length=0.4,head_width=0.3",
                color="#bbbbbb", lw=2.0,
            ),
            zorder=1,
        )

    # =======================================================================
    # Stage 1: Grounding — three parallel FFLs (Embedding -> L1 -> L2)
    # Laid out left-to-right at x = 2.5, 7.5, 12.5
    # =======================================================================
    emb_y = 1.0
    l1_y = 2.4
    l2_y = 3.8

    ffl_s1 = [
        (2.5, STAGE_COLORS[1]["rank0"], '"state"', "w = 46.3"),
        (7.5, STAGE_COLORS[1]["rank1"], '"capital" / "of in capital of"', "w = 22.0"),
        (12.5, STAGE_COLORS[1]["rank7"], '"Dallas"', "w = 11.3"),
    ]

    for x_ctr, clr, feat_label, w_label in ffl_s1:
        # Embedding (target) — gray
        _draw_node_box(ax, x_ctr, emb_y, "embedding", "#777777",
                       width=2.6, height=0.65, fontsize=9)
        # L1 mediator
        l1_label = f'{feat_label.split("/")[0].strip()} (L1)'
        _draw_node_box(ax, x_ctr, l1_y, l1_label, clr,
                       width=2.8, height=0.65, fontsize=9)
        # L2 regulator
        if "/" in feat_label:
            l2_text = feat_label.split("/")[1].strip().strip('"')
            l2_label = f'"{l2_text}" (L2)'
        else:
            l2_label = f'{feat_label} (L2)'
        _draw_node_box(ax, x_ctr, l2_y, l2_label, clr,
                       width=2.8, height=0.65, fontsize=9)

        # Role annotations
        ax.text(x_ctr + 1.55, l2_y, "R", fontsize=8, ha="left", va="center",
                color=clr, fontweight="bold", zorder=6)
        ax.text(x_ctr + 1.55, l1_y, "M", fontsize=8, ha="left", va="center",
                color=clr, fontweight="bold", zorder=6)
        ax.text(x_ctr + 1.55, emb_y, "T", fontsize=8, ha="left", va="center",
                color="#555555", fontweight="bold", zorder=6)

        # Weight label (beside the shortcut arrow, offset from M label)
        ax.text(x_ctr + 1.1, (l2_y + l1_y) / 2 + 0.15, w_label, fontsize=8,
                ha="left", va="center", color="#555555", fontstyle="italic",
                zorder=6)

        # Arrows: R->M (straight down), M->T (straight down), R->T (curved shortcut)
        _draw_ffl_arrow(ax, x_ctr, l2_y - 0.35, x_ctr, l1_y + 0.35,
                        clr, rad=0.0, linewidth=2.0)
        _draw_ffl_arrow(ax, x_ctr, l1_y - 0.35, x_ctr, emb_y + 0.35,
                        clr, rad=0.0, linewidth=2.0)
        _draw_ffl_arrow(ax, x_ctr + 0.5, l2_y - 0.35, x_ctr + 0.5, emb_y + 0.35,
                        clr, rad=0.25, linewidth=1.5)

    # =======================================================================
    # Stage 2: Entity Resolution — one FFL bridging L10 to L15/L16
    # =======================================================================
    s2_target_y = 6.2
    s2_med_y = 7.4
    s2_reg_y = 8.5

    _draw_node_box(ax, 7.5, s2_target_y, '"Austin/Texas" (L10)',
                   STAGE_COLORS[2]["rank11"], width=3.4, height=0.7, fontsize=10)
    _draw_node_box(ax, 3.5, s2_med_y, '"say Austin" (L15)',
                   STAGE_COLORS[2]["rank11"], width=3.2, height=0.65, fontsize=9.5)
    _draw_node_box(ax, 11.5, s2_reg_y, '"say Austin" (L16)',
                   STAGE_COLORS[2]["rank11"], width=3.2, height=0.65, fontsize=9.5)

    # Roles
    ax.text(13.3, s2_reg_y, "R", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[2]["rank11"], fontweight="bold", zorder=6)
    ax.text(5.3, s2_med_y, "M", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[2]["rank11"], fontweight="bold", zorder=6)
    ax.text(9.4, s2_target_y, "T", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[2]["rank11"], fontweight="bold", zorder=6)
    ax.text(7.5, s2_target_y - 0.6, "w \u2248 8", fontsize=8, ha="center",
            color="#555555", fontstyle="italic", zorder=6)

    # Arrows: R(L16)->M(L15), M(L15)->T(L10), R(L16)->T(L10)
    _draw_ffl_arrow(ax, 10.0, s2_reg_y - 0.15, 5.2, s2_med_y + 0.15,
                    STAGE_COLORS[2]["rank11"], rad=0.08, linewidth=2.0)
    _draw_ffl_arrow(ax, 4.5, s2_med_y - 0.35, 6.5, s2_target_y + 0.38,
                    STAGE_COLORS[2]["rank11"], rad=0.12, linewidth=2.0)
    _draw_ffl_arrow(ax, 10.5, s2_reg_y - 0.35, 8.5, s2_target_y + 0.38,
                    STAGE_COLORS[2]["rank11"], rad=-0.12, linewidth=1.5)

    # =======================================================================
    # Stage 3: Output Competition — two interleaved FFLs
    # Left: "say a capital" self-chain with inhibitory shortcut (rank 3)
    # Right: "say Austin" -> "say a capital" cross-talk (rank 4)
    # =======================================================================
    s3_target_y = 10.8
    s3_med_y = 12.1
    s3_reg_y = 13.6

    # Shared target: "say a capital" (L14) — convergence hub, drawn LARGER
    _draw_node_box(ax, 7.5, s3_target_y, '"say a capital" (L14)', "#b07cc8",
                   width=4.0, height=0.85, fontsize=11, bold=True, linewidth=3.0)
    ax.annotate(
        "Convergence hub\nTarget in 5 FFLs",
        xy=(7.5, s3_target_y - 0.43), xytext=(7.5, s3_target_y - 1.5),
        fontsize=9, ha="center", color="#984ea3", fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#984ea3", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#984ea3", alpha=0.9, linewidth=1.0),
        zorder=7,
    )

    # Rank 3 (purple): say a capital L15 + L16
    _draw_node_box(ax, 3.5, s3_med_y, '"say a capital" (L15)',
                   STAGE_COLORS[3]["rank3"], width=3.2, height=0.65, fontsize=9.5)
    _draw_node_box(ax, 3.5, s3_reg_y, '"say a capital" (L16)',
                   STAGE_COLORS[3]["rank3"], width=3.2, height=0.65, fontsize=9.5)

    # Rank 4 (pink): say Austin L15 + L16
    _draw_node_box(ax, 11.5, s3_med_y, '"say Austin" (L15)',
                   STAGE_COLORS[3]["rank4"], width=3.0, height=0.65, fontsize=9.5)
    _draw_node_box(ax, 11.5, s3_reg_y, '"say Austin" (L16)',
                   STAGE_COLORS[3]["rank4"], width=3.0, height=0.65, fontsize=9.5)

    # Role labels
    ax.text(5.3, s3_reg_y, "R", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[3]["rank3"], fontweight="bold", zorder=6)
    ax.text(5.3, s3_med_y, "M", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[3]["rank3"], fontweight="bold", zorder=6)
    ax.text(13.2, s3_reg_y, "R", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[3]["rank4"], fontweight="bold", zorder=6)
    ax.text(13.2, s3_med_y, "M", fontsize=8, ha="left", va="center",
            color=STAGE_COLORS[3]["rank4"], fontweight="bold", zorder=6)
    ax.text(9.7, s3_target_y, "T", fontsize=8, ha="left", va="center",
            color="#984ea3", fontweight="bold", zorder=6)

    # Weight labels
    ax.text(1.5, (s3_reg_y + s3_target_y) / 2, "w = 17.1", fontsize=8,
            color="#555555", fontstyle="italic", ha="center", zorder=6)
    ax.text(13.5, (s3_reg_y + s3_target_y) / 2, "w = 14.2", fontsize=8,
            color="#555555", fontstyle="italic", ha="center", zorder=6)

    # --- Rank 3 arrows (purple): say a capital L16 -> L15 -> L14 ---
    # R(L16) -> M(L15)
    _draw_ffl_arrow(ax, 3.5, s3_reg_y - 0.35, 3.5, s3_med_y + 0.35,
                    STAGE_COLORS[3]["rank3"], rad=0.0, linewidth=2.5)
    # M(L15) -> T(L14)
    _draw_ffl_arrow(ax, 4.5, s3_med_y - 0.35, 6.2, s3_target_y + 0.45,
                    STAGE_COLORS[3]["rank3"], rad=0.05, linewidth=2.5)
    # R(L16) -> T(L14) — INHIBITORY (dashed red)
    _draw_ffl_arrow(ax, 2.2, s3_reg_y - 0.35, 5.8, s3_target_y + 0.45,
                    "#d62728", linestyle="--", rad=0.15, linewidth=2.5)

    # Inhibitory label
    ax.text(2.0, (s3_reg_y + s3_target_y) / 2 + 0.3, "inhibitory\nshortcut",
            fontsize=8, color="#d62728", fontweight="bold", fontstyle="italic",
            ha="center", va="center", rotation=55, zorder=6)

    # --- Rank 4 arrows (pink): say Austin L16 -> L15 -> say a capital L14 ---
    # R(L16) -> M(L15)
    _draw_ffl_arrow(ax, 11.5, s3_reg_y - 0.35, 11.5, s3_med_y + 0.35,
                    STAGE_COLORS[3]["rank4"], rad=0.0, linewidth=2.5)
    # M(L15) -> T(L14)
    _draw_ffl_arrow(ax, 10.5, s3_med_y - 0.35, 8.8, s3_target_y + 0.45,
                    STAGE_COLORS[3]["rank4"], rad=-0.05, linewidth=2.5)
    # R(L16) -> T(L14)
    _draw_ffl_arrow(ax, 12.8, s3_reg_y - 0.35, 9.2, s3_target_y + 0.45,
                    STAGE_COLORS[3]["rank4"], rad=-0.15, linewidth=1.8)

    # --- Legend (positioned below Stage 1, centered) ---
    legend_elements = [
        mpatches.Patch(facecolor=STAGE_COLORS[1]["bg"], edgecolor="#2166ac",
                       linewidth=1.0, label="Stage 1: Grounding (L1\u2013L2)"),
        mpatches.Patch(facecolor=STAGE_COLORS[2]["bg"], edgecolor="#4daf4a",
                       linewidth=1.0, label="Stage 2: Entity Resolution (L10\u2013L16)"),
        mpatches.Patch(facecolor=STAGE_COLORS[3]["bg"], edgecolor="#984ea3",
                       linewidth=1.0, label="Stage 3: Output Competition (L14\u2013L16)"),
        plt.Line2D([0], [0], color="#d62728", linewidth=2.5, linestyle="--",
                   label="Inhibitory edge"),
        plt.Line2D([0], [0], color="white", linewidth=0, label=""),  # spacer
        plt.Line2D([0], [0], color="white", linewidth=0,
                   label="R = Regulator   M = Mediator   T = Target"),
    ]
    ax.legend(handles=legend_elements, loc="lower center", fontsize=9,
              framealpha=0.92, edgecolor="#cccccc", fancybox=True,
              bbox_to_anchor=(0.5, -0.06))

    fig.suptitle(
        "FFL Cascade in Multi-Hop Reasoning\n"
        '"Fact: the capital of the state containing Dallas is _"',
        fontsize=14, fontweight="bold", y=0.98, color="#111111",
    )

    OUTPUT_SCHEMATIC.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_SCHEMATIC, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved schematic to {OUTPUT_SCHEMATIC}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Real graph with FFL overlays
# ---------------------------------------------------------------------------


def generate_graph_figure(g, key_ffls):
    """Generate the attribution graph overlay as a standalone figure."""

    nxg = _igraph_to_networkx(g)
    pos, sorted_layers, layer_labels = _compute_neuronpedia_layout(g)

    has_ft = "feature_type" in g.vs.attributes()
    has_sign = "sign" in g.es.attributes() if g.ecount() > 0 else False
    has_weight = "weight" in g.es.attributes() if g.ecount() > 0 else False
    has_ctx = "ctx_idx" in g.vs.attributes()
    has_clerp = "clerp" in g.vs.attributes()

    # Collect all motif nodes and edges
    all_motif_nodes: dict[int, list[dict]] = {}
    all_motif_edges: set[tuple[int, int]] = set()
    for ffl_info in key_ffls:
        inst = ffl_info["instance"]
        for nid in inst.node_indices:
            all_motif_nodes.setdefault(nid, []).append(ffl_info)
        for e in inst.subgraph_edges:
            all_motif_edges.add(e)

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    ax.set_facecolor("white")

    # --- Context edges (very faint) ---
    all_edges = list(nxg.edges)
    non_motif_edges = [e for e in all_edges if e not in all_motif_edges]
    if non_motif_edges:
        nx.draw_networkx_edges(
            nxg, pos, edgelist=non_motif_edges, alpha=0.35,
            edge_color="#888888", arrows=True, arrowsize=4,
            connectionstyle="arc3,rad=0.05", ax=ax, node_size=18,
        )

    # --- Context nodes (very faint) ---
    emb_nodes = [v.index for v in g.vs if has_ft and v["feature_type"] == "embedding"]
    logit_nodes = [v.index for v in g.vs if has_ft and v["feature_type"] == "logit"]
    feature_nodes = [v.index for v in g.vs
                     if v.index not in set(emb_nodes + logit_nodes)]

    ctx_emb = [n for n in emb_nodes if n not in all_motif_nodes]
    ctx_logit = [n for n in logit_nodes if n not in all_motif_nodes]
    ctx_feature = [n for n in feature_nodes if n not in all_motif_nodes]

    if ctx_emb:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_emb, node_size=30,
                               node_color="#7bbbd4", edgecolors="#2171a5",
                               linewidths=0.8, node_shape="s", alpha=0.75, ax=ax)
    if ctx_feature:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_feature, node_size=22,
                               node_color="#999999", edgecolors="#444444",
                               linewidths=0.6, node_shape="o", alpha=0.65, ax=ax)
    if ctx_logit:
        nx.draw_networkx_nodes(nxg, pos, nodelist=ctx_logit, node_size=40,
                               node_color="#8fd4a0", edgecolors="#2d8b41",
                               linewidths=0.9, node_shape="p", alpha=0.75, ax=ax)

    # --- Motif edges (thick, colored by FFL) ---
    for ffl_info in key_ffls:
        inst = ffl_info["instance"]
        color = ffl_info["color"]

        for u, v in inst.subgraph_edges:
            eid = g.get_eid(u, v, error=False)
            ls = "-"
            edge_color = color
            if eid >= 0 and has_sign:
                if g.es[eid]["sign"] == "inhibitory":
                    ls = "--"
                    edge_color = "#d62728"

            width = 2.5
            if eid >= 0 and has_weight:
                w = g.es[eid]["weight"]
                width = 1.5 + 2.0 * (w / inst.total_weight)

            ec = nx.draw_networkx_edges(
                nxg, pos, edgelist=[(u, v)], edge_color=edge_color,
                width=width, arrows=True, arrowsize=10, alpha=0.85,
                connectionstyle="arc3,rad=0.08", ax=ax, node_size=120,
                style=ls,
            )
            if ec:
                for artist in (ec if hasattr(ec, '__iter__') else [ec]):
                    artist.set_zorder(5)

    # --- Motif nodes (colored, sized by participation count) ---
    for nid, ffl_list in all_motif_nodes.items():
        primary_color = ffl_list[0]["color"]
        n_ffls = len(ffl_list)
        node_size = 120 + 50 * n_ffls

        ft = g.vs[nid]["feature_type"] if has_ft else ""
        shape = "s" if ft == "embedding" else ("p" if ft == "logit" else "o")
        lw = 2.5 if n_ffls > 1 else 1.5

        coll = nx.draw_networkx_nodes(
            nxg, pos, nodelist=[nid], node_size=node_size,
            node_color=primary_color, edgecolors="black", linewidths=lw,
            node_shape=shape, ax=ax,
        )
        if coll:
            coll.set_zorder(6)

    # --- Clerp labels on motif nodes ---
    labeled_positions: list[tuple[float, float]] = []
    offset_options = [(20, 16), (-20, 16), (20, -16), (-20, -16),
                      (24, 0), (-24, 0), (0, 20), (0, -20)]

    for nid, ffl_list in all_motif_nodes.items():
        if not has_clerp:
            continue
        clerp = g.vs[nid]["clerp"]
        if not clerp:
            continue

        clerp_short = clerp if len(clerp) <= 35 else clerp[:32] + "\u2026"
        layer_str = f"L{g.vs[nid]['layer']}" if g.vs[nid]["layer"] >= 0 else "Emb"

        n_ffls = len(ffl_list)
        if n_ffls > 1:
            display = f"{clerp_short}\n({layer_str}) [in {n_ffls} FFLs]"
        else:
            display = f"{clerp_short}\n({layer_str})"

        x, y = pos[nid]
        primary_color = ffl_list[0]["color"]

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
            fontsize=12, ha=ha,
            va="bottom" if best_oy > 0 else "top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor=primary_color, alpha=0.92, linewidth=1.2),
            arrowprops=dict(arrowstyle="-|>", color=primary_color, lw=0.8),
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
    ax.set_yticklabels(layer_labels, fontsize=13, color="#333333",
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
        ax.set_xticklabels(token_tick_labels, fontsize=13, rotation=45,
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
    for rank, stage, ckey in KEY_FFLS:
        color = STAGE_COLORS[stage][ckey]
        sname = STAGE_NAMES[stage].split(":")[1].strip()
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                       markeredgecolor="black", markersize=8, linewidth=0,
                       label=f"FFL #{rank} ({sname})")
        )
    legend_handles.append(
        plt.Line2D([0], [0], color="#d62728", linewidth=2.5, linestyle="--",
                   label="Inhibitory edge")
    )
    ax.legend(handles=legend_handles, loc="upper left", fontsize=12,
              framealpha=0.92, edgecolor="#cccccc", fancybox=True)

    prompt = g["prompt"] if "prompt" in g.attributes() else ""
    prompt_display = prompt if len(prompt) <= 60 else prompt[:57] + "..."
    ax.set_title(
        f'Attribution Graph with FFL Overlays\n"{prompt_display}"'
        f'    (edge weight threshold = {WEIGHT_THRESHOLD})',
        fontsize=18, fontweight="bold", loc="left", pad=10,
    )

    plt.tight_layout()
    OUTPUT_GRAPH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_GRAPH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved graph to {OUTPUT_GRAPH}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_figures():
    """Generate both figures."""
    print("Loading data...")
    g, instances = load_data()
    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"  FFL instances found: {len(instances)}")

    key_ffls = select_key_ffls(g, instances)
    print(f"  Selected {len(key_ffls)} key FFLs\n")

    print("Generating schematic figure...")
    generate_schematic()

    print("\nGenerating graph overlay figure...")
    generate_graph_figure(g, key_ffls)

    print("\nDone!")


if __name__ == "__main__":
    generate_figures()
