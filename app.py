"""Streamlit Attribution Graph Motif Explorer.

Interactive exploration of LLM attribution graphs with motif highlighting.
Uses Plotly for interactive visualization (hover tooltips, zoom/pan) and
reuses all existing src/ modules.

Run with: streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.graph_loader import load_attribution_graph, graph_summary
from src.motif_census import (
    TRIAD_LABELS,
    CONNECTED_TRIAD_INDICES,
    MOTIF_FFL,
    MOTIF_CHAIN,
    MOTIF_FAN_IN,
    MOTIF_FAN_OUT,
    MOTIF_CYCLE,
    MOTIF_COMPLETE,
    MotifInstance,
    find_motif_instances,
)
from src.visualization import ROLE_COLORS, _compute_neuronpedia_layout
from src.pipeline import discover_graphs

import igraph as ig

# --- Constants ---

DATA_DIR = Path("data/raw")
RESULTS_DIR = Path("data/results")
ANALYSIS_SUMMARY_PATH = RESULTS_DIR / "analysis_summary.json"

MOTIF_OPTIONS: dict[str, int] = {
    "030T (FFL — Feedforward Loop)": MOTIF_FFL,
    "021C (Chain)": MOTIF_CHAIN,
    "021U (Fan-in)": MOTIF_FAN_IN,
    "021D (Fan-out)": MOTIF_FAN_OUT,
    "030C (Cycle)": MOTIF_CYCLE,
    "300 (Complete)": MOTIF_COMPLETE,
}

# Plotly marker symbols matching node types
_SYMBOL_MAP = {
    "embedding": "square",
    "cross layer transcoder": "circle",
    "logit": "pentagon",
}

# Context node border colors by type
_TYPE_BORDER = {
    "embedding": "#3182bd",
    "cross layer transcoder": "#666666",
    "logit": "#41ab5d",
}


# --- Caching ---

@st.cache_data(show_spinner=False)
def cached_discover_graphs(data_dir: str) -> dict[str, list[str]]:
    """Discover graphs and return serializable path strings."""
    raw = discover_graphs(data_dir)
    return {cat: [str(p) for p in paths] for cat, paths in raw.items()}


@st.cache_data(show_spinner=False)
def cached_load_analysis_summary(path: str) -> dict[str, Any] | None:
    """Load the pre-computed analysis summary JSON."""
    p = Path(path)
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading graph...")
def cached_load_graph(_graph_key: str, path: str, threshold: float) -> ig.Graph:
    """Load an igraph Graph object. _graph_key is used for cache identity."""
    return load_attribution_graph(path, weight_threshold=threshold)


@st.cache_data(show_spinner="Finding motif instances...")
def cached_find_instances(
    _graph_key: str,
    path: str,
    threshold: float,
    motif_isoclass: int,
) -> list[dict[str, Any]]:
    """Find motif instances and serialize to dicts for caching."""
    g = load_attribution_graph(path, weight_threshold=threshold)
    instances = find_motif_instances(
        g, motif_isoclass=motif_isoclass, size=3,
        max_instances=50, sort_by="weight",
    )
    # Serialize MotifInstance to dict for st.cache_data
    return [
        {
            "isoclass": inst.isoclass,
            "label": inst.label,
            "node_indices": list(inst.node_indices),
            "node_roles": inst.node_roles,
            "subgraph_edges": inst.subgraph_edges,
            "total_weight": inst.total_weight,
        }
        for inst in instances
    ]


def _dict_to_instance(d: dict[str, Any]) -> MotifInstance:
    """Reconstruct a MotifInstance from a serialized dict."""
    return MotifInstance(
        isoclass=d["isoclass"],
        label=d["label"],
        node_indices=tuple(d["node_indices"]),
        node_roles=d["node_roles"],
        subgraph_edges=[tuple(e) for e in d["subgraph_edges"]],
        total_weight=d["total_weight"],
    )


# --- Plotly graph builder ---

def build_plotly_graph(
    graph: ig.Graph,
    motif_instance: MotifInstance | None = None,
) -> go.Figure:
    """Build an interactive Plotly figure of the attribution graph.

    Neuronpedia-style grid layout: x=token position, y=layer.
    Context edges/nodes are semi-transparent; motif elements are highlighted.
    """
    pos, sorted_layers, layer_labels = _compute_neuronpedia_layout(graph)

    has_ft = "feature_type" in graph.vs.attributes()
    has_clerp = "clerp" in graph.vs.attributes()
    has_sign = "sign" in graph.es.attributes() if graph.ecount() > 0 else False
    has_weight = "weight" in graph.es.attributes() if graph.ecount() > 0 else False
    has_activation = "activation" in graph.vs.attributes()
    prompt_tokens = graph["prompt_tokens"] if "prompt_tokens" in graph.attributes() else []

    motif_nodes = set(motif_instance.node_indices) if motif_instance else set()
    motif_edges = set(motif_instance.subgraph_edges) if motif_instance else set()

    fig = go.Figure()

    # --- 1. Context edge traces (None-separator technique) ---
    exc_x, exc_y = [], []
    inh_x, inh_y = [], []

    for e in graph.es:
        src, tgt = e.source, e.target
        if (src, tgt) in motif_edges:
            continue
        if src not in pos or tgt not in pos:
            continue

        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        sign = e["sign"] if has_sign else "excitatory"

        if sign == "excitatory":
            exc_x.extend([x0, x1, None])
            exc_y.extend([y0, y1, None])
        else:
            inh_x.extend([x0, x1, None])
            inh_y.extend([y0, y1, None])

    if exc_x:
        fig.add_trace(go.Scatter(
            x=exc_x, y=exc_y, mode="lines",
            line=dict(color="rgba(44,160,44,0.15)", width=0.5),
            hoverinfo="skip", showlegend=True, name="Excitatory edge",
        ))
    if inh_x:
        fig.add_trace(go.Scatter(
            x=inh_x, y=inh_y, mode="lines",
            line=dict(color="rgba(214,39,40,0.15)", width=0.5),
            hoverinfo="skip", showlegend=True, name="Inhibitory edge",
        ))

    # --- 2. Context node traces (by type) ---
    for ft_key, symbol, border_color, trace_name in [
        ("embedding", "square", "#3182bd", "Embedding"),
        ("cross layer transcoder", "circle", "#666666", "Feature"),
        ("logit", "pentagon", "#41ab5d", "Logit"),
    ]:
        nx_list, ny_list, hover_list = [], [], []
        for v in graph.vs:
            if v.index in motif_nodes:
                continue
            ft = v["feature_type"] if has_ft else ""
            if ft != ft_key:
                continue
            if v.index not in pos:
                continue

            x, y = pos[v.index]
            nx_list.append(x)
            ny_list.append(y)

            # Build hover text
            clerp = v["clerp"] if has_clerp else ""
            layer = v["layer"] if "layer" in graph.vs.attributes() else "?"
            act = v["activation"] if has_activation else None
            act_str = f"{act:.3f}" if act is not None else "N/A"
            hover = (
                f"<b>{clerp}</b><br>"
                f"Type: {ft}<br>"
                f"Layer: {layer}<br>"
                f"Activation: {act_str}"
            )
            hover_list.append(hover)

        if nx_list:
            fig.add_trace(go.Scatter(
                x=nx_list, y=ny_list, mode="markers",
                marker=dict(
                    symbol=symbol, size=6,
                    color="white", opacity=0.45,
                    line=dict(color=border_color, width=1),
                ),
                text=hover_list, hoverinfo="text",
                showlegend=True, name=trace_name,
            ))

    # --- 3. Motif edges with midpoint arrow markers ---
    if motif_instance:
        arrow_x, arrow_y, arrow_angles, arrow_colors, arrow_hover = [], [], [], [], []

        for src, tgt in motif_instance.subgraph_edges:
            if src not in pos or tgt not in pos:
                continue
            eid = graph.get_eid(src, tgt, error=False)
            if eid == -1:
                edge_color = "#333333"
                sign = "?"
                weight_str = "N/A"
            else:
                sign = graph.es[eid]["sign"] if has_sign else "excitatory"
                edge_color = "#2ca02c" if sign == "excitatory" else "#d62728"
                raw_w = graph.es[eid]["raw_weight"] if "raw_weight" in graph.es.attributes() else 0
                weight_str = f"{raw_w:+.4f}"

            x0, y0 = pos[src]
            x1, y1 = pos[tgt]

            # Thick motif edge line
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color=edge_color, width=3),
                hoverinfo="skip", showlegend=False,
            ))

            # Collect arrow marker at 65% along the edge
            t = 0.65
            ax = x0 + t * (x1 - x0)
            ay = y0 + t * (y1 - y0)
            arrow_x.append(ax)
            arrow_y.append(ay)

            dx, dy = x1 - x0, y1 - y0
            theta_deg = np.degrees(np.arctan2(dy, dx))
            arrow_angles.append(90 - theta_deg)
            arrow_colors.append(edge_color)

            src_clerp = graph.vs[src]["clerp"] if has_clerp else str(src)
            tgt_clerp = graph.vs[tgt]["clerp"] if has_clerp else str(tgt)
            arrow_hover.append(
                f"<b>{src_clerp}</b> → <b>{tgt_clerp}</b><br>"
                f"Sign: {sign}<br>Weight: {weight_str}"
            )

        # Single trace for all arrow markers
        if arrow_x:
            fig.add_trace(go.Scatter(
                x=arrow_x, y=arrow_y, mode="markers",
                marker=dict(
                    symbol="triangle-up", size=14,
                    color=arrow_colors, angle=arrow_angles,
                    line=dict(color="black", width=1),
                ),
                text=arrow_hover, hoverinfo="text",
                showlegend=False,
            ))

    # --- 4. Motif node trace ---
    if motif_instance:
        mx, my, mcolors, mhover, msymbols = [], [], [], [], []
        for node_idx in motif_instance.node_indices:
            if node_idx not in pos:
                continue
            x, y = pos[node_idx]
            mx.append(x)
            my.append(y)

            role = motif_instance.node_roles.get(node_idx, "node_a")
            mcolors.append(ROLE_COLORS.get(role, "#00d4aa"))

            ft = graph.vs[node_idx]["feature_type"] if has_ft else ""
            msymbols.append(_SYMBOL_MAP.get(ft, "circle"))

            clerp = graph.vs[node_idx]["clerp"] if has_clerp else ""
            layer = graph.vs[node_idx]["layer"] if "layer" in graph.vs.attributes() else "?"
            act = graph.vs[node_idx]["activation"] if has_activation else None
            act_str = f"{act:.3f}" if act is not None else "N/A"
            hover = (
                f"<b>{clerp}</b><br>"
                f"Role: {role}<br>"
                f"Type: {ft}<br>"
                f"Layer: {layer}<br>"
                f"Activation: {act_str}"
            )
            mhover.append(hover)

        if mx:
            fig.add_trace(go.Scatter(
                x=mx, y=my, mode="markers",
                marker=dict(
                    symbol=msymbols, size=18,
                    color=mcolors, opacity=1.0,
                    line=dict(color="black", width=2),
                ),
                text=mhover, hoverinfo="text",
                showlegend=True, name="Motif node",
            ))

    # --- 5. Motif label annotations ---
    if motif_instance and has_clerp:
        offsets = [(25, 20), (-25, 20), (25, -20)]
        for i, node_idx in enumerate(motif_instance.node_indices):
            if node_idx not in pos:
                continue
            clerp = graph.vs[node_idx]["clerp"] if has_clerp else ""
            if not clerp:
                continue
            if len(clerp) > 40:
                clerp = clerp[:37] + "..."

            role = motif_instance.node_roles.get(node_idx, "node_a")
            role_label = role.replace("_", " ")
            display = f"{clerp} ({role_label})"
            role_color = ROLE_COLORS.get(role, "#00d4aa")

            x, y = pos[node_idx]
            ox, oy = offsets[i % len(offsets)]

            fig.add_annotation(
                x=x, y=y,
                text=display,
                showarrow=True, arrowhead=2,
                ax=ox, ay=-oy,  # negative because Plotly y-offset is inverted
                font=dict(size=10, color="black"),
                bgcolor="white",
                bordercolor=role_color,
                borderwidth=2, borderpad=4,
                opacity=0.95,
            )

    # --- Layout ---
    # X-axis: token labels
    has_ctx = "ctx_idx" in graph.vs.attributes()
    if has_ctx and prompt_tokens:
        all_ctx = [v["ctx_idx"] for v in graph.vs]
        min_ctx, max_ctx = min(all_ctx), max(all_ctx)
        tick_vals = list(range(min_ctx, max_ctx + 1))
        tick_text = []
        for ctx in tick_vals:
            if ctx < len(prompt_tokens):
                tick_text.append(prompt_tokens[ctx].replace("\u2191", "^"))
            else:
                tick_text.append(f"[{ctx}]")
    else:
        tick_vals, tick_text = [], []

    # Y-axis: layer labels
    layer_y_vals = list(range(len(sorted_layers)))
    layer_y_text = layer_labels

    fig.update_layout(
        title=None,
        xaxis=dict(
            tickmode="array",
            tickvals=[v * 1.0 for v in tick_vals],
            ticktext=tick_text,
            tickangle=45,
            side="bottom",
            showgrid=True, gridcolor="#f0f0f0", gridwidth=1,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[v * 1.0 for v in layer_y_vals],
            ticktext=layer_y_text,
            showgrid=True, gridcolor="#f0f0f0", gridwidth=1,
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=700,
        margin=dict(l=60, r=30, t=30, b=80),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
        hovermode="closest",
    )

    return fig


# --- Z-score bar chart (Plotly) ---

def build_zscore_bar(graph_name: str, summary_data: dict[str, Any]) -> go.Figure | None:
    """Build a Plotly Z-score bar chart from pre-computed analysis summary."""
    if summary_data is None:
        return None

    # Find matching graph entry
    entry = None
    for g in summary_data.get("graphs", []):
        if g["name"] == graph_name:
            entry = g
            break
    if entry is None:
        return None

    z_scores = entry["z_scores"]
    labels = list(z_scores.keys())
    values = list(z_scores.values())

    colors = [
        "#d62728" if z > 2.0 else "#1f77b4" if z < -2.0 else "#7f7f7f"
        for z in values
    ]

    fig = go.Figure(data=[
        go.Bar(
            x=labels, y=values,
            marker_color=colors,
            marker_line_color="black",
            marker_line_width=0.5,
        )
    ])

    fig.add_hline(y=2.0, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=-2.0, line_dash="dash", line_color="red", opacity=0.5)
    fig.add_hline(y=0, line_color="black", line_width=0.5)

    fig.update_layout(
        title=f"Z-Score Profile: {graph_name}",
        xaxis_title="Triad Class",
        yaxis_title="Z-score",
        plot_bgcolor="white",
        height=350,
        margin=dict(l=50, r=20, t=40, b=60),
    )

    return fig


# --- Streamlit App ---

def main() -> None:
    st.set_page_config(
        page_title="Attribution Graph Motif Explorer",
        layout="wide",
    )

    st.title("Attribution Graph Motif Explorer")
    st.caption("Interactive exploration of LLM attribution graphs with network motif highlighting")

    # Load analysis summary once
    summary_data = cached_load_analysis_summary(str(ANALYSIS_SUMMARY_PATH))

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")

        # Discover graphs
        if not DATA_DIR.exists():
            st.error(f"Data directory not found: {DATA_DIR}")
            st.stop()

        categories = cached_discover_graphs(str(DATA_DIR))
        if not categories:
            st.error("No graph categories found in data/raw/")
            st.stop()

        # Task category
        category = st.selectbox(
            "Task Category",
            options=sorted(categories.keys()),
        )

        # Graph selection
        graph_paths = categories[category]
        graph_names = [Path(p).stem for p in graph_paths]
        graph_idx = st.selectbox(
            "Graph",
            options=range(len(graph_names)),
            format_func=lambda i: graph_names[i],
        )
        selected_path = graph_paths[graph_idx]
        selected_name = graph_names[graph_idx]

        # Weight threshold
        weight_threshold = st.slider(
            "Edge Weight Threshold",
            min_value=0.0, max_value=5.0, value=0.0, step=0.1,
            help="Minimum absolute edge weight to include. Higher values prune weak edges.",
        )

        st.divider()

        # Motif type
        motif_label = st.selectbox(
            "Motif Type",
            options=list(MOTIF_OPTIONS.keys()),
        )
        motif_isoclass = MOTIF_OPTIONS[motif_label]

    # --- Load graph ---
    graph_key = f"{selected_path}|{weight_threshold}"
    graph = cached_load_graph(graph_key, selected_path, weight_threshold)
    stats = graph_summary(graph)

    # --- Find motif instances ---
    instances_data = cached_find_instances(
        graph_key, selected_path, weight_threshold, motif_isoclass,
    )
    n_instances = len(instances_data)

    # Instance rank selector in sidebar (after we know count)
    with st.sidebar:
        if n_instances > 0:
            instance_rank = st.slider(
                "Instance Rank",
                min_value=0, max_value=n_instances - 1, value=0,
                help="0 = highest total edge weight",
            )
        else:
            instance_rank = 0
            st.info(f"No {motif_label.split('(')[0].strip()} instances found in this graph.")

    # --- Header metrics ---
    prompt = stats.get("prompt", "")
    model = stats.get("model", "")

    st.markdown(f"**Prompt:** `{prompt}`" if prompt else "**Prompt:** N/A")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Nodes", stats["n_nodes"])
    col2.metric("Edges", stats["n_edges"])
    col3.metric("Density", f"{stats['density']:.4f}")
    col4.metric("Model", model if model else "N/A")
    col5.metric(f"{motif_label.split('(')[0].strip()} Instances", n_instances)

    # --- Main graph visualization ---
    if n_instances > 0:
        inst_dict = instances_data[instance_rank]
        motif_instance = _dict_to_instance(inst_dict)
        st.markdown(
            f"**Showing instance #{instance_rank}** — "
            f"Label: `{motif_instance.label}` — "
            f"Total weight: `{motif_instance.total_weight:.3f}`"
        )
    else:
        motif_instance = None
        st.markdown("**No motif instance selected** — showing graph context only")

    fig = build_plotly_graph(graph, motif_instance)
    st.plotly_chart(fig, use_container_width=True, config={
        "edits": {"annotationPosition": True, "annotationTail": True},
    })

    # --- Motif Instance Details ---
    if motif_instance:
        st.subheader("Motif Instance Details")

        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.markdown("**Nodes**")
            node_rows = []
            for node_idx in motif_instance.node_indices:
                v = graph.vs[node_idx]
                role = motif_instance.node_roles.get(node_idx, "?")
                clerp = v["clerp"] if "clerp" in graph.vs.attributes() else ""
                layer = v["layer"] if "layer" in graph.vs.attributes() else "?"
                ctx = v["ctx_idx"] if "ctx_idx" in graph.vs.attributes() else "?"
                ft = v["feature_type"] if "feature_type" in graph.vs.attributes() else ""
                feat = v["feature"] if "feature" in graph.vs.attributes() else ""
                node_rows.append({
                    "Role": role,
                    "Clerp": clerp[:60] if clerp else "",
                    "Layer": layer,
                    "Token Pos": ctx,
                    "Type": ft,
                    "Feature": feat if feat else "",
                })
            st.dataframe(node_rows, use_container_width=True, hide_index=True)

        with detail_col2:
            st.markdown("**Edges**")
            edge_rows = []
            for src, tgt in motif_instance.subgraph_edges:
                eid = graph.get_eid(src, tgt, error=False)
                if eid == -1:
                    continue
                e = graph.es[eid]
                raw_w = e["raw_weight"] if "raw_weight" in graph.es.attributes() else 0
                sign = e["sign"] if "sign" in graph.es.attributes() else "?"

                src_clerp = graph.vs[src]["clerp"] if "clerp" in graph.vs.attributes() else str(src)
                tgt_clerp = graph.vs[tgt]["clerp"] if "clerp" in graph.vs.attributes() else str(tgt)
                if len(src_clerp) > 30:
                    src_clerp = src_clerp[:27] + "..."
                if len(tgt_clerp) > 30:
                    tgt_clerp = tgt_clerp[:27] + "..."

                edge_rows.append({
                    "Source": src_clerp,
                    "Target": tgt_clerp,
                    "Weight": f"{raw_w:+.4f}",
                    "Sign": sign,
                })
            st.dataframe(edge_rows, use_container_width=True, hide_index=True)

    # --- Z-Score Profile (expander) ---
    with st.expander("Z-Score Profile", expanded=False):
        zscore_fig = build_zscore_bar(selected_name, summary_data)
        if zscore_fig is not None:
            st.plotly_chart(zscore_fig, use_container_width=True)

            # Instance count summary from analysis_summary
            entry = None
            for g in summary_data.get("graphs", []):
                if g["name"] == selected_name:
                    entry = g
                    break
            if entry and "instance_counts" in entry:
                ic = entry["instance_counts"]
                ic_cols = st.columns(len(ic))
                for col, (k, v) in zip(ic_cols, ic.items()):
                    col.metric(k, f"{v:,}")
        else:
            st.info(
                "No pre-computed Z-scores found for this graph. "
                "Run `python -m src.pipeline` to generate analysis_summary.json."
            )


if __name__ == "__main__":
    main()
