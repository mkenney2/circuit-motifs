"""Build neuron-level attribution graphs for Gemma-2-2B.

Constructs directed graphs where nodes are raw MLP neurons (not SAE/transcoder
features) and edges are activation × gradient attributions between neurons in
different layers. The output JSON format is compatible with graph_loader.py
so the existing motif pipeline works unchanged.

Designed to run on GPU (Colab) via TransformerLens. The local analysis scripts
only need the JSON output, not this module's GPU dependencies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class NeuronGraphConfig:
    """Configuration for neuron-level graph construction.

    Attributes:
        model_name: HuggingFace model identifier.
        top_k: Number of neurons to select per layer by |activation|.
        max_layer_gap: Maximum layer distance for attribution edges.
        threshold_pct: Percentile threshold — keep top (100 - threshold_pct)%
            of attributions by magnitude.
        device: Torch device for computation.
    """
    model_name: str = "google/gemma-2-2b"
    top_k: int = 100
    max_layer_gap: int = 5
    threshold_pct: float = 95
    device: str = "cuda"


def select_top_neurons(
    model: Any,
    tokens: Any,
    config: NeuronGraphConfig,
) -> dict[int, list[tuple[int, float]]]:
    """Forward pass → capture MLP post-activations → top-k per layer by |activation|.

    Args:
        model: A TransformerLens HookedTransformer.
        tokens: Tokenized input (shape [1, seq_len]).
        config: Graph construction configuration.

    Returns:
        Dict mapping layer index to list of (neuron_index, activation_value) tuples,
        sorted by |activation| descending, limited to top_k per layer.
        Activations are taken from the last token position.
    """
    import torch

    n_layers = model.cfg.n_layers
    activations: dict[int, Any] = {}

    # Run forward pass, caching MLP post-activations
    hook_names = [f"blocks.{i}.mlp.hook_post" for i in range(n_layers)]
    _, cache = model.run_with_cache(tokens, names_filter=hook_names)

    top_neurons: dict[int, list[tuple[int, float]]] = {}

    for layer_idx in range(n_layers):
        hook_name = f"blocks.{layer_idx}.mlp.hook_post"
        # Shape: [batch, seq_len, d_mlp] — take last token
        act = cache[hook_name][0, -1, :]  # [d_mlp]
        act_np = act.detach().cpu().float().numpy()

        # Top-k by absolute activation
        abs_act = np.abs(act_np)
        if len(abs_act) <= config.top_k:
            top_indices = np.argsort(-abs_act)
        else:
            top_indices = np.argpartition(-abs_act, config.top_k)[:config.top_k]
            top_indices = top_indices[np.argsort(-abs_act[top_indices])]

        top_neurons[layer_idx] = [
            (int(idx), float(act_np[idx])) for idx in top_indices
        ]

    return top_neurons


def compute_all_attributions(
    model: Any,
    tokens: Any,
    top_neurons: dict[int, list[tuple[int, float]]],
    config: NeuronGraphConfig,
) -> list[dict]:
    """Compute activation × gradient attributions between selected neurons.

    For each target layer, runs one forward pass with hooks that keep the
    computation graph alive, then backward passes per target neuron to get
    gradients at all relevant source layers simultaneously.

    Args:
        model: A TransformerLens HookedTransformer.
        tokens: Tokenized input (shape [1, seq_len]).
        config: Graph construction configuration.

    Returns:
        List of edge dicts with keys: source_layer, source_neuron, source_act,
        target_layer, target_neuron, target_act, attribution.
    """
    import torch

    n_layers = model.cfg.n_layers

    # Collect all attribution magnitudes for thresholding
    all_attrs: list[float] = []
    raw_edges: list[dict] = []

    # Process one target layer at a time to bound memory usage.
    # Each iteration: one forward pass + K backward passes (K = target neurons).
    for tgt_layer in range(1, n_layers):
        tgt_neurons = top_neurons.get(tgt_layer, [])
        if not tgt_neurons:
            continue

        # Source layers within max_layer_gap
        min_src = max(0, tgt_layer - config.max_layer_gap)
        src_layers = [l for l in range(min_src, tgt_layer)
                      if top_neurons.get(l)]
        if not src_layers:
            continue

        # Capture activations via hooks that preserve the computation graph
        activations: dict[int, Any] = {}

        def _make_hook(layer_idx: int):
            def hook_fn(tensor, hook):
                tensor.retain_grad()
                activations[layer_idx] = tensor
                return tensor
            return hook_fn

        layers_to_hook = set(src_layers) | {tgt_layer}
        fwd_hooks = [
            (f"blocks.{l}.mlp.hook_post", _make_hook(l))
            for l in layers_to_hook
        ]

        # Forward pass — run_with_hooks keeps the computation graph alive
        model.zero_grad()
        with torch.enable_grad():
            _ = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)

        tgt_act_tensor = activations[tgt_layer]  # [1, seq, d_mlp]

        # Backward per target neuron → gradients at all source layers
        for ti, (tgt_idx, tgt_act_val) in enumerate(tgt_neurons):
            tgt_scalar = tgt_act_tensor[0, -1, tgt_idx]

            # Zero grads on all source activations
            for sl in src_layers:
                if activations[sl].grad is not None:
                    activations[sl].grad.zero_()

            tgt_scalar.backward(retain_graph=True)

            # Collect attributions from all source layers
            for sl in src_layers:
                src_act = activations[sl]
                if src_act.grad is None:
                    continue
                grad = src_act.grad[0, -1, :]  # [d_mlp]

                for src_idx, src_act_val in top_neurons[sl]:
                    g_val = grad[src_idx].item()
                    attr = float(src_act_val) * float(g_val)
                    if attr == 0.0:
                        continue

                    raw_edges.append({
                        "source_layer": sl,
                        "source_neuron": src_idx,
                        "source_act": src_act_val,
                        "target_layer": tgt_layer,
                        "target_neuron": tgt_idx,
                        "target_act": tgt_act_val,
                        "attribution": attr,
                    })
                    all_attrs.append(abs(attr))

        # Free computation graph for this target layer
        del activations, tgt_act_tensor
        if config.device == "cuda":
            torch.cuda.empty_cache()

        print(f"  target layer {tgt_layer}/{n_layers-1}: "
              f"{len(tgt_neurons)} neurons, {len(raw_edges)} edges so far",
              flush=True)

    # Threshold: keep top (100 - threshold_pct)% by |attribution|
    if all_attrs and config.threshold_pct > 0:
        threshold = float(np.percentile(all_attrs, config.threshold_pct))
        edges = [e for e in raw_edges if abs(e["attribution"]) >= threshold]
    else:
        edges = raw_edges

    return edges


def build_neuron_graph_json(
    prompt: str,
    prompt_tokens: list[str],
    slug: str,
    category: str,
    top_neurons: dict[int, list[tuple[int, float]]],
    edges: list[dict],
    config: NeuronGraphConfig,
) -> dict:
    """Build JSON matching circuit-tracer format for graph_loader.py compatibility.

    Args:
        prompt: The text prompt.
        prompt_tokens: Tokenized prompt as string list.
        slug: Graph identifier slug.
        category: Task category (e.g. "arithmetic").
        top_neurons: Per-layer selected neurons from select_top_neurons().
        edges: Attribution edges from compute_all_attributions().
        config: Graph construction configuration.

    Returns:
        Dict in circuit-tracer JSON format, ready for json.dump().
    """
    # Collect all neurons that participate in edges
    active_neurons: set[tuple[int, int]] = set()
    for e in edges:
        active_neurons.add((e["source_layer"], e["source_neuron"]))
        active_neurons.add((e["target_layer"], e["target_neuron"]))

    # Build activation lookup
    act_lookup: dict[tuple[int, int], float] = {}
    for layer_idx, neurons in top_neurons.items():
        for neuron_idx, act_val in neurons:
            act_lookup[(layer_idx, neuron_idx)] = act_val

    # Build nodes
    nodes = []
    for layer_idx, neuron_idx in sorted(active_neurons):
        node_id = f"L{layer_idx}_N{neuron_idx}"
        act_val = act_lookup.get((layer_idx, neuron_idx), 0.0)
        nodes.append({
            "node_id": node_id,
            "feature": neuron_idx,
            "layer": str(layer_idx),
            "ctx_idx": 0,
            "feature_type": "mlp_neuron",
            "clerp": f"neuron {neuron_idx} @ L{layer_idx}",
            "activation": act_val,
            "influence": 0.0,
            "is_target_logit": False,
            "token_prob": 0.0,
        })

    # Build links
    links = []
    for e in edges:
        source_id = f"L{e['source_layer']}_N{e['source_neuron']}"
        target_id = f"L{e['target_layer']}_N{e['target_neuron']}"
        links.append({
            "source": source_id,
            "target": target_id,
            "weight": e["attribution"],
        })

    graph_json = {
        "metadata": {
            "slug": slug,
            "scan": config.model_name,
            "prompt": prompt,
            "prompt_tokens": prompt_tokens,
            "schema_version": 1,
            "info": {
                "graph_type": "neuron_level",
                "category": category,
                "top_k": config.top_k,
                "max_layer_gap": config.max_layer_gap,
                "threshold_pct": config.threshold_pct,
            },
        },
        "qParams": {
            "pinnedIds": [],
            "supernodes": [],
            "linkType": "both",
        },
        "nodes": nodes,
        "links": links,
    }

    return graph_json


def generate_neuron_graph(
    model: Any,
    prompt: str,
    slug: str,
    category: str,
    output_dir: str | Path,
    config: NeuronGraphConfig | None = None,
) -> Path:
    """End-to-end: tokenize → select → attribute → build JSON → save.

    Args:
        model: A TransformerLens HookedTransformer.
        prompt: The text prompt.
        slug: Graph identifier slug.
        category: Task category (e.g. "arithmetic").
        output_dir: Directory to save the JSON file.
        config: Graph construction configuration. Uses defaults if None.

    Returns:
        Path to the saved JSON file.
    """
    if config is None:
        config = NeuronGraphConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tokenize
    tokens = model.to_tokens(prompt)
    prompt_tokens = model.to_str_tokens(prompt)
    if isinstance(prompt_tokens, list) and isinstance(prompt_tokens[0], list):
        prompt_tokens = prompt_tokens[0]
    prompt_tokens = [str(t) for t in prompt_tokens]

    # Select top neurons
    top_neurons = select_top_neurons(model, tokens, config)

    # Compute attributions
    edges = compute_all_attributions(model, tokens, top_neurons, config)

    # Build JSON
    graph_json = build_neuron_graph_json(
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        slug=slug,
        category=category,
        top_neurons=top_neurons,
        edges=edges,
        config=config,
    )

    # Save
    out_path = output_dir / f"{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(graph_json, f, indent=2)

    return out_path


def characterize_graph(graph_json: dict) -> dict[str, Any]:
    """Compute structural properties of a neuron graph JSON.

    Args:
        graph_json: Parsed JSON dict (or loaded graph JSON).

    Returns:
        Dict with n_nodes, n_edges, density, degree_gini,
        excitatory_fraction, nodes_per_layer, mean_degree.
    """
    nodes = graph_json.get("nodes", [])
    links = graph_json.get("links", [])
    n_nodes = len(nodes)
    n_edges = len(links)

    # Density
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_edges if max_edges > 0 else 0.0

    # Degree distribution (in + out)
    degree: dict[str, int] = {}
    for node in nodes:
        degree[node["node_id"]] = 0
    for link in links:
        degree[link["source"]] = degree.get(link["source"], 0) + 1
        degree[link["target"]] = degree.get(link["target"], 0) + 1

    degrees = list(degree.values())
    mean_deg = float(np.mean(degrees)) if degrees else 0.0

    # Gini coefficient of degree distribution
    degree_gini = _gini(degrees) if degrees else 0.0

    # Excitatory fraction
    n_exc = sum(1 for l in links if l.get("weight", 0) >= 0)
    exc_frac = n_exc / n_edges if n_edges > 0 else 0.0

    # Nodes per layer
    nodes_per_layer: dict[str, int] = {}
    for node in nodes:
        layer = node.get("layer", "?")
        nodes_per_layer[layer] = nodes_per_layer.get(layer, 0) + 1

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "degree_gini": degree_gini,
        "excitatory_fraction": exc_frac,
        "mean_degree": mean_deg,
        "nodes_per_layer": nodes_per_layer,
    }


def _gini(values: list[int | float]) -> float:
    """Compute the Gini coefficient of a list of values.

    Args:
        values: List of non-negative values.

    Returns:
        Gini coefficient in [0, 1]. 0 = perfect equality, 1 = perfect inequality.
    """
    if not values:
        return 0.0
    arr = np.array(values, dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) - (n + 1) * np.sum(arr)) / (n * np.sum(arr)))
