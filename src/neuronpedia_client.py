"""Client for fetching attribution graphs from Neuronpedia and Anthropic's public bucket.

Two data sources:
1. Anthropic bucket: 99 pre-published graphs from the circuit-tracing paper (Haiku/CLT/PLT).
   No auth needed. Metadata at transformer-circuits.pub, JSON downloadable directly.
2. Neuronpedia API: community-generated graphs for gemma-2-2b, qwen3-4b, gemma-3-4b-it.
   Can also generate new graphs on-demand (no GPU required locally).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

import igraph as ig

from src.graph_loader import parse_attribution_graph


# Neuronpedia API
NEURONPEDIA_API_URL = "https://www.neuronpedia.org/api"

# Anthropic public bucket for circuit-tracing paper graphs
ANTHROPIC_METADATA_URL = (
    "https://transformer-circuits.pub/2025/attribution-graphs/data/graph-metadata.json"
)
ANTHROPIC_GRAPH_URL_TEMPLATE = (
    "https://transformer-circuits.pub/2025/attribution-graphs/graph_data/{slug}.json"
)

DEFAULT_HEADERS = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "User-Agent": "network-motif-analysis/0.1.0",
}

DEFAULT_RATE_LIMIT_DELAY = 1.0  # seconds between requests

# Suggested category assignments for known Anthropic bucket graph slugs.
# These are based on the prompt content and can be overridden.
ANTHROPIC_GRAPH_CATEGORIES: dict[str, str] = {
    # ── Factual recall ──────────────────────────────────────────────
    "michael-jordan-ha": "factual_recall",
    "michael-batkin-ha": "factual_recall",
    "michael-clt-clean": "factual_recall",
    "michael-plt-clean": "factual_recall",
    "opposite_of_small": "factual_recall",
    "opposite-of-small-clt-clean": "factual_recall",
    "opposite-of-small-plt-clean": "factual_recall",
    "opposite-hot-clt-clean": "factual_recall",
    "opposite-hot-plt-clean": "factual_recall",
    "common-colors-clt-clean": "factual_recall",
    "common-colors-plt-clean": "factual_recall",
    "ndag-18l-analytics": "factual_recall",
    "ndag-18l": "factual_recall",
    "ndag-clt-clean": "factual_recall",
    "ndag-plt-clean": "factual_recall",
    "iasg-clt-clean": "factual_recall",
    "iasg-plt-clean": "factual_recall",
    "iasg-clt-18l-p95": "factual_recall",
    "iasg-clt-18l-p90": "factual_recall",
    "iasg-clt-18l-p80": "factual_recall",
    "iasg-clt-18l-p70": "factual_recall",
    "uspto-telephone-clt-18l": "factual_recall",
    "uspto-telephone-slt-18l": "factual_recall",
    "uspto-telephone-clt-clean": "factual_recall",
    "uspto-telephone-plt-clean": "factual_recall",
    "multiple-choice-qk": "factual_recall",
    "mj-18l": "factual_recall",
    # ── Multi-hop reasoning ─────────────────────────────────────────
    "capital-state-dallas": "multihop",
    "capital-state-oakland": "multihop",
    "capital-analogy-clt-clean": "multihop",
    "capital-analogy-plt-clean": "multihop",
    "capital-analogy-clt-18l": "multihop",
    "capital-analogy-slt-18l": "multihop",
    "capital-analogy-clt-18l-path-highlight": "multihop",
    "capital-analogy-slt-18l-path-highlight": "multihop",
    "currency-analogy-clt-clean": "multihop",
    "currency-analogy-plt-clean": "multihop",
    # ── Arithmetic ──────────────────────────────────────────────────
    "calc-36-plus-59": "arithmetic",
    "calc-23-plus-28": "arithmetic",
    "calc-17-plus-22": "arithmetic",
    "calc-11-plus-4": "arithmetic",
    "calc-6-plus-9": "arithmetic",
    "calc-36-plus-59-18l": "arithmetic",
    "five-plus-three-clt-clean": "arithmetic",
    "five-plus-three-plt-clean": "arithmetic",
    "count-by-9-56": "arithmetic",
    "count-by-sevens": "arithmetic",
    "count-by-sevens-clt-clean": "arithmetic",
    "count-by-sevens-plt-clean": "arithmetic",
    "order-of-operations-paren": "arithmetic",
    "polymer-add-9": "arithmetic",
    # ── Creative ────────────────────────────────────────────────────
    "rabbit-poem": "creative",
    "rabbit-poem-like": "creative",
    "mo-poem": "creative",
    # ── Multilingual ────────────────────────────────────────────────
    "opposite_of_petit": "multilingual",
    "opposite_of_small_zh": "multilingual",
    "michael-fr-clt-clean": "multilingual",
    "michael-fr-plt-clean": "multilingual",
    "season-after-spring-fr-clt-clean": "multilingual",
    "season-after-spring-fr-plt-clean": "multilingual",
    # ── Safety ──────────────────────────────────────────────────────
    "bomb-baseline": "safety",
    "bomb-however": "safety",
    "bomb-bomb": "safety",
    "bomb-to": "safety",
    "bomb-make": "safety",
    "bomb-a": "safety",
    "bomb-bomb2": "safety",
    "bomb-comma": "safety",
    "bomb-mix": "safety",
    "bleach-ad": "safety",
    "bon-errors": "safety",
    # ── Reasoning ───────────────────────────────────────────────────
    "medical-diagnosis": "reasoning",
    "medical-diagnosis-meningitis": "reasoning",
    "medical-diagnosis-heart": "reasoning",
    "medical-diagnosis-sah": "reasoning",
    "cot-unfaithful-math-4": "reasoning",
    "cot-unfaithful-math-base": "reasoning",
    "cot-faithful-math-sqrt": "reasoning",
    "sally-induction-qk": "reasoning",
    "sally-school-clt-clean": "reasoning",
    "sally-school-plt-clean": "reasoning",
    "mo-911": "reasoning",
    "mo-chocolate": "reasoning",
    "batson-hallucination": "reasoning",
    "karpathy-hallucination": "reasoning",
    # ── Code ────────────────────────────────────────────────────────
    "str-indexing-pos-0-clt-clean": "code",
    "str-indexing-pos-0-plt-clean": "code",
    "pandas-group-clt-clean": "code",
    "pandas-group-plt-clean": "code",
}


class NeuronpediaClient:
    """Client for the Neuronpedia REST API and Anthropic's public graph bucket.

    Args:
        api_key: Optional API key for authenticated Neuronpedia access.
        rate_limit_delay: Seconds to wait between requests.
    """

    def __init__(
        self,
        api_key: str | None = None,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
    ) -> None:
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)
        if api_key:
            self.session.headers["X-Api-Key"] = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        """Make a rate-limited GET request.

        Args:
            url: Full URL to request.
            params: Optional query parameters.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        self._rate_limit()
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, url: str, data: dict[str, Any]) -> Any:
        """Make a rate-limited POST request.

        Args:
            url: Full URL to request.
            data: JSON body.

        Returns:
            Parsed JSON response.

        Raises:
            requests.HTTPError: On non-2xx response.
        """
        self._rate_limit()
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()

    # ─── Anthropic Bucket (pre-published graphs) ───────────────────────

    def list_anthropic_graphs(self) -> list[dict[str, Any]]:
        """List all 99 graphs from the Anthropic circuit-tracing paper.

        Returns:
            List of graph metadata dicts with keys: slug, scan, prompt_tokens, prompt,
            title_prefix. Each also gets a 'source' key set to 'anthropic_bucket'.
        """
        data = self._get(ANTHROPIC_METADATA_URL)
        graphs = data.get("graphs", [])
        for g in graphs:
            g["source"] = "anthropic_bucket"
            g["category"] = ANTHROPIC_GRAPH_CATEGORIES.get(g["slug"], "uncategorized")
        return graphs

    def download_anthropic_graph_json(self, slug: str) -> dict[str, Any]:
        """Download a graph JSON from the Anthropic public bucket.

        Args:
            slug: Graph slug (e.g., "capital-state-dallas").

        Returns:
            Parsed graph JSON dict.
        """
        url = ANTHROPIC_GRAPH_URL_TEMPLATE.format(slug=slug)
        return self._get(url)

    def fetch_anthropic_graph(
        self,
        slug: str,
        weight_threshold: float = 0.0,
    ) -> ig.Graph:
        """Download and parse an Anthropic bucket graph into igraph.

        Args:
            slug: Graph slug.
            weight_threshold: Minimum absolute edge weight to include.

        Returns:
            Directed igraph.Graph.
        """
        data = self.download_anthropic_graph_json(slug)
        return parse_attribution_graph(
            data,
            weight_threshold=weight_threshold,
            source_path=f"anthropic:{slug}",
        )

    def save_anthropic_graph(
        self,
        slug: str,
        output_dir: str | Path,
    ) -> Path:
        """Download an Anthropic bucket graph and save to disk.

        Args:
            slug: Graph slug.
            output_dir: Directory to save the JSON file.

        Returns:
            Path to the saved JSON file.
        """
        data = self.download_anthropic_graph_json(slug)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slug}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return output_path

    def download_all_anthropic_graphs(
        self,
        output_base_dir: str | Path,
        categorize: bool = True,
        show_progress: bool = True,
    ) -> list[Path]:
        """Download all Anthropic bucket graphs, optionally organized by category.

        Args:
            output_base_dir: Base directory for saving. If categorize=True, graphs
                are saved into subdirectories by category.
            categorize: If True, save into category subdirectories.
            show_progress: Whether to show a progress bar.

        Returns:
            List of paths to saved JSON files.
        """
        graphs = self.list_anthropic_graphs()
        output_base_dir = Path(output_base_dir)
        saved_paths: list[Path] = []

        iterator = graphs
        if show_progress:
            iterator = tqdm(graphs, desc="Downloading Anthropic graphs", unit="graph")

        for meta in iterator:
            slug = meta["slug"]
            if categorize:
                category = meta.get("category", "uncategorized")
                out_dir = output_base_dir / category
            else:
                out_dir = output_base_dir

            try:
                path = self.save_anthropic_graph(slug, out_dir)
                saved_paths.append(path)
            except requests.HTTPError as e:
                print(f"Warning: Failed to download {slug}: {e}")

        return saved_paths

    # ─── Neuronpedia API (community graphs + generation) ──────────────

    def get_graph_metadata(self, model_id: str, slug: str) -> dict[str, Any]:
        """Fetch metadata for a specific Neuronpedia graph.

        Args:
            model_id: Model identifier (e.g., "gemma-2-2b").
            slug: Graph slug identifier.

        Returns:
            Dict with graph metadata including 'url' (S3 link to JSON).
        """
        url = f"{NEURONPEDIA_API_URL}/graph/{model_id}/{slug}"
        return self._get(url)

    def download_graph_json(self, json_url: str) -> dict[str, Any]:
        """Download graph JSON from an S3 or other URL.

        Args:
            json_url: URL to the graph JSON file.

        Returns:
            Parsed graph JSON dict.
        """
        self._rate_limit()
        response = self.session.get(json_url)
        response.raise_for_status()
        return response.json()

    def fetch_graph(
        self,
        model_id: str,
        slug: str,
        weight_threshold: float = 0.0,
    ) -> ig.Graph:
        """Fetch a Neuronpedia graph and parse it into an igraph DiGraph.

        Args:
            model_id: Model identifier.
            slug: Graph slug.
            weight_threshold: Minimum absolute edge weight to include.

        Returns:
            Directed igraph.Graph.
        """
        metadata = self.get_graph_metadata(model_id, slug)
        json_url = metadata.get("url")
        if not json_url:
            raise ValueError(f"No JSON URL found for graph {model_id}/{slug}")

        graph_data = self.download_graph_json(json_url)
        return parse_attribution_graph(
            graph_data,
            weight_threshold=weight_threshold,
            source_path=f"neuronpedia:{model_id}/{slug}",
        )

    def save_graph_json(
        self,
        model_id: str,
        slug: str,
        output_dir: str | Path,
    ) -> Path:
        """Download and save a Neuronpedia graph JSON file to disk.

        Args:
            model_id: Model identifier.
            slug: Graph slug.
            output_dir: Directory to save the JSON file.

        Returns:
            Path to the saved JSON file.
        """
        metadata = self.get_graph_metadata(model_id, slug)
        json_url = metadata.get("url")
        if not json_url:
            raise ValueError(f"No JSON URL found for graph {model_id}/{slug}")

        graph_data = self.download_graph_json(json_url)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{slug}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        return output_path

    def generate_graph(
        self,
        prompt: str,
        model_id: str = "gemma-2-2b",
        slug: str | None = None,
        node_threshold: float = 0.8,
        edge_threshold: float = 0.85,
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        max_feature_nodes: int = 5000,
    ) -> dict[str, Any]:
        """Generate a new attribution graph via Neuronpedia's API.

        This runs circuit-tracing on Neuronpedia's GPU servers. No local GPU needed.
        Supported models: gemma-2-2b, qwen3-4b, gemma-3-4b-it.

        Args:
            prompt: The prompt to trace (max 64 tokens).
            model_id: Model to use.
            slug: Optional slug for the graph. Auto-generated if not provided.
            node_threshold: Node pruning threshold (0.5-1.0).
            edge_threshold: Edge pruning threshold (0.8-1.0).
            max_n_logits: Max number of output logits to include (5-15).
            desired_logit_prob: Cumulative probability coverage for logits (0.6-0.99).
            max_feature_nodes: Max feature nodes in graph (3000-10000).

        Returns:
            Dict with keys: message, s3url, url, numNodes, numLinks.

        Raises:
            requests.HTTPError: On failure (e.g., 503 if GPUs are busy).
        """
        payload: dict[str, Any] = {
            "prompt": prompt,
            "modelId": model_id,
            "slug": slug or "",
            "nodeThreshold": node_threshold,
            "edgeThreshold": edge_threshold,
            "maxNLogits": max_n_logits,
            "desiredLogitProb": desired_logit_prob,
            "maxFeatureNodes": max_feature_nodes,
        }
        url = f"{NEURONPEDIA_API_URL}/graph/generate"
        return self._post(url, payload)

    def generate_and_save(
        self,
        prompt: str,
        output_dir: str | Path,
        model_id: str = "gemma-2-2b",
        slug: str | None = None,
        **generate_kwargs: Any,
    ) -> Path:
        """Generate a graph and save it to disk.

        Args:
            prompt: The prompt to trace.
            output_dir: Directory to save the JSON file.
            model_id: Model to use.
            slug: Optional slug.
            **generate_kwargs: Additional arguments for generate_graph().

        Returns:
            Path to the saved JSON file.
        """
        result = self.generate_graph(
            prompt=prompt,
            model_id=model_id,
            slug=slug,
            **generate_kwargs,
        )

        s3_url = result.get("s3url") or result.get("url")
        if not s3_url:
            raise ValueError(f"No URL returned from generation: {result}")

        graph_data = self.download_graph_json(s3_url)

        # Determine filename from slug or response
        file_slug = slug or graph_data.get("metadata", {}).get("slug", "generated")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{file_slug}.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)

        return output_path

    def tokenize(
        self,
        prompt: str,
        model_id: str = "gemma-2-2b",
    ) -> dict[str, Any]:
        """Tokenize a prompt to check token count before graph generation.

        Args:
            prompt: The prompt to tokenize.
            model_id: Model to tokenize for.

        Returns:
            Dict with prompt, input_tokens, salient_logits, etc.
        """
        url = f"{NEURONPEDIA_API_URL}/graph/tokenize"
        return self._post(url, {"prompt": prompt, "modelId": model_id})

    def get_feature_details(
        self,
        model_id: str,
        source: str,
        feature_index: int,
    ) -> dict[str, Any]:
        """Fetch details for a specific feature.

        Args:
            model_id: Model identifier.
            source: Source set (e.g., "gemmascope-transcoder-16k").
            feature_index: Feature index.

        Returns:
            Feature details dict.
        """
        url = f"{NEURONPEDIA_API_URL}/feature/{model_id}/{source}/{feature_index}"
        return self._get(url)
