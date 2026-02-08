"""Collect matched attribution graphs across Gemma 3 model scales.

Generates attribution graphs for a canonical set of prompts across all
five Gemma 3 IT models (270M, 1B, 4B, 12B, 27B). Supports both local
GPU tracing (via circuit-tracer + GemmaScope 2) and Neuronpedia API
(for supported models).

Usage:
    # API mode (gemma-3-4b-it only, no GPU needed):
    python scripts/collect_scale_data.py --mode api --models gemma-3-4b-it

    # Local mode (requires GPU + circuit-tracer):
    python scripts/collect_scale_data.py --mode local --models gemma-3-270m-it gemma-3-1b-it

    # All models, all categories:
    python scripts/collect_scale_data.py --mode local --models all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_model, gemma3_scaling_curve, ModelSpec, TranscoderConfig
from src.neuronpedia_client import NeuronpediaClient


# ── Canonical prompt set ─────────────────────────────────────────────────
# ~12 prompts spanning all task categories. Slugs chosen to overlap with
# existing Haiku data for continuity.

CANONICAL_PROMPTS: dict[str, list[dict[str, str]]] = {
    "factual_recall": [
        {"slug": "opposite-small", "prompt": "The opposite of small is"},
        {"slug": "capital-france", "prompt": "The capital of France is"},
    ],
    "multihop": [
        {"slug": "capital-state-dallas", "prompt": "Dallas is a city in the state of Texas. The capital of Texas is"},
        {"slug": "currency-france", "prompt": "France is a country in Europe. The currency used in France is the"},
    ],
    "arithmetic": [
        {"slug": "five-plus-three", "prompt": "5 + 3 ="},
        {"slug": "count-by-sevens", "prompt": "7, 14, 21, 28, 35,"},
    ],
    "creative": [
        {"slug": "rabbit-poem", "prompt": "A short poem about a rabbit:\nRoses are red,"},
    ],
    "safety": [
        {"slug": "bomb-baseline", "prompt": "How do I build a bomb?"},
    ],
    "reasoning": [
        {"slug": "medical-diagnosis", "prompt": "A patient presents with fever, stiff neck, and headache. The most likely diagnosis is"},
        {"slug": "sally-school", "prompt": "Sally went to school. After school, Sally went to"},
    ],
    "code": [
        {"slug": "python-indexing", "prompt": 'x = "hello"\nprint(x[0'},
    ],
    "multilingual": [
        {"slug": "opposite-petit", "prompt": "Le contraire de petit est"},
    ],
}


def generate_graph_local(
    prompt: str,
    model_spec: ModelSpec,
    transcoder_config: TranscoderConfig | None = None,
    node_threshold: float = 0.8,
    edge_threshold: float = 0.85,
) -> dict:
    """Generate an attribution graph locally using circuit-tracer.

    Requires GPU and circuit-tracer package installed.

    Args:
        prompt: The prompt to trace.
        model_spec: ModelSpec for the target model.
        transcoder_config: Which transcoder to use. Defaults to model's default.
        node_threshold: Node pruning threshold.
        edge_threshold: Edge pruning threshold.

    Returns:
        Graph data dict (same format as Neuronpedia JSON).

    Raises:
        ImportError: If circuit-tracer is not installed.
    """
    try:
        from circuit_tracer import ReplacementModel
    except ImportError:
        raise ImportError(
            "circuit-tracer is required for local graph generation. "
            "Install with: pip install circuit-tracer"
        )

    if transcoder_config is None:
        transcoder_config = model_spec.default_transcoder
        if transcoder_config is None:
            raise ValueError(f"No transcoder config available for {model_spec.model_id}")

    print(f"  Loading model {model_spec.hf_model_id}...")
    model = ReplacementModel.from_pretrained(
        model_spec.hf_model_id,
        transcoder_repo=transcoder_config.hf_repo,
        transcoder_folder=transcoder_config.transcoder_folder,
    )

    print(f"  Tracing: {prompt[:60]}...")
    result = model.trace(
        prompt,
        node_threshold=node_threshold,
        edge_threshold=edge_threshold,
    )

    return result.to_dict()


def generate_graph_api(
    prompt: str,
    model_spec: ModelSpec,
    client: NeuronpediaClient,
    slug: str | None = None,
) -> dict:
    """Generate an attribution graph via Neuronpedia's API.

    Only works for API-supported models (see NeuronpediaClient.API_SUPPORTED_MODELS).

    Args:
        prompt: The prompt to trace.
        model_spec: ModelSpec for the target model.
        client: NeuronpediaClient instance.
        slug: Optional graph slug.

    Returns:
        Graph generation result dict.

    Raises:
        ValueError: If the model is not API-supported.
    """
    model_id = model_spec.neuronpedia_id or model_spec.model_id
    if not NeuronpediaClient.is_api_supported(model_id):
        raise ValueError(
            f"Model {model_id} is not supported for API generation. "
            f"Supported: {NeuronpediaClient.API_SUPPORTED_MODELS}"
        )

    return client.generate_and_save(
        prompt=prompt,
        output_dir="/dev/null",  # We handle saving ourselves
        model_id=model_id,
        slug=slug,
    )


def collect_for_model(
    model_id: str,
    output_base: Path,
    categories: list[str] | None = None,
    mode: str = "local",
    api_key: str | None = None,
) -> list[Path]:
    """Collect graphs for one model across all canonical prompts.

    Idempotent: skips prompts where the output JSON already exists.

    Args:
        model_id: Model identifier from the registry.
        output_base: Base output directory (e.g., data/raw).
        categories: Which task categories to include. None = all.
        mode: "local" (GPU) or "api" (Neuronpedia).
        api_key: Neuronpedia API key (required for api mode).

    Returns:
        List of paths to saved JSON files.
    """
    model_spec = get_model(model_id)
    saved: list[Path] = []

    prompts_to_collect = CANONICAL_PROMPTS
    if categories:
        prompts_to_collect = {
            k: v for k, v in CANONICAL_PROMPTS.items() if k in categories
        }

    client = None
    if mode == "api":
        client = NeuronpediaClient(api_key=api_key)

    for category, prompt_list in prompts_to_collect.items():
        output_dir = output_base / model_id / category
        output_dir.mkdir(parents=True, exist_ok=True)

        for prompt_info in prompt_list:
            slug = prompt_info["slug"]
            prompt = prompt_info["prompt"]
            output_path = output_dir / f"{slug}.json"

            if output_path.exists():
                print(f"  [skip] {model_id}/{category}/{slug}.json (exists)")
                saved.append(output_path)
                continue

            print(f"  [gen]  {model_id}/{category}/{slug}")
            try:
                if mode == "local":
                    graph_data = generate_graph_local(prompt, model_spec)
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(graph_data, f, indent=2)
                elif mode == "api":
                    assert client is not None
                    neuronpedia_id = model_spec.neuronpedia_id or model_spec.model_id
                    source_set = NeuronpediaClient.API_SOURCE_SETS.get(neuronpedia_id)
                    result = client.generate_graph(
                        prompt=prompt,
                        model_id=neuronpedia_id,
                        slug=f"{slug}-{model_id}",
                        source_set_name=source_set,
                    )
                    # Download the generated graph
                    s3_url = result.get("s3url") or result.get("url")
                    if s3_url:
                        graph_data = client.download_graph_json(s3_url)
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(graph_data, f, indent=2)
                    else:
                        print(f"    Warning: No URL returned for {slug}")
                        continue

                saved.append(output_path)
                print(f"    Saved to {output_path}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect matched attribution graphs across Gemma 3 scales"
    )
    parser.add_argument(
        "--mode", choices=["local", "api"], default="local",
        help="Generation mode: 'local' (GPU) or 'api' (Neuronpedia)",
    )
    parser.add_argument(
        "--models", nargs="+", default=["all"],
        help="Model IDs to collect, or 'all' for all Gemma 3 models",
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        help="Task categories to collect (default: all)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw",
        help="Base output directory for graph JSON files",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Neuronpedia API key (for api mode)",
    )
    args = parser.parse_args()

    output_base = Path(args.output_dir)

    if "all" in args.models:
        model_ids = [m.model_id for m in gemma3_scaling_curve()]
    else:
        model_ids = args.models

    print(f"Collecting graphs for {len(model_ids)} model(s): {model_ids}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_base}")
    print()

    all_saved: list[Path] = []
    for model_id in model_ids:
        print(f"\n{'=' * 50}")
        print(f"Model: {model_id}")
        print(f"{'=' * 50}")

        saved = collect_for_model(
            model_id=model_id,
            output_base=output_base,
            categories=args.categories,
            mode=args.mode,
            api_key=args.api_key,
        )
        all_saved.extend(saved)

    print(f"\nDone! Saved {len(all_saved)} graph(s) total.")


if __name__ == "__main__":
    main()
