"""Model registry and transcoder configurations for cross-scale analysis.

Frozen dataclasses for model metadata (layer counts, hidden dims, transcoder
HuggingFace repos) for the Gemma 3 family (270M-27B) and legacy models
(Gemma-2-2B, Qwen3-4B).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TranscoderConfig:
    """Configuration for a model's transcoder set.

    Attributes:
        hf_repo: HuggingFace repository ID (e.g., "google/gemma-scope-2-2b-it").
        transcoder_folder: Subfolder within the repo (e.g., "transcoder_all").
        width: Transcoder dictionary width (e.g., 16384).
        is_clt: Whether this is a CLT (cross-layer) transcoder vs PLT.
    """

    hf_repo: str
    transcoder_folder: str = "transcoder_all"
    width: int = 16384
    is_clt: bool = False


@dataclass(frozen=True)
class ModelSpec:
    """Specification for a model available for motif analysis.

    Attributes:
        model_id: Canonical identifier (e.g., "gemma-3-1b-it").
        family: Model family name (e.g., "gemma-3").
        variant: Variant tag (e.g., "it" for instruction-tuned).
        n_params: Parameter count in millions (e.g., 270 for 270M).
        n_layers: Number of transformer layers.
        hidden_dim: Hidden dimension size.
        transcoders: Available transcoder configurations.
        neuronpedia_id: Model ID on Neuronpedia, if different from model_id.
    """

    model_id: str
    family: str
    variant: str
    n_params: int  # millions
    n_layers: int
    hidden_dim: int
    transcoders: tuple[TranscoderConfig, ...] = ()
    neuronpedia_id: str | None = None

    @property
    def log_params(self) -> float:
        """Log10 of total parameter count (for scaling plots)."""
        return math.log10(self.n_params * 1e6)

    @property
    def default_transcoder(self) -> TranscoderConfig | None:
        """First available transcoder config, or None."""
        return self.transcoders[0] if self.transcoders else None

    @property
    def hf_model_id(self) -> str:
        """HuggingFace model ID for loading."""
        return f"google/{self.model_id}"


# ── Gemma 3 IT family (GemmaScope 2 transcoders) ────────────────────────

GEMMA_3_MODELS: dict[str, ModelSpec] = {
    "gemma-3-270m-it": ModelSpec(
        model_id="gemma-3-270m-it",
        family="gemma-3",
        variant="it",
        n_params=270,
        n_layers=18,
        hidden_dim=1536,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-270m-it",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-270m-it",
                transcoder_folder="transcoder_all_clt",
                width=16384,
                is_clt=True,
            ),
        ),
    ),
    "gemma-3-1b-it": ModelSpec(
        model_id="gemma-3-1b-it",
        family="gemma-3",
        variant="it",
        n_params=1000,
        n_layers=26,
        hidden_dim=2048,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-1b-it",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-1b-it",
                transcoder_folder="transcoder_all_clt",
                width=16384,
                is_clt=True,
            ),
        ),
    ),
    "gemma-3-4b-it": ModelSpec(
        model_id="gemma-3-4b-it",
        family="gemma-3",
        variant="it",
        n_params=4000,
        n_layers=34,
        hidden_dim=2560,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-4b-it",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
        ),
        neuronpedia_id="gemma-3-4b-it",
    ),
    "gemma-3-12b-it": ModelSpec(
        model_id="gemma-3-12b-it",
        family="gemma-3",
        variant="it",
        n_params=12000,
        n_layers=42,
        hidden_dim=3840,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-12b-it",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
        ),
    ),
    "gemma-3-27b-it": ModelSpec(
        model_id="gemma-3-27b-it",
        family="gemma-3",
        variant="it",
        n_params=27000,
        n_layers=46,
        hidden_dim=4608,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2-27b-it",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
        ),
    ),
}

# ── Legacy models (existing cross-model comparison) ──────────────────────

LEGACY_MODELS: dict[str, ModelSpec] = {
    "gemma-2-2b": ModelSpec(
        model_id="gemma-2-2b",
        family="gemma-2",
        variant="base",
        n_params=2000,
        n_layers=26,
        hidden_dim=2304,
        transcoders=(
            TranscoderConfig(
                hf_repo="google/gemma-scope-2b-pt-transcoders",
                transcoder_folder="transcoder_all",
                width=16384,
                is_clt=False,
            ),
        ),
        neuronpedia_id="gemma-2-2b",
    ),
    "qwen3-4b": ModelSpec(
        model_id="qwen3-4b",
        family="qwen-3",
        variant="base",
        n_params=4000,
        n_layers=36,
        hidden_dim=2560,
        transcoders=(),
        neuronpedia_id="qwen3-4b",
    ),
}

# ── Combined registry ────────────────────────────────────────────────────

ALL_MODELS: dict[str, ModelSpec] = {**GEMMA_3_MODELS, **LEGACY_MODELS}


def get_model(model_id: str) -> ModelSpec:
    """Look up a model by its canonical ID.

    Args:
        model_id: Canonical model identifier.

    Returns:
        ModelSpec for the requested model.

    Raises:
        KeyError: If model_id is not in the registry.
    """
    if model_id not in ALL_MODELS:
        available = ", ".join(sorted(ALL_MODELS.keys()))
        raise KeyError(
            f"Unknown model '{model_id}'. Available models: {available}"
        )
    return ALL_MODELS[model_id]


def gemma3_scaling_curve() -> list[ModelSpec]:
    """Return Gemma 3 IT models sorted by parameter count (ascending).

    Returns:
        List of ModelSpec in order: 270M, 1B, 4B, 12B, 27B.
    """
    return sorted(GEMMA_3_MODELS.values(), key=lambda m: m.n_params)
