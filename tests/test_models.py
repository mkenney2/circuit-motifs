"""Tests for src/models.py — model registry and transcoder configs."""

import math

import pytest

from src.models import (
    ModelSpec,
    TranscoderConfig,
    GEMMA_3_MODELS,
    LEGACY_MODELS,
    ALL_MODELS,
    get_model,
    gemma3_scaling_curve,
)


class TestModelSpec:
    def test_log_params(self):
        spec = get_model("gemma-3-270m-it")
        expected = math.log10(270 * 1e6)
        assert abs(spec.log_params - expected) < 1e-10

    def test_log_params_large(self):
        spec = get_model("gemma-3-27b-it")
        expected = math.log10(27000 * 1e6)
        assert abs(spec.log_params - expected) < 1e-10

    def test_default_transcoder_returns_first(self):
        spec = get_model("gemma-3-270m-it")
        assert spec.default_transcoder is not None
        assert spec.default_transcoder.is_clt is False

    def test_default_transcoder_none_when_empty(self):
        spec = ModelSpec(
            model_id="test",
            family="test",
            variant="base",
            n_params=100,
            n_layers=12,
            hidden_dim=768,
            transcoders=(),
        )
        assert spec.default_transcoder is None

    def test_hf_model_id(self):
        spec = get_model("gemma-3-4b-it")
        assert spec.hf_model_id == "google/gemma-3-4b-it"

    def test_frozen(self):
        spec = get_model("gemma-3-1b-it")
        with pytest.raises(AttributeError):
            spec.n_layers = 99


class TestGemma3Registry:
    def test_all_five_sizes_registered(self):
        expected_ids = {
            "gemma-3-270m-it",
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        }
        assert set(GEMMA_3_MODELS.keys()) == expected_ids

    def test_correct_layer_counts(self):
        expected_layers = {
            "gemma-3-270m-it": 18,
            "gemma-3-1b-it": 26,
            "gemma-3-4b-it": 34,
            "gemma-3-12b-it": 42,
            "gemma-3-27b-it": 46,
        }
        for model_id, expected in expected_layers.items():
            assert GEMMA_3_MODELS[model_id].n_layers == expected

    def test_all_are_gemma3_family(self):
        for spec in GEMMA_3_MODELS.values():
            assert spec.family == "gemma-3"
            assert spec.variant == "it"

    def test_clt_only_for_270m_and_1b(self):
        for model_id, spec in GEMMA_3_MODELS.items():
            clt_transcoders = [t for t in spec.transcoders if t.is_clt]
            if model_id in ("gemma-3-270m-it", "gemma-3-1b-it"):
                assert len(clt_transcoders) == 1, f"{model_id} should have CLT"
            else:
                assert len(clt_transcoders) == 0, f"{model_id} should not have CLT"

    def test_all_have_plt_transcoder(self):
        for model_id, spec in GEMMA_3_MODELS.items():
            plt_transcoders = [t for t in spec.transcoders if not t.is_clt]
            assert len(plt_transcoders) >= 1, f"{model_id} missing PLT transcoder"


class TestLegacyModels:
    def test_gemma2_registered(self):
        assert "gemma-2-2b" in LEGACY_MODELS

    def test_qwen_registered(self):
        assert "qwen3-4b" in LEGACY_MODELS

    def test_in_all_models(self):
        for model_id in LEGACY_MODELS:
            assert model_id in ALL_MODELS


class TestGetModel:
    def test_lookup_valid(self):
        spec = get_model("gemma-3-4b-it")
        assert spec.model_id == "gemma-3-4b-it"
        assert spec.n_params == 4000

    def test_lookup_legacy(self):
        spec = get_model("gemma-2-2b")
        assert spec.family == "gemma-2"

    def test_unknown_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("nonexistent-model-9000")

    def test_error_message_lists_available(self):
        with pytest.raises(KeyError, match="gemma-3-4b-it"):
            get_model("bad-id")


class TestGemma3ScalingCurve:
    def test_returns_five_models(self):
        curve = gemma3_scaling_curve()
        assert len(curve) == 5

    def test_sorted_by_param_count(self):
        curve = gemma3_scaling_curve()
        params = [m.n_params for m in curve]
        assert params == sorted(params)

    def test_ordering(self):
        curve = gemma3_scaling_curve()
        ids = [m.model_id for m in curve]
        assert ids == [
            "gemma-3-270m-it",
            "gemma-3-1b-it",
            "gemma-3-4b-it",
            "gemma-3-12b-it",
            "gemma-3-27b-it",
        ]

    def test_all_are_model_spec(self):
        for spec in gemma3_scaling_curve():
            assert isinstance(spec, ModelSpec)
