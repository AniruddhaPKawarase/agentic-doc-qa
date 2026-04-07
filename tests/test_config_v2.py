"""
Tests for v2 configuration fields and feature flags.
──────────────────────────────────────────────────────────────────────────────
Validates that all v2-specific settings have correct defaults and can be
overridden via environment variables.
"""

import pytest

from config import Settings
from services.token_tracker import PRICING


def _settings(**overrides) -> Settings:
    """Create a Settings instance without reading .env files."""
    defaults = {"openai_api_key": "test-key", "_env_file": None}
    defaults.update(overrides)
    return Settings(**defaults)


class TestV2FeatureFlags:
    """Feature flags should default to enabled for the v2 pipeline."""

    def test_v2_feature_flags_have_defaults(self) -> None:
        s = _settings()
        assert s.enable_full_context is True
        assert s.enable_bm25 is True
        assert s.enable_summary_generation is True
        assert s.enable_llm_guard is True

    def test_v2_context_thresholds(self) -> None:
        s = _settings()
        assert s.full_context_threshold == 80_000
        assert s.summary_threshold == 200_000
        assert s.bm25_weight == pytest.approx(0.30)

    def test_v2_retrieval_defaults(self) -> None:
        s = _settings()
        assert s.primary_max_output_tokens == 4096

    def test_v2_guard_thresholds(self) -> None:
        s = _settings()
        assert s.guard_general_threshold == pytest.approx(0.20)
        assert s.guard_specific_threshold == pytest.approx(0.30)
        assert s.guard_marginal_low == pytest.approx(0.25)
        assert s.guard_marginal_high == pytest.approx(0.50)

    def test_v2_session_extended_ttl(self) -> None:
        s = _settings()
        assert s.session_ttl_hours == 168

    def test_v1_pipeline_flag(self) -> None:
        s = _settings(pipeline_version="v1")
        assert s.pipeline_version == "v1"

    def test_v2_monitoring_defaults(self) -> None:
        s = _settings()
        assert s.enable_metrics is True
        assert s.alert_latency_warning_ms == 15_000
        assert s.alert_latency_critical_ms == 30_000
        assert s.alert_cost_warning_usd == pytest.approx(0.50)
        assert s.alert_cost_critical_usd == pytest.approx(1.00)
        assert s.alert_groundedness_warning == pytest.approx(0.40)
        assert s.alert_groundedness_critical == pytest.approx(0.20)

    def test_token_tracker_gpt4o_pricing(self) -> None:
        assert "gpt-4o" in PRICING
        assert PRICING["gpt-4o"]["input"] == pytest.approx(2.50)
        assert PRICING["gpt-4o"]["output"] == pytest.approx(10.00)


class TestV2PipelineDefaults:
    """Pipeline model configuration defaults."""

    def test_default_pipeline_version(self) -> None:
        s = _settings()
        assert s.pipeline_version == "v2"

    def test_default_models(self) -> None:
        s = _settings()
        assert s.primary_model == "gpt-4o"
        assert s.secondary_model == "gpt-4o-mini"

    def test_max_history_extended(self) -> None:
        s = _settings()
        assert s.max_history_messages == 50
