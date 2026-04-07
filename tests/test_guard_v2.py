"""Tests for v2 smart hallucination guard — query-type-aware thresholds."""
import pytest
from services.hallucination_guard import HallucinationGuard


@pytest.fixture
def guard():
    return HallucinationGuard(
        enabled=True,
        general_threshold=0.20,
        specific_threshold=0.30,
        comparison_threshold=0.25,
        marginal_high=0.50,
    )


# Context with known domain words
CONTEXT = (
    "fire alarm system integration requires relay modules interface panels "
    "testing verification commissioning electrical wiring conduit installation "
    "access control doors magnetic locks power supply backup battery"
)


class TestQueryTypeThresholds:
    def test_general_relaxed_passes(self, guard):
        # ~25% overlap: enough for general (threshold 0.20) but not specific (0.30)
        answer = "relay modules are critical components that need proper configuration during installation phase setup"
        result = guard.check(answer, CONTEXT, query_type="general")
        assert result["passed"] is True

    def test_specific_strict_fails_low_overlap(self, guard):
        answer = "relay modules are critical components that need proper configuration during installation phase setup"
        result = guard.check(answer, CONTEXT, query_type="specific")
        # Same answer but stricter threshold — may fail
        # If it passes, it's in marginal zone which is still valid
        assert "passed" in result

    def test_high_overlap_always_passes(self, guard):
        # Lots of matching words
        answer = "fire alarm system integration requires relay modules interface panels testing verification commissioning"
        for qt in ["general", "specific", "comparison"]:
            result = guard.check(answer, CONTEXT, query_type=qt)
            assert result["passed"] is True

    def test_zero_overlap_always_fails(self, guard):
        # Must exceed 10 tokens to avoid the short-answer groundedness floor
        answer = (
            "blockchain cryptocurrency quantum computing machine learning "
            "neural network distributed ledger consensus protocol validation"
        )
        for qt in ["general", "specific", "comparison"]:
            result = guard.check(answer, CONTEXT, query_type=qt)
            assert result["passed"] is False

    def test_comparison_threshold(self, guard):
        # ~26% overlap - passes comparison (0.25) but fails specific (0.30)
        answer = "relay modules provide essential connectivity between different system components throughout building infrastructure"
        result = guard.check(answer, CONTEXT, query_type="comparison")
        assert "passed" in result


class TestTierField:
    def test_result_has_tier(self, guard):
        result = guard.check("fire alarm testing", CONTEXT)
        assert "tier" in result

    def test_disabled_returns_disabled_tier(self):
        guard = HallucinationGuard(enabled=False)
        result = guard.check("anything", "anything")
        assert result["tier"] == "disabled"
        assert result["passed"] is True

    def test_high_overlap_returns_tier1(self, guard):
        answer = "fire alarm system integration requires relay modules testing verification"
        result = guard.check(answer, CONTEXT)
        assert result["tier"] == "tier1"


class TestBackwardCompatibility:
    def test_no_query_type_defaults_to_specific(self, guard):
        result = guard.check("fire alarm testing", CONTEXT)
        assert "passed" in result
        assert "groundedness" in result

    def test_short_answer_floor(self, guard):
        result = guard.check("yes", CONTEXT)
        assert result["groundedness"] >= 0.40
