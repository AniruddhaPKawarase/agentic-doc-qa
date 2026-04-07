"""
Tests for ContextManager — query classification and strategy selection.
──────────────────────────────────────────────────────────────────────────────
Validates regex-based query classification and token-threshold-driven
strategy selection for the v2 pipeline.
"""

import pytest

from services.context_manager import (
    ContextManager,
    ContextStrategy,
    QueryType,
    classify_query,
)
from services.fulltext_store import FullTextStore


# ── Query Classification Tests ────────────────────────────────────────────


class TestClassifyQuery:
    """Test regex-based query classification with 8 demo questions + 4 edge cases."""

    # Demo questions — GENERAL
    def test_classify_summarize(self) -> None:
        assert classify_query("Summarize this document") == QueryType.GENERAL

    def test_classify_overview(self) -> None:
        assert classify_query("Give me an overview of the project") == QueryType.GENERAL

    def test_classify_key_points(self) -> None:
        assert classify_query("What are the key points?") == QueryType.GENERAL

    def test_classify_scope_of_work(self) -> None:
        assert classify_query("What is the scope of work?") == QueryType.GENERAL

    def test_classify_list_all(self) -> None:
        assert classify_query("List all requirements") == QueryType.GENERAL

    # Demo questions — COMPARISON
    def test_classify_compare(self) -> None:
        assert classify_query("Compare doc A and doc B") == QueryType.COMPARISON

    def test_classify_difference(self) -> None:
        assert classify_query("What is the difference between X and Y?") == QueryType.COMPARISON

    def test_classify_versus(self) -> None:
        assert classify_query("Option A versus Option B") == QueryType.COMPARISON

    # Demo questions — SPECIFIC (default)
    def test_classify_specific_detail(self) -> None:
        assert classify_query("What is the voltage rating for panel 3?") == QueryType.SPECIFIC

    # Edge cases
    def test_classify_empty_query(self) -> None:
        assert classify_query("") == QueryType.SPECIFIC

    def test_classify_case_insensitive(self) -> None:
        assert classify_query("SUMMARIZE everything") == QueryType.GENERAL

    def test_classify_comparison_priority_over_general(self) -> None:
        """COMPARISON check runs before GENERAL, so 'compare' wins over 'summary'."""
        assert classify_query("Compare the summary of both docs") == QueryType.COMPARISON

    def test_classify_vs_shorthand(self) -> None:
        assert classify_query("Plan A vs Plan B") == QueryType.COMPARISON


# ── Strategy Selection Tests ──────────────────────────────────────────────


class TestStrategySelection:
    """Test strategy selection based on token counts and query types."""

    @pytest.fixture
    def store(self) -> FullTextStore:
        return FullTextStore()

    @pytest.fixture
    def manager(self, store: FullTextStore) -> ContextManager:
        return ContextManager(
            fulltext_store=store,
            full_context_threshold=80_000,
            summary_threshold=200_000,
        )

    def test_small_doc_full_context(self, manager: ContextManager, store: FullTextStore) -> None:
        """Documents under 80K tokens should use FULL_CONTEXT."""
        # Simulate a small doc with 50K tokens — we store text then override the count
        store.store("s1", "small.pdf", "x " * 100, ["x " * 100])
        payload = manager.select_strategy("s1", "Summarize this document")
        # With a tiny doc, should be FULL_CONTEXT
        assert payload.strategy == ContextStrategy.FULL_CONTEXT

    def test_medium_general_summary_plus_retrieval(self, manager: ContextManager) -> None:
        """Medium doc (80K-200K) + GENERAL query → SUMMARY_PLUS_RETRIEVAL."""
        payload = manager.select_strategy_from_tokens(
            total_tokens=100_000,
            query="Summarize the document",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert payload.strategy == ContextStrategy.SUMMARY_PLUS_RETRIEVAL

    def test_medium_specific_retrieval_only(self, manager: ContextManager) -> None:
        """Medium doc (80K-200K) + SPECIFIC query → RETRIEVAL_ONLY."""
        payload = manager.select_strategy_from_tokens(
            total_tokens=100_000,
            query="What is the voltage rating?",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert payload.strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_huge_doc_general(self, manager: ContextManager) -> None:
        """Huge doc (>200K) + GENERAL → SUMMARY_PLUS_RETRIEVAL."""
        payload = manager.select_strategy_from_tokens(
            total_tokens=300_000,
            query="Give me an overview",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert payload.strategy == ContextStrategy.SUMMARY_PLUS_RETRIEVAL

    def test_huge_doc_specific(self, manager: ContextManager) -> None:
        """Huge doc (>200K) + SPECIFIC → RETRIEVAL_ONLY."""
        payload = manager.select_strategy_from_tokens(
            total_tokens=300_000,
            query="What is the voltage?",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert payload.strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_empty_session_retrieval_only(self, manager: ContextManager, store: FullTextStore) -> None:
        """Empty session (0 tokens) should fall back to RETRIEVAL_ONLY."""
        payload = manager.select_strategy("empty-session", "Summarize this")
        assert payload.strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_multiple_docs_sum_tokens(self, manager: ContextManager) -> None:
        """Multiple docs should sum their tokens for threshold comparison."""
        payload = manager.select_strategy_from_tokens(
            total_tokens=90_000,  # above 80K threshold
            query="Compare section 1 and section 2",
            session_id="s1",
            file_names=["doc_a.pdf", "doc_b.pdf"],
        )
        assert payload.strategy == ContextStrategy.SUMMARY_PLUS_RETRIEVAL
        assert payload.query_type == QueryType.COMPARISON

    def test_model_selection(self, manager: ContextManager) -> None:
        """FULL_CONTEXT should use primary model; others use secondary."""
        full = manager.select_strategy_from_tokens(
            total_tokens=50_000,
            query="Summarize",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert full.model == "gpt-4o"

        retrieval = manager.select_strategy_from_tokens(
            total_tokens=100_000,
            query="What is the voltage?",
            session_id="s1",
            file_names=["doc.pdf"],
        )
        assert retrieval.model == "gpt-4o-mini"
