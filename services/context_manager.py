"""
Context Manager — Query classification and context strategy selection.
──────────────────────────────────────────────────────────────────────────────
Classifies incoming queries by type (GENERAL / SPECIFIC / COMPARISON) and
selects the optimal context strategy (FULL_CONTEXT / SUMMARY_PLUS_RETRIEVAL /
RETRIEVAL_ONLY) based on document token counts and thresholds.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from services.fulltext_store import FullTextStore

logger = logging.getLogger("docqa.context_manager")


# ── Enums ─────────────────────────────────────────────────────────────────


class QueryType(Enum):
    GENERAL = "general"
    SPECIFIC = "specific"
    COMPARISON = "comparison"


class ContextStrategy(Enum):
    FULL_CONTEXT = "full_context"
    SUMMARY_PLUS_RETRIEVAL = "summary_plus_retrieval"
    RETRIEVAL_ONLY = "retrieval_only"


# ── Query Classification ──────────────────────────────────────────────────

_COMPARISON_PATTERN = re.compile(
    r"\b(compare|contrast|difference|versus|vs\b|relate\s+to|between\s+\w+\s+and\s+\w+)",
    re.IGNORECASE,
)

_GENERAL_PATTERN = re.compile(
    r"\b(summarize|summary|overview|what\s+is\s+this\s+document|key\s+points|"
    r"key\s+requirements|list\s+all|what\s+are\s+the|scope\s+of\s+work|"
    r"business\s+objectives)\b",
    re.IGNORECASE,
)


def classify_query(query: str) -> QueryType:
    """Classify a query into GENERAL, SPECIFIC, or COMPARISON.

    Check order: COMPARISON first (more specific), then GENERAL, default SPECIFIC.
    """
    if _COMPARISON_PATTERN.search(query):
        return QueryType.COMPARISON
    if _GENERAL_PATTERN.search(query):
        return QueryType.GENERAL
    return QueryType.SPECIFIC


# ── Context Payload ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextPayload:
    """Immutable result of strategy selection."""

    strategy: ContextStrategy
    query_type: QueryType
    model: str
    total_doc_tokens: int
    file_names: tuple  # tuple for frozen dataclass


# ── Context Manager ──────────────────────────────────────────────────────


class ContextManager:
    """Selects context strategy based on document size and query type."""

    def __init__(
        self,
        fulltext_store: FullTextStore,
        full_context_threshold: int = 80_000,
        summary_threshold: int = 200_000,
        primary_model: str = "gpt-4o",
        secondary_model: str = "gpt-4o-mini",
    ) -> None:
        self._store = fulltext_store
        self._full_context_threshold = full_context_threshold
        self._summary_threshold = summary_threshold
        self._primary_model = primary_model
        self._secondary_model = secondary_model

    def select_strategy(
        self,
        session_id: str,
        query: str,
        file_names: Optional[List[str]] = None,
    ) -> ContextPayload:
        """Select strategy using the fulltext store to get token counts."""
        total_tokens = self._store.get_session_token_count(
            session_id, file_names=file_names
        )
        resolved_files = (
            file_names
            if file_names is not None
            else self._store.get_file_names(session_id)
        )
        return self._build_payload(total_tokens, query, session_id, resolved_files)

    def select_strategy_from_tokens(
        self,
        total_tokens: int,
        query: str,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> ContextPayload:
        """Select strategy from a pre-computed token count (for testing)."""
        return self._build_payload(
            total_tokens, query, session_id, file_names or []
        )

    def _build_payload(
        self,
        total_tokens: int,
        query: str,
        session_id: str,
        file_names: Optional[List[str]],
    ) -> ContextPayload:
        query_type = classify_query(query)
        strategy = self._select(total_tokens, query_type)
        model = (
            self._primary_model
            if strategy == ContextStrategy.FULL_CONTEXT
            else self._secondary_model
        )
        logger.info(
            "Session %s | tokens=%d | query_type=%s | strategy=%s | model=%s",
            session_id,
            total_tokens,
            query_type.value,
            strategy.value,
            model,
        )
        return ContextPayload(
            strategy=strategy,
            query_type=query_type,
            model=model,
            total_doc_tokens=total_tokens,
            file_names=tuple(file_names or []),
        )

    def _select(self, total_tokens: int, query_type: QueryType) -> ContextStrategy:
        """Core strategy decision tree."""
        if total_tokens == 0:
            return ContextStrategy.RETRIEVAL_ONLY

        if total_tokens <= self._full_context_threshold:
            return ContextStrategy.FULL_CONTEXT

        is_broad = query_type in (QueryType.GENERAL, QueryType.COMPARISON)

        if total_tokens <= self._summary_threshold:
            if is_broad:
                return ContextStrategy.SUMMARY_PLUS_RETRIEVAL
            return ContextStrategy.RETRIEVAL_ONLY

        # total_tokens > summary_threshold
        if is_broad:
            return ContextStrategy.SUMMARY_PLUS_RETRIEVAL
        return ContextStrategy.RETRIEVAL_ONLY
