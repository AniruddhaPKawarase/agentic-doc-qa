# DocQA v2 Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the foundation layer for v2 — config/feature flags, full text store, query classifier, context manager, and updated token tracker. After this phase, the new services exist and are unit-tested, but are NOT yet wired into the routers.

**Architecture:** New services (`context_manager`, `fulltext_store`) sit between routers and existing retrieval/generation services. A feature flag (`PIPELINE_VERSION`) controls v1 vs v2 routing. The context manager classifies queries and selects the optimal context strategy (full doc, summary+retrieval, or retrieval-only).

**Tech Stack:** Python 3.11+, FastAPI, pydantic-settings, tiktoken, pytest

**Working Directory:** `C:\Users\ANIRUDDHA ASUS\Downloads\projects\VCS\VCS\PROD_SETUP\document-qa-agent`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `config.py` | MODIFY | Add v2 settings: feature flags, model routing, thresholds |
| `services/fulltext_store.py` | CREATE | Store and retrieve full document text per session |
| `services/context_manager.py` | CREATE | Query classification + context strategy selection |
| `services/token_tracker.py` | MODIFY | Add gpt-4o pricing |
| `tests/test_fulltext_store.py` | CREATE | Unit tests for full text store |
| `tests/test_context_manager.py` | CREATE | Unit tests for context manager |
| `tests/test_config_v2.py` | CREATE | Verify v2 config loads correctly |
| `requirements.txt` | MODIFY | Add `rank-bm25>=0.2.2` (needed in Phase 3 but add now) |

---

## Task 1: Add v2 Configuration & Feature Flags

**Files:**
- Modify: `config.py:13-81` (add new settings to `Settings` class)
- Create: `tests/test_config_v2.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_v2.py`:

```python
"""Tests for v2 configuration settings."""

import os
import pytest


def test_v2_feature_flags_have_defaults():
    """v2 settings should have sensible defaults without any env vars."""
    # Clear any cached settings
    from config import Settings
    s = Settings(
        openai_api_key="test-key",
        _env_file=None,
    )

    assert s.pipeline_version == "v2"
    assert s.primary_model == "gpt-4o"
    assert s.secondary_model == "gpt-4o-mini"
    assert s.enable_full_context is True
    assert s.enable_bm25 is True
    assert s.enable_summary_generation is True
    assert s.enable_llm_guard is True


def test_v2_context_thresholds():
    """Context strategy thresholds should be set."""
    from config import Settings
    s = Settings(openai_api_key="test-key", _env_file=None)

    assert s.full_context_threshold == 80000
    assert s.summary_threshold == 200000


def test_v2_retrieval_defaults():
    """v2 retrieval settings should override v1 defaults."""
    from config import Settings
    s = Settings(
        openai_api_key="test-key",
        pipeline_version="v2",
        _env_file=None,
    )

    assert s.primary_max_output_tokens == 4096


def test_v2_guard_thresholds():
    """v2 guard should have query-type-aware thresholds."""
    from config import Settings
    s = Settings(openai_api_key="test-key", _env_file=None)

    assert s.guard_general_threshold == 0.20
    assert s.guard_specific_threshold == 0.30
    assert s.guard_marginal_low == 0.25
    assert s.guard_marginal_high == 0.50


def test_v2_session_extended_ttl():
    """v2 sessions should default to 7 days."""
    from config import Settings
    s = Settings(openai_api_key="test-key", _env_file=None)

    assert s.session_ttl_hours == 168


def test_v1_pipeline_flag():
    """Setting pipeline_version=v1 should be allowed."""
    from config import Settings
    s = Settings(
        openai_api_key="test-key",
        pipeline_version="v1",
        _env_file=None,
    )

    assert s.pipeline_version == "v1"


def test_v2_monitoring_defaults():
    """Monitoring alert thresholds should have defaults."""
    from config import Settings
    s = Settings(openai_api_key="test-key", _env_file=None)

    assert s.enable_metrics is True
    assert s.alert_latency_warning_ms == 15000
    assert s.alert_cost_critical_usd == 1.00
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd "C:\Users\ANIRUDDHA ASUS\Downloads\projects\VCS\VCS\PROD_SETUP\document-qa-agent"
python -m pytest tests/test_config_v2.py -v
```

Expected: FAIL — `Settings` class doesn't have `pipeline_version`, `primary_model`, etc.

- [ ] **Step 3: Implement v2 config settings**

Modify `config.py` — add these fields to the `Settings` class after the S3 section (after line 67):

```python
    # ── v2 Pipeline ──────────────────────────────────────
    pipeline_version: str = "v2"
    primary_model: str = "gpt-4o"
    secondary_model: str = "gpt-4o-mini"
    primary_max_output_tokens: int = 4096

    # ── v2 Feature Flags ─────────────────────────────────
    enable_full_context: bool = True
    enable_bm25: bool = True
    enable_summary_generation: bool = True
    enable_llm_guard: bool = True

    # ── v2 Context Strategy ──────────────────────────────
    full_context_threshold: int = 80000
    summary_threshold: int = 200000
    bm25_weight: float = 0.30

    # ── v2 Hallucination Guard ───────────────────────────
    guard_general_threshold: float = 0.20
    guard_specific_threshold: float = 0.30
    guard_marginal_low: float = 0.25
    guard_marginal_high: float = 0.50

    # ── v2 Monitoring ────────────────────────────────────
    enable_metrics: bool = True
    alert_latency_warning_ms: int = 15000
    alert_latency_critical_ms: int = 30000
    alert_cost_warning_usd: float = 0.50
    alert_cost_critical_usd: float = 1.00
    alert_groundedness_warning: float = 0.40
    alert_groundedness_critical: float = 0.20
```

Also update these existing defaults to v2 values:
- `session_ttl_hours: int = 24` → `session_ttl_hours: int = 168`
- `max_history_messages: int = 20` → `max_history_messages: int = 50`

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config_v2.py -v
```

Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config_v2.py
git commit -m "feat(config): add v2 pipeline settings, feature flags, and monitoring thresholds"
```

---

## Task 2: Update Token Tracker with gpt-4o Pricing

**Files:**
- Modify: `services/token_tracker.py:15-18` (update PRICING dict)
- Test: inline verification (existing tests still pass)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_v2.py`:

```python
def test_token_tracker_gpt4o_pricing():
    """Token tracker should know gpt-4o pricing."""
    from services.token_tracker import PRICING

    assert "gpt-4o" in PRICING
    assert PRICING["gpt-4o"]["input"] == 2.50
    assert PRICING["gpt-4o"]["output"] == 10.00
    # Existing models still present
    assert "gpt-4o-mini" in PRICING
    assert "text-embedding-3-small" in PRICING
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config_v2.py::test_token_tracker_gpt4o_pricing -v
```

Expected: FAIL — `"gpt-4o"` not in `PRICING`

- [ ] **Step 3: Add gpt-4o pricing**

In `services/token_tracker.py`, replace the `PRICING` dict (lines 15-18):

```python
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02},
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_config_v2.py::test_token_tracker_gpt4o_pricing -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add services/token_tracker.py tests/test_config_v2.py
git commit -m "feat(token-tracker): add gpt-4o pricing for v2 cost estimation"
```

---

## Task 3: Create Full Text Store

**Files:**
- Create: `services/fulltext_store.py`
- Create: `tests/test_fulltext_store.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fulltext_store.py`:

```python
"""Tests for FullTextStore — stores complete document text per session."""

import pytest
from services.fulltext_store import FullTextStore, StoredDocument


@pytest.fixture
def store():
    return FullTextStore()


SAMPLE_TEXT = "This is a sample document about fire alarm systems. " * 50
SAMPLE_PAGES = [
    "Page 1: Introduction to fire alarm systems.",
    "Page 2: Components required for integration.",
    "Page 3: Testing requirements after installation.",
]


class TestStore:
    def test_store_and_retrieve(self, store: FullTextStore):
        """Store a document and retrieve it by session + file name."""
        doc = store.store("sess1", "scope.docx", SAMPLE_TEXT, SAMPLE_PAGES)

        assert isinstance(doc, StoredDocument)
        assert doc.file_name == "scope.docx"
        assert doc.full_text == SAMPLE_TEXT
        assert doc.token_count > 0
        assert doc.page_texts == SAMPLE_PAGES

    def test_get_document(self, store: FullTextStore):
        """Retrieve a single stored document."""
        store.store("sess1", "scope.docx", SAMPLE_TEXT, SAMPLE_PAGES)

        doc = store.get_document("sess1", "scope.docx")
        assert doc is not None
        assert doc.file_name == "scope.docx"

    def test_get_document_missing(self, store: FullTextStore):
        """Missing document returns None."""
        assert store.get_document("sess1", "nope.docx") is None

    def test_get_session_text_all(self, store: FullTextStore):
        """Get combined text for all files in a session."""
        store.store("sess1", "a.docx", "Doc A content", [])
        store.store("sess1", "b.docx", "Doc B content", [])

        combined = store.get_session_text("sess1")
        assert "Doc A content" in combined
        assert "Doc B content" in combined

    def test_get_session_text_filtered(self, store: FullTextStore):
        """Get text for specific files only."""
        store.store("sess1", "a.docx", "Doc A content", [])
        store.store("sess1", "b.docx", "Doc B content", [])

        filtered = store.get_session_text("sess1", file_names=["a.docx"])
        assert "Doc A content" in filtered
        assert "Doc B content" not in filtered

    def test_get_session_token_count(self, store: FullTextStore):
        """Token count sums across all session files."""
        store.store("sess1", "a.docx", "Hello world", [])
        store.store("sess1", "b.docx", "Goodbye world", [])

        total = store.get_session_token_count("sess1")
        assert total > 0

    def test_get_session_token_count_filtered(self, store: FullTextStore):
        """Token count for specific files only."""
        store.store("sess1", "a.docx", "Hello world", [])
        store.store("sess1", "b.docx", "Goodbye world" * 100, [])

        total_a = store.get_session_token_count("sess1", file_names=["a.docx"])
        total_all = store.get_session_token_count("sess1")
        assert total_a < total_all

    def test_delete_session(self, store: FullTextStore):
        """Delete clears all documents for a session."""
        store.store("sess1", "a.docx", "content", [])
        store.delete_session("sess1")

        assert store.get_document("sess1", "a.docx") is None
        assert store.get_session_token_count("sess1") == 0

    def test_session_isolation(self, store: FullTextStore):
        """Different sessions don't see each other's documents."""
        store.store("sess1", "a.docx", "Session 1 content", [])
        store.store("sess2", "b.docx", "Session 2 content", [])

        assert store.get_document("sess1", "b.docx") is None
        assert store.get_document("sess2", "a.docx") is None

    def test_get_file_names(self, store: FullTextStore):
        """List all file names in a session."""
        store.store("sess1", "a.docx", "A", [])
        store.store("sess1", "b.pdf", "B", [])

        names = store.get_file_names("sess1")
        assert set(names) == {"a.docx", "b.pdf"}

    def test_empty_session(self, store: FullTextStore):
        """Empty session returns empty results, not errors."""
        assert store.get_session_text("empty") == ""
        assert store.get_session_token_count("empty") == 0
        assert store.get_file_names("empty") == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_fulltext_store.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'services.fulltext_store'`

- [ ] **Step 3: Implement FullTextStore**

Create `services/fulltext_store.py`:

```python
"""
Full Text Store — Stores complete document text per session.
──────────────────────────────────────────────────────────────────────────────
Used for FULL_CONTEXT mode where the entire document is loaded into the
LLM prompt. Stores text in-memory with token counts pre-computed.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger("docqa.fulltext_store")

_enc = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class StoredDocument:
    """Immutable container for a stored document's full text."""
    file_name: str
    full_text: str
    token_count: int
    page_texts: List[str]


class FullTextStore:
    """In-memory store for complete document texts, keyed by session + file."""

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, StoredDocument]] = {}
        self._lock = threading.Lock()

    def store(
        self,
        session_id: str,
        file_name: str,
        text: str,
        page_texts: List[str],
    ) -> StoredDocument:
        """Store full text for a document. Returns the StoredDocument."""
        token_count = len(_enc.encode(text))
        doc = StoredDocument(
            file_name=file_name,
            full_text=text,
            token_count=token_count,
            page_texts=list(page_texts),
        )
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][file_name] = doc

        logger.info(
            f"Stored full text: session={session_id}, "
            f"file={file_name}, tokens={token_count}"
        )
        return doc

    def get_document(
        self,
        session_id: str,
        file_name: str,
    ) -> Optional[StoredDocument]:
        """Retrieve a single document. Returns None if not found."""
        with self._lock:
            session_docs = self._store.get(session_id, {})
            return session_docs.get(file_name)

    def get_session_text(
        self,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> str:
        """
        Get combined text for a session.
        If file_names is provided, only include those files.
        Each file's text is separated with a header for citation clarity.
        """
        with self._lock:
            session_docs = self._store.get(session_id, {})

        if not session_docs:
            return ""

        parts = []
        for fname, doc in session_docs.items():
            if file_names and fname not in file_names:
                continue
            header = f"=== FILE: {fname} ==="
            parts.append(f"{header}\n{doc.full_text}")

        return "\n\n".join(parts)

    def get_session_token_count(
        self,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> int:
        """Get total token count for session (or filtered files)."""
        with self._lock:
            session_docs = self._store.get(session_id, {})

        total = 0
        for fname, doc in session_docs.items():
            if file_names and fname not in file_names:
                continue
            total += doc.token_count
        return total

    def get_file_names(self, session_id: str) -> List[str]:
        """List all file names stored for a session."""
        with self._lock:
            session_docs = self._store.get(session_id, {})
            return list(session_docs.keys())

    def delete_session(self, session_id: str) -> None:
        """Remove all stored documents for a session."""
        with self._lock:
            self._store.pop(session_id, None)
        logger.info(f"Deleted full text store for session {session_id}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_fulltext_store.py -v
```

Expected: All 12 tests PASS

- [ ] **Step 5: Commit**

```bash
git add services/fulltext_store.py tests/test_fulltext_store.py
git commit -m "feat(fulltext-store): add in-memory full document text storage for v2 FULL_CONTEXT mode"
```

---

## Task 4: Create Context Manager

**Files:**
- Create: `services/context_manager.py`
- Create: `tests/test_context_manager.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_manager.py`:

```python
"""Tests for ContextManager — query classification + context strategy selection."""

import pytest
from services.context_manager import (
    ContextManager,
    ContextPayload,
    QueryType,
    ContextStrategy,
    classify_query,
)
from services.fulltext_store import FullTextStore


# ── Query Classification Tests ────────────────────────────────────────────────

class TestClassifyQuery:
    def test_summary_query(self):
        assert classify_query("Summarize this document") == QueryType.GENERAL

    def test_overview_query(self):
        assert classify_query("What is this document about?") == QueryType.GENERAL

    def test_list_all_query(self):
        assert classify_query("List all the systems mentioned") == QueryType.GENERAL

    def test_business_objectives_query(self):
        assert classify_query("What are the business objectives?") == QueryType.GENERAL

    def test_scope_of_work_query(self):
        assert classify_query("What is the scope of work?") == QueryType.GENERAL

    def test_key_requirements_query(self):
        assert classify_query("What are the key requirements?") == QueryType.GENERAL

    def test_specific_query(self):
        assert classify_query("What voltage is required for the relay module?") == QueryType.SPECIFIC

    def test_fire_alarm_specific(self):
        assert classify_query("What happens to access-controlled doors during a fire alarm event?") == QueryType.SPECIFIC

    def test_comparison_query(self):
        assert classify_query("Compare the electrical requirements with safety requirements") == QueryType.COMPARISON

    def test_versus_query(self):
        assert classify_query("Phase 1 vs Phase 2 scope differences") == QueryType.COMPARISON

    def test_relate_query(self):
        assert classify_query("How does fire alarm relate to access control?") == QueryType.COMPARISON

    def test_empty_query_defaults_to_specific(self):
        assert classify_query("tell me about section 5.3") == QueryType.SPECIFIC


# ── Strategy Selection Tests ──────────────────────────────────────────────────

class TestStrategySelection:
    @pytest.fixture
    def store(self):
        s = FullTextStore()
        return s

    @pytest.fixture
    def manager(self, store):
        return ContextManager(
            fulltext_store=store,
            full_context_threshold=80000,
            summary_threshold=200000,
            primary_model="gpt-4o",
        )

    def test_small_doc_uses_full_context(self, manager: ContextManager, store: FullTextStore):
        """Documents under 80K tokens should use FULL_CONTEXT."""
        store.store("sess1", "small.docx", "Hello world " * 100, [])

        strategy = manager.select_strategy("sess1", QueryType.GENERAL)
        assert strategy == ContextStrategy.FULL_CONTEXT

    def test_medium_doc_general_uses_summary(self, manager: ContextManager, store: FullTextStore):
        """Medium docs with general queries use SUMMARY_PLUS_RETRIEVAL."""
        # ~100K tokens: "word " is ~1 token, so 100000 words ~ 100K tokens
        big_text = "construction specification requirements " * 30000
        store.store("sess1", "medium.docx", big_text, [])

        strategy = manager.select_strategy("sess1", QueryType.GENERAL)
        assert strategy == ContextStrategy.SUMMARY_PLUS_RETRIEVAL

    def test_medium_doc_specific_uses_retrieval(self, manager: ContextManager, store: FullTextStore):
        """Medium docs with specific queries use RETRIEVAL_ONLY."""
        big_text = "construction specification requirements " * 30000
        store.store("sess1", "medium.docx", big_text, [])

        strategy = manager.select_strategy("sess1", QueryType.SPECIFIC)
        assert strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_huge_doc_uses_retrieval(self, manager: ContextManager, store: FullTextStore):
        """Documents over 200K tokens always use RETRIEVAL_ONLY for specific."""
        huge_text = "construction specification requirements " * 80000
        store.store("sess1", "huge.docx", huge_text, [])

        strategy = manager.select_strategy("sess1", QueryType.SPECIFIC)
        assert strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_empty_session_uses_retrieval(self, manager: ContextManager):
        """Session with no stored text falls back to RETRIEVAL_ONLY."""
        strategy = manager.select_strategy("empty", QueryType.GENERAL)
        assert strategy == ContextStrategy.RETRIEVAL_ONLY

    def test_multiple_small_docs_sum_tokens(self, manager: ContextManager, store: FullTextStore):
        """Multiple small docs together may exceed threshold."""
        for i in range(5):
            store.store("sess1", f"doc{i}.docx", "word " * 20000, [])
        # 5 * 20000 = 100K tokens → over 80K threshold

        strategy = manager.select_strategy("sess1", QueryType.GENERAL)
        assert strategy == ContextStrategy.SUMMARY_PLUS_RETRIEVAL

    def test_model_is_always_primary(self, manager: ContextManager, store: FullTextStore):
        """All strategies use the primary model (gpt-4o)."""
        store.store("sess1", "small.docx", "Hello world", [])

        model = manager.get_model_for_strategy(ContextStrategy.FULL_CONTEXT)
        assert model == "gpt-4o"
        model = manager.get_model_for_strategy(ContextStrategy.RETRIEVAL_ONLY)
        assert model == "gpt-4o"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_context_manager.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'services.context_manager'`

- [ ] **Step 3: Implement Context Manager**

Create `services/context_manager.py`:

```python
"""
Context Manager — Query classification + context strategy selection.
──────────────────────────────────────────────────────────────────────────────
The brain of the v2 pipeline. Decides HOW to answer each query:
- Classify query type (general, specific, comparison)
- Select context strategy (full_context, summary+retrieval, retrieval_only)
- Route to appropriate model

Does NOT call the LLM or retrieval services — it decides the strategy,
and the router orchestrates the actual pipeline.
"""

import logging
import re
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

from services.fulltext_store import FullTextStore

logger = logging.getLogger("docqa.context_manager")


# ── Enums ─────────────────────────────────────────────────────────────────────

class QueryType(str, Enum):
    GENERAL = "general"
    SPECIFIC = "specific"
    COMPARISON = "comparison"


class ContextStrategy(str, Enum):
    FULL_CONTEXT = "full_context"
    SUMMARY_PLUS_RETRIEVAL = "summary_plus_retrieval"
    RETRIEVAL_ONLY = "retrieval_only"


# ── Query Classification ──────────────────────────────────────────────────────

GENERAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bsummariz\w*\b|\bsummary\b|\boverview\b",
        r"\bwhat is this (document|file|report|spec)\b",
        r"\bkey (points|takeaways|highlights|requirements|features)\b",
        r"\blist all\b|\blist the\b|\bwhat are the\b",
        r"\bmain (topics|themes|sections|objectives|goals)\b",
        r"\bscope of (work|project|this)\b",
        r"\bbusiness objectives\b",
        r"\bwhat does this (document|file|cover|contain)\b",
        r"\btell me about this (document|file)\b",
        r"\bwhat('s| is) (in|covered|included)\b",
        r"\bgive me (an? )?(overview|summary|rundown)\b",
    ]
]

COMPARISON_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bcompar\w*\b|\bcontrast\b|\bdifferen\w*\b",
        r"\bversus\b|\bvs\.?\b",
        r"\bhow does .+ relate to\b",
        r"\bbetween .+ and\b",
        r"\bsimilarit\w*\b",
    ]
]


def classify_query(query: str) -> QueryType:
    """
    Classify a query as GENERAL, SPECIFIC, or COMPARISON.

    Rule-based classification using regex patterns.
    General: summaries, overviews, broad topic extraction
    Comparison: comparing documents or sections
    Specific: everything else (factual questions, data lookups)
    """
    for pattern in COMPARISON_PATTERNS:
        if pattern.search(query):
            return QueryType.COMPARISON

    for pattern in GENERAL_PATTERNS:
        if pattern.search(query):
            return QueryType.GENERAL

    return QueryType.SPECIFIC


# ── Context Payload ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContextPayload:
    """Immutable result from context manager — tells the router what to do."""
    strategy: ContextStrategy
    query_type: QueryType
    model: str
    total_doc_tokens: int
    file_names: List[str]


# ── Context Manager ──────────────────────────────────────────────────────────

class ContextManager:
    """Selects the optimal context strategy for a given query + session."""

    def __init__(
        self,
        fulltext_store: FullTextStore,
        full_context_threshold: int = 80000,
        summary_threshold: int = 200000,
        primary_model: str = "gpt-4o",
    ) -> None:
        self._store = fulltext_store
        self._full_context_threshold = full_context_threshold
        self._summary_threshold = summary_threshold
        self._primary_model = primary_model

    def select_strategy(
        self,
        session_id: str,
        query_type: QueryType,
        file_names: Optional[List[str]] = None,
    ) -> ContextStrategy:
        """
        Select context strategy based on document size and query type.

        Args:
            session_id: The session to check.
            query_type: Classified query type.
            file_names: If provided, only count tokens for these files.
        """
        total_tokens = self._store.get_session_token_count(
            session_id, file_names=file_names
        )

        if total_tokens == 0:
            return ContextStrategy.RETRIEVAL_ONLY

        if total_tokens <= self._full_context_threshold:
            return ContextStrategy.FULL_CONTEXT

        if total_tokens <= self._summary_threshold:
            if query_type in (QueryType.GENERAL, QueryType.COMPARISON):
                return ContextStrategy.SUMMARY_PLUS_RETRIEVAL
            return ContextStrategy.RETRIEVAL_ONLY

        # > summary_threshold
        if query_type in (QueryType.GENERAL, QueryType.COMPARISON):
            return ContextStrategy.SUMMARY_PLUS_RETRIEVAL
        return ContextStrategy.RETRIEVAL_ONLY

    def get_model_for_strategy(self, strategy: ContextStrategy) -> str:
        """All v2 strategies use the primary model (gpt-4o)."""
        return self._primary_model

    def build_payload(
        self,
        session_id: str,
        query: str,
        target_file_names: Optional[List[str]] = None,
    ) -> ContextPayload:
        """
        Full decision pipeline: classify query → select strategy → build payload.

        Args:
            session_id: Active session.
            query: User's question.
            target_file_names: Files to scope to (from file scope resolver).
        """
        query_type = classify_query(query)
        strategy = self.select_strategy(session_id, query_type, target_file_names)
        model = self.get_model_for_strategy(strategy)

        file_names = target_file_names or self._store.get_file_names(session_id)
        total_tokens = self._store.get_session_token_count(
            session_id, file_names=target_file_names
        )

        logger.info(
            f"Context decision: session={session_id}, "
            f"query_type={query_type.value}, strategy={strategy.value}, "
            f"model={model}, doc_tokens={total_tokens}, files={file_names}"
        )

        return ContextPayload(
            strategy=strategy,
            query_type=query_type,
            model=model,
            total_doc_tokens=total_tokens,
            file_names=file_names,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_context_manager.py -v
```

Expected: All 19 tests PASS (12 classification + 7 strategy)

- [ ] **Step 5: Commit**

```bash
git add services/context_manager.py tests/test_context_manager.py
git commit -m "feat(context-manager): add query classifier and context strategy selector for v2 pipeline"
```

---

## Task 5: Add rank-bm25 Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Add dependency**

Append to `requirements.txt` after the `boto3` line:

```
# ── v2 Hybrid Search ─────────────────────────────────
rank-bm25>=0.2.2
```

- [ ] **Step 2: Install**

```bash
pip install rank-bm25>=0.2.2
```

- [ ] **Step 3: Verify import**

```bash
python -c "from rank_bm25 import BM25Okapi; print('rank-bm25 OK')"
```

Expected: `rank-bm25 OK`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt
git commit -m "chore(deps): add rank-bm25 for v2 hybrid search"
```

---

## Task 6: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: All tests PASS — both new v2 tests and existing tests. The existing tests should not break because:
- `config.py` only added new fields with defaults
- `token_tracker.py` only added a new key to the dict
- New files don't touch existing code

- [ ] **Step 2: Verify the agent still starts**

```bash
cd "C:\Users\ANIRUDDHA ASUS\Downloads\projects\VCS\VCS\PROD_SETUP\document-qa-agent"
timeout 5 python main.py || true
```

Expected: Agent starts and logs "Document Q&A Agent ready" (then times out — that's fine)

- [ ] **Step 3: Final commit if any fixes needed**

```bash
git status
# If clean: no action needed
# If fixes were made: commit them
```

---

## Phase 1 Complete Checklist

After all tasks, you should have:

| Artifact | Status |
|----------|--------|
| `config.py` — 25+ new v2 settings | Done |
| `services/fulltext_store.py` — full text store (130 lines) | Done |
| `services/context_manager.py` — query classifier + strategy selector (170 lines) | Done |
| `services/token_tracker.py` — gpt-4o pricing | Done |
| `requirements.txt` — rank-bm25 added | Done |
| `tests/test_config_v2.py` — 8 tests | Done |
| `tests/test_fulltext_store.py` — 12 tests | Done |
| `tests/test_context_manager.py` — 19 tests | Done |
| All existing tests still pass | Done |
| Agent starts without errors | Done |
| 5 commits total | Done |

**Phase 1 does NOT wire anything into the routers.** The new services exist, are tested, and are ready to be integrated in Phase 2-6.
