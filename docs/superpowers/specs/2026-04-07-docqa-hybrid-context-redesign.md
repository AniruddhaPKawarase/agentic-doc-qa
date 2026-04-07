# Document Q&A Agent v2 — Hybrid Context Redesign

**Date:** 2026-04-07
**Status:** PENDING APPROVAL
**Approach:** Hybrid Context (Approach A)
**Author:** Claude Opus 4.6

---

## Table of Contents

1. [User Story & Problem Statement](#1-user-story--problem-statement)
2. [Requirements Summary](#2-requirements-summary)
3. [Architecture Overview](#3-architecture-overview)
4. [Component Design](#4-component-design)
5. [Data Flow](#5-data-flow)
6. [API Changes](#6-api-changes)
7. [Configuration & Feature Flags](#7-configuration--feature-flags)
8. [Evaluation: 12-Parameter Review](#8-evaluation-12-parameter-review)
9. [Testing Strategy](#9-testing-strategy)
10. [Phased Implementation Plan](#10-phased-implementation-plan)
11. [Deployment Plan](#11-deployment-plan)
12. [Risk Register](#12-risk-register)

---

## 1. User Story & Problem Statement

### Problem
The document-qa-agent fails on general questions. When a user asks "What are the Business Objectives?" the agent responds "The provided document context does not explicitly list 'Business Objectives.' Therefore, I cannot provide that information." But when rephrased as "What are the business objectives of this project?" it works.

**Root causes identified:**
1. **FAISS retrieval too narrow** — `top_k=8`, `score_threshold=0.30` means broad queries fail to retrieve relevant chunks
2. **No full-document fallback** — agent only sees 8 best-matching chunks, cannot synthesize across the whole document
3. **System prompt overly restrictive** — rejects outright when FAISS returns nothing above threshold
4. **Hallucination guard false positives** — token-overlap scoring triggers on valid general answers because summaries naturally use different vocabulary than source text
5. **Model limitation** — `gpt-4o-mini` has weaker reasoning for complex synthesis tasks

### User Story
> As a construction project manager, I want to upload any document (scope, specs, product data, drawings) and ask ANY type of question — general summaries, specific data points, cross-section analysis — and get accurate, cited answers, just like I would with Claude or ChatGPT's web interface.

### Demo Questions That Must All Work
1. "What happens to access-controlled doors during a fire alarm event?"
2. "What components are required for fire alarm integration (relays, interface modules)?"
3. "What testing is required after integration with fire alarm systems?"
4. "Summarize this document"
5. "What is the scope of work?"
6. "List all the systems mentioned in this document"
7. "What are the key requirements?"
8. "Compare the electrical requirements with safety requirements"

---

## 2. Requirements Summary

| Dimension | Requirement | Source |
|-----------|-------------|--------|
| Document sizes | 1-500+ pages, scalable | Q1.1 |
| Document volume | Varies widely per session | Q1.2 |
| All formats | PDF, DOCX, XLSX, images, text | Q1.3 |
| Drawing priority | Secondary — text Q&A first | Q1.4 |
| Query types | ANY question about the document | Q2.1 |
| Multi-document | Yes, with per-document citations | Q2.2 |
| Conversation depth | All depths (1-30+ questions) | Q2.3 |
| Response style | Adaptive — match user's style | Q2.4 |
| LLM provider | OpenAI only | Q3.1 |
| Cost tolerance | Quality is priority, no strict limit | Q3.2 |
| Latency | Streaming, start quickly | Q3.3 |
| Concurrency | 100+ customer-facing users | Q3.4 |
| Frontend | Separate React/Vue (IFS platform) | Q4.1 |
| API contract | Can change freely | Q4.2 |
| Session persistence | Days/weeks | Q4.3 |
| Hallucination guard | LLM-based + relaxed thresholds | Q5.1 |
| Out-of-context | Strict refusal | Q5.2 |
| Citations | Critical — every claim cites page/section | Q5.3 |
| Deployment | Sandbox first, then production | Q6.1 |
| Rollback | Feature flag | Q6.2 |
| Monitoring | Full (logs + metrics + alerting) | Q6.3 |

---

## 3. Architecture Overview

### Current vs New Pipeline

```
CURRENT (v1):
  Upload → Chunk (512 tok) → Embed → FAISS
  Query → Embed → FAISS top-8 → gpt-4o-mini → token-overlap guard → response

NEW (v2 — Hybrid Context):
  Upload → Chunk → Embed → FAISS
       └→ Store full text → Compute token count → Generate doc summary

  Query → Context Manager decides strategy:
    ├─ FULL_CONTEXT (<80K tokens): Full doc text → gpt-4o → LLM-based guard → response
    ├─ SUMMARY_PLUS_RETRIEVAL (80K-200K): Summary + enhanced FAISS top-20 → gpt-4o → guard
    └─ RETRIEVAL_ONLY (>200K): Enhanced FAISS top-30 + BM25 hybrid → gpt-4o → guard
```

### New Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Nginx (port 8000)                                   │
│                       prefix: /docqa/ → :8006                                │
└───────────────────────────────┬─────────────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────────────┐
│                      FastAPI App (port 8006)                                 │
│                                                                              │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────────┐ │
│  │ Converse Router │  │ Chat Router    │  │ Session Router                 │ │
│  │ POST /converse  │  │ POST /chat     │  │ CRUD + files + metrics         │ │
│  │ POST /stream    │  │ POST /stream   │  │                                │ │
│  └───────┬────────┘  └───────┬────────┘  └────────────────────────────────┘ │
│          │                   │                                               │
│          └─────────┬─────────┘                                               │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    Context Manager (NEW)                                 │ │
│  │                                                                         │ │
│  │  1. Classify query (general / specific / comparison)                    │ │
│  │  2. Resolve file scope (existing logic)                                 │ │
│  │  3. Select context strategy:                                            │ │
│  │     ├─ FULL_CONTEXT: load entire doc text into prompt                   │ │
│  │     ├─ SUMMARY_PLUS_RETRIEVAL: doc summary + FAISS chunks              │ │
│  │     └─ RETRIEVAL_ONLY: enhanced FAISS + BM25 hybrid search             │ │
│  │  4. Build context payload                                               │ │
│  │  5. Route to appropriate model                                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                    │                                                         │
│  ┌─────────┬──────┴────────┬──────────────────┐                             │
│  │         │               │                  │                             │
│  ▼         ▼               ▼                  ▼                             │
│ ┌────────┐ ┌────────────┐ ┌───────────────┐ ┌──────────────────┐           │
│ │ Full   │ │ Document   │ │ Enhanced      │ │ BM25 Keyword     │           │
│ │ Text   │ │ Summary    │ │ FAISS         │ │ Search (NEW)     │           │
│ │ Store  │ │ Service    │ │ Retrieval     │ │                  │           │
│ │ (NEW)  │ │ (NEW)      │ │ (MODIFIED)    │ │ rank_bm25        │           │
│ └────────┘ └────────────┘ └───────────────┘ └──────────────────┘           │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  Generation Service (MODIFIED)                          │ │
│  │                                                                         │ │
│  │  Model Router: gpt-4o (primary) | gpt-4o-mini (follow-ups)             │ │
│  │  Citation-enforced system prompt                                        │ │
│  │  Adaptive response style                                                │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                    │                                                         │
│                    ▼                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  Smart Hallucination Guard (MODIFIED)                    │ │
│  │                                                                         │ │
│  │  Query-type aware: relaxed for general, strict for factual              │ │
│  │  LLM-based faithfulness check (gpt-4o-mini judge)                       │ │
│  │  Citation validation                                                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                    │                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                  Monitoring & Metrics (NEW)                              │ │
│  │                                                                         │ │
│  │  Structured JSON logs | Per-query metrics | /metrics endpoint           │ │
│  │  Alert thresholds: latency >30s, cost >$1, groundedness <0.3           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Component Design

### 4.1 Context Manager (NEW FILE: `services/context_manager.py`)

The brain of v2. Decides HOW to answer each query.

**Responsibilities:**
1. Classify query type (general, specific, comparison, summary)
2. Compute document token count from stored full text
3. Select context strategy based on document size + query type
4. Build the context payload for the generation service

**Context Strategy Decision Matrix:**

| Total Doc Tokens | Query Type | Strategy | Model |
|-----------------|------------|----------|-------|
| < 80,000 | Any | FULL_CONTEXT | gpt-4o |
| 80,000 - 200,000 | General/Summary | SUMMARY + top-20 FAISS | gpt-4o |
| 80,000 - 200,000 | Specific | Top-20 FAISS + BM25 | gpt-4o |
| > 200,000 | General/Summary | Summary + top-30 FAISS | gpt-4o |
| > 200,000 | Specific | Top-30 FAISS + BM25 | gpt-4o |

**Query Classification (rule-based + LLM fallback):**

```python
GENERAL_PATTERNS = [
    r"summarize|summary|overview|about",
    r"what is this (document|file|report)",
    r"key (points|takeaways|highlights|requirements)",
    r"list all|list the|what are the",
    r"main (topics|themes|sections|objectives)",
    r"scope of (work|project)",
    r"business objectives",
]

COMPARISON_PATTERNS = [
    r"compare|contrast|difference|versus|vs\b",
    r"how does .+ relate to",
    r"between .+ and",
]

# If no pattern matches → classify as SPECIFIC
# Fallback: use gpt-4o-mini one-shot classification for ambiguous queries
```

**Interface:**

```python
@dataclass(frozen=True)
class ContextPayload:
    context_text: str          # The actual context to send to LLM
    strategy: str              # "full_context" | "summary_plus_retrieval" | "retrieval_only"
    query_type: str            # "general" | "specific" | "comparison" | "summary"
    model: str                 # "gpt-4o" or "gpt-4o-mini"
    sources: List[SourceInfo]  # Source chunks for citation
    total_doc_tokens: int      # For cost tracking
    file_names: List[str]      # Files included in context

class ContextManager:
    async def build_context(
        self,
        session_id: str,
        query: str,
        uploaded_file_names: List[str],
    ) -> ContextPayload: ...
```

### 4.2 Full Text Store (NEW FILE: `services/fulltext_store.py`)

Stores the complete extracted text of each document per session. Used for FULL_CONTEXT mode.

**Storage:**
- In-memory: `{session_id: {file_name: full_text}}` — fast access
- S3 backup: `{prefix}/session_data/{session_id}/fulltext/{file_name}.txt` — persistence

**Token counting:**
- On upload, count tokens via tiktoken and store alongside text
- Cached per file — never recount

```python
@dataclass(frozen=True)
class StoredDocument:
    file_name: str
    full_text: str
    token_count: int
    page_texts: List[str]  # Per-page text for page-level citations

class FullTextStore:
    def store(self, session_id: str, file_name: str, text: str, page_texts: List[str]) -> StoredDocument: ...
    def get_session_text(self, session_id: str, file_names: List[str] = None) -> str: ...
    def get_session_token_count(self, session_id: str, file_names: List[str] = None) -> int: ...
    def get_document(self, session_id: str, file_name: str) -> Optional[StoredDocument]: ...
```

### 4.3 Document Summary Service (NEW FILE: `services/summary_service.py`)

Generates and caches comprehensive document summaries on upload.

**When used:**
- Always generated on upload (async, non-blocking)
- Included in context for SUMMARY_PLUS_RETRIEVAL strategy
- Included as a "summary chunk" in FAISS index for better general query retrieval

**Summary prompt:**

```
Analyze this document thoroughly and produce a structured summary:

1. DOCUMENT TYPE: What kind of document is this?
2. MAIN SUBJECT: What is this document about?
3. KEY SECTIONS: List every major section/topic with a brief description
4. KEY DATA POINTS: Important numbers, specs, dates, names, requirements
5. SCOPE: What does this document cover and NOT cover?
6. RELATIONSHIPS: How do different sections relate to each other?

Be exhaustive. Include every topic, requirement, and data point.
```

**Storage:** Alongside full text in memory + S3.

```python
class SummaryService:
    async def generate_summary(self, file_name: str, full_text: str) -> str: ...
    async def generate_summary_batch(self, documents: List[Tuple[str, str]]) -> Dict[str, str]: ...
```

### 4.4 BM25 Keyword Search (NEW FILE: `services/bm25_service.py`)

Adds keyword-based retrieval alongside FAISS semantic search.

**Why needed:** FAISS misses queries where the exact keywords matter (e.g., "fire alarm" should strongly match chunks containing "fire alarm" even if semantic similarity is moderate).

**Implementation:**
- Uses `rank_bm25` library (lightweight, no external service needed)
- Tokenizes chunks on upload, builds BM25 index per session
- Hybrid scoring: `final_score = alpha * semantic_score + (1 - alpha) * bm25_score`
- Default alpha = 0.7 (favor semantic, but keyword search helps)

```python
class BM25Service:
    def index_session(self, session_id: str, chunks: List[Chunk]) -> None: ...
    def search(self, session_id: str, query: str, top_k: int = 30) -> List[Tuple[int, float]]: ...
    def delete_session(self, session_id: str) -> None: ...
```

### 4.5 Enhanced Retrieval Service (MODIFIED: `services/retrieval_service.py`)

**Changes:**
- Default `top_k` increased: 8 → 20
- Default `score_threshold` lowered: 0.30 → 0.15
- Add hybrid search combining FAISS + BM25 scores
- Add "always include summary chunk" option
- Over-fetch factor increased for file-scoped queries

```python
class RetrievalService:
    async def retrieve(
        self,
        session_id: str,
        query: str,
        target_files: Optional[List[str]] = None,
        scope_mode: str = "global",
        use_bm25: bool = True,          # NEW
        include_summary: bool = False,    # NEW
    ) -> RetrievalResult: ...
```

### 4.6 Generation Service (MODIFIED: `services/generation_service.py`)

**Changes:**
- Model routing: `gpt-4o` for primary generation, `gpt-4o-mini` for follow-ups
- Citation-enforced system prompt
- Adaptive response style based on query type
- Accepts full-context mode (no chunk labels, uses page-level citations)
- Max output tokens increased: 2048 → 4096 (for detailed responses)

**New citation-enforced system prompt:**

```
You are a Document Q&A Assistant. You answer questions based on the provided document context.

CITATION RULES (MANDATORY):
1. Every factual claim MUST include a citation: [Source: {filename}, page {N}] or [Source: {filename}, section "{heading}"]
2. If you cannot cite a claim from the provided context, do not include it.
3. For summaries, cite the overall document and specific pages for key facts.
4. When information comes from multiple documents, cite each source separately.

RESPONSE RULES:
1. Answer questions thoroughly using ONLY the provided document context.
2. If the context does not contain enough information, say so clearly with what IS available.
3. Match the user's question style — concise for simple questions, detailed for complex ones.
4. Use bullet points, tables, or paragraphs as appropriate for the content.
5. For summaries and overviews, cover ALL major topics in the document.
6. For specific questions, quote relevant text when helpful.

OUT-OF-CONTEXT:
If the question is completely unrelated to the provided documents, respond:
"This question doesn't appear to be covered in the uploaded documents. The documents contain information about: [list key topics]. Could you ask about one of these topics?"
```

### 4.7 Smart Hallucination Guard (MODIFIED: `services/hallucination_guard.py`)

**Changes:**
- Add query-type-aware thresholds
- Add LLM-based faithfulness scoring (for high-stakes factual queries)
- Keep token-overlap as fast first-pass, escalate to LLM check when marginal
- Citation validation: check that cited pages/sections exist in source

**Two-tier guard:**

| Tier | Method | When Used | Threshold |
|------|--------|-----------|-----------|
| Tier 1 (fast) | Token-overlap | Always runs first | 0.25 (lowered from 0.35) |
| Tier 2 (LLM) | gpt-4o-mini judge | When Tier 1 score is 0.25-0.50 (marginal zone) | Binary pass/fail |

**Query-type adjustments:**

| Query Type | Tier 1 Threshold | Tier 2 Required? |
|------------|-----------------|------------------|
| General/Summary | 0.20 | Only if < 0.20 |
| Specific/Factual | 0.30 | If 0.25-0.50 |
| Comparison | 0.25 | If 0.25-0.50 |

**LLM faithfulness prompt (Tier 2):**
```
Given the SOURCE CONTEXT and the ANSWER, determine if the answer is faithful to the source.
Score: "FAITHFUL" or "UNFAITHFUL"
Reason: One sentence explaining why.
```

### 4.8 Token Tracker (MODIFIED: `services/token_tracker.py`)

**Changes:**
- Add gpt-4o pricing
- Add per-strategy cost tracking
- Add model field to pipeline log

```python
PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},       # per 1M tokens
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02},
}
```

### 4.9 Session Service (MODIFIED: `services/session_service.py`)

**Changes:**
- Extend default TTL: 24h → 7 days (configurable to 30 days)
- Store document summaries in session
- Store full text references (file names + token counts)
- Increase history window: 20 → 50 messages

### 4.10 Monitoring Service (NEW FILE: `services/monitoring_service.py`)

**Responsibilities:**
- Structured JSON logging with per-query fields
- In-memory metrics aggregation (recent 1000 queries)
- `/metrics` endpoint for Prometheus scraping
- Alert threshold checking

**Metrics tracked per query:**

```python
@dataclass(frozen=True)
class QueryMetrics:
    session_id: str
    query_type: str           # general / specific / comparison
    context_strategy: str     # full_context / summary_plus_retrieval / retrieval_only
    model_used: str           # gpt-4o / gpt-4o-mini
    total_tokens: int
    estimated_cost_usd: float
    latency_ms: float
    groundedness_score: float
    guard_passed: bool
    cached: bool
    file_count: int
    timestamp: str
```

**Alert thresholds (configurable):**

| Metric | Warning | Critical |
|--------|---------|----------|
| Latency (total_ms) | > 15,000 | > 30,000 |
| Cost (per query) | > $0.50 | > $1.00 |
| Groundedness | < 0.40 | < 0.20 |
| Error rate (5min window) | > 5% | > 15% |

**Endpoints:**
- `GET /metrics` — Prometheus-format metrics
- `GET /api/metrics/summary` — JSON dashboard summary (last 1h/24h/7d)

### 4.11 Feature Flag System (NEW: in `config.py`)

```python
# Feature flags
pipeline_version: str = "v2"      # "v1" (legacy) or "v2" (hybrid context)
enable_full_context: bool = True   # Allow full-document context mode
enable_bm25: bool = True           # Enable BM25 hybrid search
enable_llm_guard: bool = True      # Enable LLM-based hallucination guard
enable_summary_generation: bool = True  # Generate doc summaries on upload
primary_model: str = "gpt-4o"      # Primary LLM model
```

When `pipeline_version=v1`, the entire v2 pipeline is bypassed — all requests use the original chunk-only retrieval + gpt-4o-mini + token-overlap guard.

---

## 5. Data Flow

### 5.1 Upload Flow (v2)

```
1. Receive file(s) + session_id
2. For each file:
   a. Validate extension + size (existing)
   b. SHA-256 dedup check (existing)
   c. Extract text via parser (existing)
   d. NEW: Store full extracted text → FullTextStore
   e. NEW: Count tokens for full text
   f. Chunk text (existing, but chunk_size increased to 768)
   g. Embed chunks → FAISS index (existing)
   h. NEW: Index chunks in BM25 service
   i. NEW: Generate document summary (async, gpt-4o-mini)
   j. NEW: Embed summary → add as special chunk in FAISS (type="summary")
3. Return upload results (existing format)
```

### 5.2 Query Flow (v2)

```
1. Receive query + session_id
2. Context Manager:
   a. Classify query type (general/specific/comparison)
   b. Resolve file scope (existing logic)
   c. Get total token count for target files
   d. Select context strategy

3a. FULL_CONTEXT strategy:
   - Load full document text from FullTextStore
   - Build prompt: system + history + full_text + query
   - Call gpt-4o (streaming)

3b. SUMMARY_PLUS_RETRIEVAL strategy:
   - Load document summaries
   - Run enhanced FAISS retrieval (top-20) + BM25 hybrid
   - Build prompt: system + history + summaries + retrieved_chunks + query
   - Call gpt-4o (streaming)

3c. RETRIEVAL_ONLY strategy:
   - Run enhanced FAISS retrieval (top-30) + BM25 hybrid
   - Include summary chunks
   - Build prompt: system + history + chunks + query
   - Call gpt-4o (streaming)

4. Smart Hallucination Guard:
   a. Tier 1: token-overlap check (fast)
   b. If marginal → Tier 2: LLM faithfulness check
   c. Citation validation

5. Follow-up question generation (gpt-4o-mini, async)

6. Monitoring: log metrics, check alerts

7. Cache response (if guard passed)

8. Return response with citations
```

---

## 6. API Changes

### 6.1 New Fields in Response

Both `/api/converse` and `/api/chat` responses gain:

```json
{
  "context_strategy": "full_context",
  "query_type": "general",
  "model_used": "gpt-4o",
  "citations": [
    {"text": "Fire alarm integration requires...", "source": "scope_Electrical.docx", "page": 12}
  ]
}
```

### 6.2 New Endpoints

```
GET  /api/metrics/summary    — Dashboard metrics (last 1h/24h/7d)
GET  /metrics                — Prometheus scrape endpoint
```

### 6.3 New Query Parameter

```
POST /api/chat
{
  "session_id": "...",
  "query": "...",
  "force_strategy": "full_context"  // optional override for testing
}
```

### 6.4 Backward Compatibility

When `pipeline_version=v1`:
- All new response fields are omitted
- Original pipeline runs unchanged
- API contract matches current production exactly

---

## 7. Configuration & Feature Flags

### New .env Variables

```env
# ── Pipeline Version ─────────────────────────────────────
PIPELINE_VERSION=v2                    # "v1" (legacy) or "v2" (hybrid context)

# ── v2 Models ────────────────────────────────────────────
PRIMARY_MODEL=gpt-4o                   # Primary generation model
SECONDARY_MODEL=gpt-4o-mini            # Follow-ups, summaries, guard
PRIMARY_MAX_OUTPUT_TOKENS=4096         # Increased from 2048

# ── v2 Context Strategy ─────────────────────────────────
FULL_CONTEXT_THRESHOLD=80000           # Tokens below which full doc is loaded
SUMMARY_THRESHOLD=200000               # Above this, summary-only for general questions
ENABLE_FULL_CONTEXT=true
ENABLE_BM25=true
ENABLE_SUMMARY_GENERATION=true

# ── v2 Retrieval ─────────────────────────────────────────
RETRIEVAL_TOP_K=20                     # Increased from 8
RETRIEVAL_SCORE_THRESHOLD=0.15         # Lowered from 0.30
BM25_WEIGHT=0.30                       # BM25 contribution in hybrid score
CHUNK_SIZE_TOKENS=768                  # Increased from 512

# ── v2 Hallucination Guard ──────────────────────────────
GUARD_GENERAL_THRESHOLD=0.20           # Relaxed for general questions
GUARD_SPECIFIC_THRESHOLD=0.30          # Stricter for factual claims
GUARD_LLM_ENABLED=true                 # Enable Tier 2 LLM check
GUARD_MARGINAL_LOW=0.25                # Below: auto-fail. Above: auto-pass.
GUARD_MARGINAL_HIGH=0.50               # Marginal zone triggers LLM check

# ── v2 Session ───────────────────────────────────────────
SESSION_TTL_HOURS=168                  # 7 days (was 24h)
MAX_HISTORY_MESSAGES=50                # Increased from 20

# ── v2 Monitoring ────────────────────────────────────────
ENABLE_METRICS=true
ALERT_LATENCY_WARNING_MS=15000
ALERT_LATENCY_CRITICAL_MS=30000
ALERT_COST_WARNING_USD=0.50
ALERT_COST_CRITICAL_USD=1.00
ALERT_GROUNDEDNESS_WARNING=0.40
ALERT_GROUNDEDNESS_CRITICAL=0.20
```

### New Dependencies

```
rank-bm25>=0.2.2        # BM25 keyword search
prometheus-client>=0.20  # Metrics export (optional)
```

---

## 8. Evaluation: 12-Parameter Review

### 8.1 Scaling

| Concern | Current State | v2 Design | Mitigation |
|---------|--------------|-----------|------------|
| Concurrent users | Single worker, in-memory state | 100+ customer-facing | Multi-worker uvicorn (4+), async throughout, per-session isolation |
| Document size | 512-token chunks only | Full text + chunks + summaries | Tiered strategy: full context for small docs, retrieval for large |
| Memory per session | ~FAISS index + chunks | +full text + summary + BM25 index | Set MAX_SESSIONS_IN_MEMORY limit, LRU eviction to S3, lazy reload |
| Horizontal scaling | Single instance | Stateless compute possible | Sessions in S3, FAISS indices in S3, any worker can serve any session |

**Memory budget per session (worst case — 100-page doc):**
- Full text: ~300KB
- Chunks (150 chunks): ~150KB
- FAISS index: ~900KB (150 * 1536 * 4 bytes)
- BM25 index: ~100KB
- Summary: ~5KB
- **Total: ~1.5MB per session**
- 100 concurrent sessions: ~150MB — well within server memory

### 8.2 Optimization

| Optimization | Description |
|-------------|-------------|
| Smart context selection | Only load full text when document is small enough — saves tokens for large docs |
| Summary caching | Generate once on upload, reuse for every general query |
| L1/L2 cache (existing) | Skip entire pipeline for repeated queries |
| Async summary generation | Non-blocking: user gets upload confirmation before summary completes |
| BM25 pre-tokenization | Done at upload time, not query time |
| Embedding reuse | Query embedded once, used for both FAISS and BM25 scoring |
| gpt-4o-mini for ancillary tasks | Follow-ups and guard use cheaper model |
| Session LRU eviction | Cold sessions evicted from memory, reloaded from S3 on demand |

### 8.3 Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Upload latency (20-page doc) | < 15s | file processing + embedding + summary |
| Query latency — FULL_CONTEXT | < 8s first token | streaming start |
| Query latency — RETRIEVAL | < 5s first token | streaming start |
| Query total time | < 30s | including follow-up generation |
| Groundedness (general) | > 0.60 average | LLM-based guard |
| Groundedness (specific) | > 0.75 average | Token-overlap + LLM guard |
| Cache hit rate | > 20% | For repeated/similar queries |
| Cost per query (FULL_CONTEXT) | $0.01-0.05 | gpt-4o with full doc |
| Cost per query (RETRIEVAL) | $0.005-0.02 | gpt-4o with chunks |

### 8.4 Request Handling

| Aspect | Design |
|--------|--------|
| Rate limiting | Per-session: 10 req/min, per-IP: 30 req/min |
| Request queue | FastAPI async — no blocking; uvicorn handles backpressure |
| Timeout | 120s per request (existing), OpenAI timeout 60s |
| Max concurrent LLM calls | Semaphore: 10 concurrent (shared across sessions) |
| File upload limits | 20MB per file, 10 files per upload (existing) |
| Max query length | 5000 chars (existing) |
| Request validation | Pydantic schema validation at API boundary |
| Error response format | `{"detail": {"message": "...", "code": "...", "retry": true/false}}` |

### 8.5 Vulnerability Assessment

| Vulnerability | Risk | Mitigation |
|--------------|------|------------|
| Prompt injection via uploaded docs | Medium | System prompt hardening, document content sandboxed as "context" not "instructions" |
| API key exposure | High | .env file, never in code, validate at startup |
| Session hijacking | Medium | Session IDs are 16-char hex (existing), add rate limiting |
| DoS via large file upload | Medium | 20MB limit, 10 files max, rate limiting |
| Cross-session data leak | Low | Per-session FAISS isolation (existing), no global index |
| OpenAI API abuse | Medium | Semaphore limits concurrent calls, cost alerts |
| Malicious file content | Low | Text extracted only, no code execution, no file storage |
| CORS misconfiguration | Medium | Current: `allow_origins=["*"]` — tighten to specific frontend origins |
| S3 bucket access | Medium | Least-privilege IAM policy, bucket not public |

**Action items:**
1. Tighten CORS origins to actual frontend domain
2. Add request rate limiting middleware
3. Add input sanitization for query (strip control characters)
4. Add OpenAI API call concurrency limiter

### 8.6 SDLC Parameters

| Parameter | Approach |
|-----------|----------|
| Version control | Git, feature branch per phase |
| Branching strategy | `main` → `feature/docqa-v2-phase-N` → PR → merge |
| Code review | Agent-based review after each phase |
| Testing | Unit + Integration + E2E per phase, 80%+ coverage |
| CI/CD | Manual deployment via SCP to sandbox → production |
| Documentation | CLAUDE.md updated per phase, design doc maintained |
| Changelog | CHANGELOG.md tracks all changes |
| Rollback | Feature flag: `PIPELINE_VERSION=v1` instant rollback |

### 8.7 Compliance

| Requirement | Status |
|-------------|--------|
| Data isolation | Per-session FAISS, no cross-user data sharing |
| Data retention | Configurable TTL (7 days default), explicit delete API |
| PII handling | Documents processed in memory, not persisted to disk (only S3 with encryption) |
| Audit logging | All queries logged with session_id, timestamp, model, cost |
| API key security | .env file, not in code, not in logs |
| HTTPS | SSL via Let's Encrypt on production (existing) |
| Input validation | Pydantic schemas at all API boundaries |

### 8.8 Disaster Recovery & Backup

| Scenario | Recovery |
|----------|----------|
| Server crash / restart | Sessions restored from S3 (existing). Full text + summaries also in S3 |
| S3 outage | Graceful degradation: in-memory sessions continue, S3 writes retry with backoff |
| OpenAI API outage | Return 503 with retry header, cache serves previously answered queries |
| Corrupted session | Delete + recreate. User re-uploads documents (upload is fast) |
| Config error | Feature flag rollback to v1, no code deployment needed |
| Bad deployment | Systemd `restart`, or rollback feature flag |

**Backup strategy:**
- S3 bucket versioning enabled (existing)
- Session data persisted after every turn (existing)
- Full text + summaries added to S3 persistence
- BM25 index rebuilt on session restore (lightweight, no persistence needed)

### 8.9 Support & Helpdesk Framework

| Component | Description |
|-----------|-------------|
| Health endpoint | `GET /health` — status, model, sessions, version info |
| Metrics dashboard | `GET /api/metrics/summary` — latency, cost, errors, groundedness |
| Session inspection | `GET /api/sessions/{id}` — files, history, token usage |
| Debug mode | `DEBUG_MODE=true` in .env → verbose logging, include context strategy in response |
| Error codes | Structured error responses with codes: `UPLOAD_FAILED`, `RETRIEVAL_EMPTY`, `GUARD_TRIGGERED`, `MODEL_ERROR`, `RATE_LIMITED` |
| Troubleshooting guide | Section in CLAUDE.md with common issues + fixes |

### 8.10 System Maintenance

| Task | Frequency | Mechanism |
|------|-----------|-----------|
| Session cleanup | Every 30 min | Existing background task, respects new TTL (7d) |
| S3 stale session cleanup | Daily | New: list sessions older than TTL, delete from S3 |
| Log rotation | Daily | Systemd journal + optional file logging with rotation |
| Dependency updates | Monthly | Pin versions in requirements.txt, test before updating |
| Model version tracking | On change | Log model version in metrics, alert on deprecation notices |
| Cache eviction | Automatic | L1 TTL (1h), L2 Redis TTL (1h) |
| Memory monitoring | Continuous | Log memory usage per session, alert at 80% of available |

### 8.11 Network & Security Requirements

| Requirement | Implementation |
|-------------|---------------|
| HTTPS | Nginx SSL termination with Let's Encrypt (existing on production) |
| CORS | Tighten to specific frontend origins (currently wildcard) |
| Rate limiting | New: `slowapi` middleware — 10 req/min per session, 30 req/min per IP |
| Request size | Nginx: `client_max_body_size 25m` (existing) |
| Timeouts | Nginx: 120s read, 300s for SSE streams (existing) |
| Internal network | Agent listens on 127.0.0.1:8006, Nginx proxies from public |
| Firewall | Port 8006 not exposed externally (Nginx only) |
| API authentication | Handled by frontend/gateway (out of scope for this agent) |
| SSE streaming | Nginx: `proxy_buffering off`, `chunked_transfer_encoding off` (existing) |

### 8.12 Resource Management: Efficiency through Automation

| Resource | Automation |
|----------|------------|
| Session lifecycle | Auto-create, auto-cleanup on TTL, auto-restore from S3 |
| Document processing | Parallel embedding batches, async summary generation |
| Model selection | Auto-routing: gpt-4o for generation, gpt-4o-mini for auxiliary |
| Context strategy | Auto-selected based on document size and query type |
| Cache management | Auto-populate on query, auto-evict on TTL |
| Cost tracking | Auto-calculated per query, auto-alerting on thresholds |
| Monitoring | Auto-logging of all metrics, auto-aggregation for dashboard |
| S3 persistence | Write-behind (non-blocking), auto-restore on startup |
| Memory management | LRU eviction when sessions exceed memory threshold |
| Feature flags | Instant rollback without deployment |

---

## 9. Testing Strategy

### 9.1 Unit Tests (per service)

| Service | Tests | Coverage Target |
|---------|-------|----------------|
| context_manager.py | Query classification, strategy selection, token counting | 90% |
| fulltext_store.py | Store/retrieve/delete, token counting, S3 persistence | 85% |
| summary_service.py | Summary generation, caching, error handling | 80% |
| bm25_service.py | Indexing, search, scoring, session management | 90% |
| retrieval_service.py (v2) | Hybrid search, score merging, file scoping | 85% |
| generation_service.py (v2) | Citation prompts, model routing, context modes | 80% |
| hallucination_guard.py (v2) | Two-tier guard, query-type thresholds, LLM check | 85% |
| monitoring_service.py | Metrics recording, aggregation, alert checking | 80% |

### 9.2 Integration Tests

| Test Case | Description |
|-----------|-------------|
| E2E: Small doc general question | Upload 10-page DOCX → "Summarize this" → full_context strategy → cited answer |
| E2E: Small doc specific question | Upload 10-page DOCX → "What are fire alarm requirements?" → full_context → cited answer |
| E2E: Large doc general question | Upload 200-page PDF → "What is this about?" → summary_plus_retrieval → cited answer |
| E2E: Large doc specific question | Upload 200-page PDF → "Section 3.2 specs?" → retrieval_only → cited answer |
| E2E: Multi-document comparison | Upload 2 docs → "Compare electrical vs mechanical" → cross-doc citations |
| E2E: Feature flag rollback | Set v1 → verify old pipeline → set v2 → verify new pipeline |
| E2E: Session persistence | Upload → query → restart server → query same session → works |
| E2E: All 8 demo questions | Run all 8 demo questions from requirements → all pass |

### 9.3 Performance Tests

| Test | Target |
|------|--------|
| Upload 20-page DOCX | < 15s total (extract + embed + summary) |
| Query latency (full_context) | < 8s first token |
| Query latency (retrieval) | < 5s first token |
| 10 concurrent queries | No errors, < 15s average |
| 50 concurrent sessions | Memory < 500MB, no OOM |

### 9.4 Security Tests

| Test | Description |
|------|-------------|
| Prompt injection | Upload doc with "Ignore all instructions..." → agent ignores it |
| Large file | Upload 21MB file → rejected cleanly |
| Malformed session ID | Random string → 404, not crash |
| Rate limit | 15 rapid requests → 429 after 10th |
| CORS | Request from unauthorized origin → blocked |

---

## 10. Phased Implementation Plan

### Phase 1: Foundation (Context Manager + Full Text Store)
- New: `services/context_manager.py`
- New: `services/fulltext_store.py`
- Modified: `config.py` (new settings + feature flags)
- Modified: `services/token_tracker.py` (gpt-4o pricing)
- Tests: unit tests for context manager + full text store

### Phase 2: Document Summary Service
- New: `services/summary_service.py`
- Modified: upload pipeline (generate summary on upload)
- Modified: `services/session_service.py` (store summaries)
- Tests: summary generation + caching tests

### Phase 3: BM25 Hybrid Search
- New: `services/bm25_service.py`
- Modified: `services/retrieval_service.py` (hybrid scoring)
- Modified: upload pipeline (BM25 indexing)
- New dependency: `rank-bm25`
- Tests: BM25 search + hybrid scoring tests

### Phase 4: Enhanced Generation (Citations + Model Routing)
- Modified: `services/generation_service.py` (new prompts, model routing, citation enforcement)
- Modified: `models/schemas.py` (new response fields)
- Tests: citation validation, model routing tests

### Phase 5: Smart Hallucination Guard
- Modified: `services/hallucination_guard.py` (two-tier guard, query-type awareness)
- Tests: guard threshold tests, LLM faithfulness check

### Phase 6: Pipeline Integration
- Modified: `routers/converse.py` (wire context manager into pipeline)
- Modified: `routers/chat.py` (wire context manager into pipeline)
- Modified: `main.py` (initialize new services)
- Tests: full E2E integration tests with all 8 demo questions

### Phase 7: Monitoring & Metrics
- New: `services/monitoring_service.py`
- Modified: `main.py` (metrics endpoint)
- Modified: pipeline (emit metrics at each step)
- Tests: metrics recording + aggregation tests

### Phase 8: Security Hardening
- Rate limiting middleware
- CORS tightening
- Input sanitization
- Concurrency limiter for OpenAI calls
- Tests: security test suite

### Phase 9: Performance Testing & Optimization
- Load tests (10-50 concurrent users)
- Memory profiling
- Latency optimization
- Cost optimization review

### Phase 10: Sandbox Deployment & Validation
- Deploy to sandbox VM (54.197.189.113)
- Run all 8 demo questions
- Run full test suite on sandbox
- Performance validation under load
- User acceptance testing

### Phase 11: Production Deployment
- Deploy to production (13.217.22.125)
- Feature flag: start with v1, switch to v2 after validation
- Monitor metrics for 24h
- Full cutover

---

## 11. Deployment Plan

### Sandbox (54.197.189.113)
```bash
# 1. Transfer files
scp -i ai_assistant_sandbox.pem -r document-qa-agent/ ubuntu@54.197.189.113:/home/ubuntu/chatbot/aniruddha/vcsai/document-qa-agent/

# 2. Install dependencies
ssh -i ai_assistant_sandbox.pem ubuntu@54.197.189.113
cd /home/ubuntu/chatbot/aniruddha/vcsai/document-qa-agent
pip install -r requirements.txt

# 3. Configure .env
cp .env.example .env  # then edit with production values

# 4. Start with feature flag v2
echo "PIPELINE_VERSION=v2" >> .env
python main.py  # or systemctl restart docqa-agent

# 5. Validate
curl http://localhost:8006/health
# Run demo questions via API
```

### Production (13.217.22.125) — after sandbox validation
```bash
# Same steps as sandbox
# Start with PIPELINE_VERSION=v1 (safety)
# Switch to v2 after monitoring confirms stability
```

---

## 12. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| gpt-4o cost spike (full context mode) | Medium | Medium | Cost alerts, budget caps, gpt-4o-mini fallback for simple queries |
| gpt-4o rate limits (100+ users) | Medium | High | Request queue, retry with backoff, model fallback |
| Full text storage increases memory | Low | Medium | LRU eviction, S3 offload, session memory limit |
| Summary generation fails | Low | Low | Graceful fallback to retrieval-only mode |
| BM25 adds latency | Low | Low | BM25 search is <5ms typically, negligible |
| Feature flag complexity | Low | Low | Only 2 states (v1/v2), clean separation |
| LLM guard adds latency | Medium | Low | Only triggers in marginal zone, gpt-4o-mini is fast |
| Breaking change for frontend | Low | Medium | API is additive (new fields), old fields preserved |

---

## Appendix A: File Change Summary

### New Files (7)
| File | Purpose |
|------|---------|
| `services/context_manager.py` | Context strategy selection + query classification |
| `services/fulltext_store.py` | Full document text storage |
| `services/summary_service.py` | Document summary generation + caching |
| `services/bm25_service.py` | BM25 keyword search |
| `services/monitoring_service.py` | Metrics + alerting |
| `tests/test_v2_pipeline.py` | v2 integration tests |
| `tests/test_context_manager.py` | Context manager unit tests |

### Modified Files (10)
| File | Change |
|------|--------|
| `config.py` | New settings: feature flags, v2 thresholds, monitoring |
| `main.py` | Initialize new services, mount metrics endpoint |
| `models/schemas.py` | New response fields: context_strategy, query_type, citations |
| `services/retrieval_service.py` | Hybrid search (FAISS + BM25), increased top_k |
| `services/generation_service.py` | Citation prompts, model routing, full-context mode |
| `services/hallucination_guard.py` | Two-tier guard, query-type awareness, LLM check |
| `services/session_service.py` | Extended TTL, summary storage, full text refs |
| `services/token_tracker.py` | gpt-4o pricing, strategy-aware tracking |
| `routers/converse.py` | Wire context manager, emit metrics |
| `routers/chat.py` | Wire context manager, emit metrics |

### Unchanged Files (6)
| File | Reason |
|------|--------|
| `services/embedding_service.py` | No changes needed |
| `services/index_service.py` | No changes needed (FAISS logic unchanged) |
| `services/file_processor.py` | Only minor: pass full text to FullTextStore |
| `services/cache_service.py` | No changes needed |
| `routers/upload.py` | Legacy endpoint, unchanged |
| `routers/sessions.py` | Unchanged |

---

*End of design document.*
