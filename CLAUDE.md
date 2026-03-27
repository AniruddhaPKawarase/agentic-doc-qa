# Document Q&A Agent — System Design & Development Guide

---

## File-Aware Retrieval Redesign v2 — 2026-03-26 (APPROVED, PENDING DEVELOPMENT)

**Design doc:** [docs/superpowers/specs/2026-03-26-docqa-retrieval-redesign.md](docs/superpowers/specs/2026-03-26-docqa-retrieval-redesign.md)

### Problem
When multiple files exist in a session, the agent answers from the WRONG file. FAISS retrieval returns top-k chunks from ALL files by cosine similarity with no file-level scoping. User uploads "Air Diffusers.pdf" but gets answer about "A101-FLOOR-PLAN.pdf".

### Solution: File Scope Resolver + Post-Retrieval Filter + File-Aware Prompt
| Scenario | Scope | Behavior |
|----------|-------|----------|
| Upload + ask | Current file only | Strict scoping to just-uploaded file |
| Follow-up, no file | Global | Search all session files |
| Mentions past file name | Referenced file | Detect file name in query → scope to that file |
| Duplicate upload | Existing copy | Scope answer to already-indexed chunks |

### Confirmed Decisions
- Single FAISS index with metadata-based file filtering (not per-file indexes)
- Pre-filter chunks + inject filename into system prompt (both)
- Follow-up without file → global search across all session files
- User mentions file name → intelligent context switching (like human conversation)

### Files to Modify
- `services/retrieval_service.py` — add `resolve_file_scope()`, file-filtered `retrieve()`
- `services/generation_service.py` — inject current file name + file awareness rules into prompt
- `routers/converse.py` — pass uploaded file names through pipeline
- `routers/chat.py` — preserve global search behavior

---

## Overview

**Port:** 8006 | **Gateway Prefix:** `/docqa/` | **Model:** gpt-4o-mini
**Purpose:** Users upload documents (PDF, Word, Excel, TXT, images) and ask questions answered
strictly from uploaded content. Out-of-context questions are rejected.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Nginx (port 8000)                            │
│                     prefix: /docqa/ → :8006                         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                    FastAPI App (port 8006)                           │
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ Upload Router │  │ Chat Router  │  │ Session Router           │  │
│  │ POST /upload  │  │ POST /chat   │  │ GET /sessions            │  │
│  │               │  │ POST /stream │  │ GET /sessions/{id}       │  │
│  │               │  │              │  │ DELETE /sessions/{id}     │  │
│  │               │  │              │  │ GET /sessions/{id}/files  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────┘  │
│         │                 │                                         │
│  ┌──────▼───────┐  ┌──────▼─────────────────────────────────────┐  │
│  │ File         │  │ Generation Pipeline                        │  │
│  │ Processor    │  │                                            │  │
│  │              │  │  Query → Retrieval → LLM → Guard → Stream  │  │
│  │ PDF/Word/    │  │         (FAISS)    (gpt-4o   (token       │  │
│  │ Excel/TXT    │  │                    -mini)    overlap)      │  │
│  │              │  │                                            │  │
│  │ → Chunking   │  └────────────────────────────────────────────┘  │
│  │ → Embedding  │                                                  │
│  │ → FAISS idx  │  ┌────────────────────────────────────────────┐  │
│  └──────────────┘  │ Shared Services                            │  │
│                    │  SessionService  │ TokenTracker             │  │
│                    │  CacheService    │ HallucinationGuard       │  │
│                    └────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Flow (per chat request)

```
1. Receive query + session_id
2. Load session (conversation history + FAISS index)
3. Check L1/L2 cache for semantic match → return if hit
4. Embed query with text-embedding-3-small
5. FAISS similarity search (top-k=8, threshold=0.30)
6. If no relevant chunks found → reject as out-of-context
7. Build LLM prompt: system + history + context chunks + query
8. Enforce token budget (input ≤ 100k tokens)
9. Call gpt-4o-mini (streaming or blocking)
10. Hallucination guard: token-overlap score against source chunks
11. If groundedness < 0.65 → replace with clarification questions
12. Generate 3+ follow-up questions from context
13. Track tokens (input/output/embedding) per step
14. Cache response (skip if hallucinated)
15. Store turn in session history
16. Return response (or stream SSE chunks)
```

---

## API Endpoints

### Unified Converse (NEW — UI-Friendly)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/converse` | **Unified endpoint.** Multipart form: `files[]` (required) + `query` (required) + `session_id` (optional form override). File(s) are uploaded, embedded, and indexed first; then query is answered from indexed content. Session managed via `docqa_session_id` cookie (set on every response). Returns `ConverseResponse` (upload results + full Q&A answer). |
| `POST` | `/api/converse/stream` | SSE streaming variant of `/api/converse`. Same request format. Streams `upload_done` → `sources` → `chunk`… → `done` events. Cookie is set on SSE response headers. |

> **Design doc:** See `DESIGN_CONVERSE_API.md` for full architecture, sequence diagrams, error handling, and DSA rationale.

### Upload (legacy — kept for backward compatibility)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/upload` | Upload files to a session. Multipart form: `files[]` + `session_id` (optional, auto-created). Max 20MB per file. Returns session_id + file processing status. |

### Chat (legacy — kept for backward compatibility)
| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/chat` | Blocking Q&A. Body: `{session_id, query}`. Returns answer + sources + follow_ups + token_usage + groundedness_score. |
| `POST` | `/api/chat/stream` | SSE streaming. Same body. Streams `data: {chunk}` events, final event has metadata. |

### Sessions
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/sessions` | List all active sessions (id, created_at, file_count, message_count). |
| `GET` | `/api/sessions/{id}` | Session detail: files, history, token totals. |
| `DELETE` | `/api/sessions/{id}` | Delete session, its FAISS index, and uploaded file chunks. |
| `GET` | `/api/sessions/{id}/files` | List files in session with processing status. |

### System
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check: status, model available, active sessions count. |
| `GET` | `/` | Service info and endpoint listing. |

---

## File Processing Pipeline

### Supported Formats
| Format | Library | Strategy |
|--------|---------|----------|
| PDF | `pdfplumber` | Page-by-page text extraction. Falls back to `PyPDF2` if pdfplumber fails. |
| Word (.docx) | `python-docx` | Paragraph + table extraction. Preserves heading structure. |
| Excel (.xlsx/.xls) | `openpyxl` | Sheet-by-sheet, row-by-row. Headers become field labels. |
| TXT/CSV/JSON | Built-in | Direct read with encoding detection (`chardet`). |

### Chunking Strategy
- **Chunk size:** 512 tokens (measured by `tiktoken` cl100k_base)
- **Overlap:** 64 tokens (12.5%)
- **Splitting:** Sentence-boundary aware (split on `.!?` then merge to fill chunk)
- **Metadata per chunk:** `{file_name, file_type, chunk_index, page_number (if PDF), sheet_name (if Excel), char_start, char_end}`

### Embedding
- **Model:** `text-embedding-3-small` (1536 dimensions)
- **Batch size:** 100 chunks per API call
- **Rate limiting:** Max 3 concurrent embedding calls

### FAISS Index
- **One index per session** (not global — isolates user data)
- **Index type:** `IndexFlatIP` (inner product on L2-normalized vectors = cosine similarity)
- **Storage:** In-memory dict `{session_id: {index, metadata_list}}`
- **Cleanup:** Sessions expire after `SESSION_TTL_HOURS` (default 24h)

---

## Converse API — Cookie Session Management

The unified `/api/converse` endpoint manages session state via an `HttpOnly` cookie.

- **Cookie name:** `docqa_session_id`
- **Priority:** form field `session_id` > cookie `docqa_session_id` > new session created
- **Refresh:** Cookie TTL is refreshed on every `/api/converse` response (session stays alive during active use)
- **Prod config:** Set `COOKIE_SECURE=true` in `.env` when running behind HTTPS

**How the UI uses it:**
1. First request: no cookie → backend creates session → response sets cookie
2. Follow-up requests: browser auto-sends cookie → backend resolves same session → continues conversation
3. New conversation: UI sends `session_id=""` in form body to force a new session (overrides cookie)

## File Deduplication

The `Session` object tracks SHA-256 hashes of all uploaded file contents.
Uploading the same file twice returns `status="duplicate_skipped"` with zero cost.
Hash is computed from raw file bytes before any processing.

---

## Environment Variables

```env
# ── Server ────────────────────────────────────────────
DOCQA_HOST=0.0.0.0
DOCQA_PORT=8006

# ── OpenAI ────────────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_MAX_OUTPUT_TOKENS=2048

# ── Retrieval ─────────────────────────────────────────
RETRIEVAL_TOP_K=8
RETRIEVAL_SCORE_THRESHOLD=0.30
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=64
EMBEDDING_BATCH_SIZE=100
MAX_CONTEXT_TOKENS=80000

# ── Upload ────────────────────────────────────────────
MAX_FILE_SIZE_MB=20
MAX_FILES_PER_UPLOAD=10
ALLOWED_EXTENSIONS=.pdf,.docx,.xlsx,.xls,.txt,.csv,.json

# ── Session ───────────────────────────────────────────
SESSION_TTL_HOURS=24
MAX_HISTORY_MESSAGES=20
SESSION_CLEANUP_INTERVAL_MIN=30

# ── Hallucination Guard ──────────────────────────────
HALLUCINATION_GUARD_ENABLED=true
GROUNDEDNESS_THRESHOLD=0.65
MIN_FOLLOWUP_QUESTIONS=3

# ── Cache ─────────────────────────────────────────────
CACHE_L1_MAXSIZE=500
CACHE_L1_TTL=3600
REDIS_URL=redis://localhost:6379/2
CACHE_SEMANTIC_ENABLED=true

# ── Token Budget ──────────────────────────────────────
MAX_INPUT_TOKENS=100000
TOKEN_WARNING_THRESHOLD=0.85

# ── Cookie (for /api/converse unified endpoint) ───────
COOKIE_SECURE=false      # set true in production (HTTPS only)
```

---

## Project Structure

```
document-qa-agent/
├── CLAUDE.md                    ← This file
├── DESIGN_CONVERSE_API.md       ← Unified converse API design doc (user story, Q&A, architecture)
├── .env                         ← Environment config
├── requirements.txt             ← Python dependencies
├── main.py                      ← FastAPI app + lifespan + mount routers
│
├── config.py                    ← Settings(BaseSettings) + @lru_cache getter
│
├── models/
│   ├── __init__.py
│   └── schemas.py               ← UploadResponse, ChatRequest, ChatResponse, ConverseResponse, etc.
│
├── routers/
│   ├── __init__.py
│   ├── upload.py                ← POST /api/upload  (legacy, unchanged)
│   ├── chat.py                  ← POST /api/chat, POST /api/chat/stream  (legacy, unchanged)
│   ├── converse.py              ← POST /api/converse, POST /api/converse/stream  (NEW — unified)
│   └── sessions.py              ← Session CRUD endpoints
│
├── services/
│   ├── __init__.py
│   ├── file_processor.py        ← PDF/Word/Excel/TXT extraction + chunking
│   ├── embedding_service.py     ← OpenAI embedding with batching + rate limit
│   ├── index_service.py         ← FAISS index management (create/query/delete per session)
│   ├── retrieval_service.py     ← Query embedding → FAISS search → ranked chunks
│   ├── generation_service.py    ← Prompt building → LLM call → streaming
│   ├── session_service.py       ← Session CRUD, history, anti-hallucination filter + file dedup hashes
│   ├── hallucination_guard.py   ← Token-overlap groundedness scoring
│   ├── token_tracker.py         ← PipelineTokenLog + TokenTracker
│   └── cache_service.py         ← L1 TTLCache + L2 Redis, semantic normalization
│
└── tests/
    ├── __init__.py
    └── test_upload.py            ← Basic integration tests
```

---

## Key Design Decisions

### 1. Per-Session FAISS (not global)
Each user session gets its own FAISS index. This ensures:
- Complete data isolation between users
- No cross-contamination of document context
- Simple cleanup (delete session = delete index)
- Scales horizontally (sessions are independent)

### 2. Strict Out-of-Context Rejection
If FAISS retrieval returns no chunks above `RETRIEVAL_SCORE_THRESHOLD` (0.30),
the agent responds: *"I can only answer questions based on your uploaded documents.
Your question doesn't appear to be covered. Could you rephrase or upload relevant documents?"*

### 3. Hallucination Rollback
Token-overlap scoring compares LLM output tokens against source chunk tokens.
If `groundedness < 0.65`:
- Replace answer with: *"I'm not confident enough to answer accurately. Let me ask some clarifying questions:"*
- Return 3+ clarification questions derived from the source chunks
- Do NOT cache the hallucinated response
- Filter low-groundedness turns from session history (prevents cascading hallucination)

### 4. Streaming for Perceived Latency
SSE streaming starts sending tokens within ~200ms of LLM response start.
Hallucination guard runs on accumulated text after stream completes.
If guard triggers, a final SSE event overwrites with clarification.

### 5. Token Tracking Granularity
Every request logs per-step:
```json
{
  "embedding_tokens": 45,
  "retrieval_ms": 12,
  "prompt_tokens": 3200,
  "completion_tokens": 512,
  "total_tokens": 3757,
  "estimated_cost_usd": 0.0008,
  "cache_hit": false,
  "pipeline_ms": 1450
}
```

---

## Thresholds & Tuning Reference

| Parameter | Default | Effect |
|-----------|---------|--------|
| `RETRIEVAL_TOP_K` | 8 | More chunks = more context but higher token cost |
| `RETRIEVAL_SCORE_THRESHOLD` | 0.30 | Lower = more permissive retrieval, higher = stricter |
| `GROUNDEDNESS_THRESHOLD` | 0.65 | Lower = more answers pass, higher = more rollbacks |
| `CHUNK_SIZE_TOKENS` | 512 | Larger = fewer chunks but less precise retrieval |
| `CHUNK_OVERLAP_TOKENS` | 64 | More overlap = better boundary coverage |
| `MAX_HISTORY_MESSAGES` | 20 | Sliding window size for conversation memory |
| `CACHE_L1_TTL` | 3600 | How long cached answers survive (seconds) |
| `MAX_CONTEXT_TOKENS` | 80000 | Token budget for retrieved chunks in prompt |
| `MIN_FOLLOWUP_QUESTIONS` | 3 | Minimum follow-up questions per response |

---

## Gateway Integration

### Nginx Location Block (add to vcs-agents.conf)
```nginx
upstream docqa_agent {
    server 127.0.0.1:8006;
    keepalive 8;
}

# Standard endpoints
location /docqa/ {
    client_max_body_size 25m;
    proxy_pass http://docqa_agent/;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header Connection '';
    proxy_read_timeout 120s;
}

# SSE streaming endpoint
location /docqa/api/chat/stream {
    proxy_pass http://docqa_agent/api/chat/stream;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header Connection '';
    proxy_buffering off;
    proxy_cache off;
    chunked_transfer_encoding off;
    proxy_read_timeout 300s;
}
```

### systemd Service
```ini
[Unit]
Description=VCS Document Q&A Agent (Port 8006)
After=network.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory=/home/ubuntu/PROD_SETUP/document-qa-agent
EnvironmentFile=/home/ubuntu/PROD_SETUP/document-qa-agent/.env
ExecStart=/home/ubuntu/PROD_SETUP/document-qa-agent/venv/bin/python main.py
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=docqa-agent
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

### Gateway Health Registry (add to health_service/main.py AGENTS dict)
```python
"docqa": {
    "name": "Document Q&A Agent",
    "url": "http://127.0.0.1:8006",
    "health_path": "/health",
    "port": 8006,
    "prefix": "/docqa/",
    "description": "Upload documents and ask questions — RAG with per-session FAISS",
},
```

---

## Development Phases

### Original Build (complete)
| Phase | Scope | Files |
|-------|-------|-------|
| 1 | Project structure, config, schemas, dependencies | `config.py`, `models/schemas.py`, `requirements.txt`, `.env` |
| 2 | File processing (extract + chunk + embed) | `services/file_processor.py`, `services/embedding_service.py` |
| 3 | Retrieval layer (FAISS index + search) | `services/index_service.py`, `services/retrieval_service.py` |
| 4 | Generation layer (LLM + streaming + follow-ups) | `services/generation_service.py` |
| 5 | Session, token tracker, hallucination guard, cache | `services/session_service.py`, `services/token_tracker.py`, `services/hallucination_guard.py`, `services/cache_service.py` |
| 6 | API routers (upload, chat, sessions) | `routers/upload.py`, `routers/chat.py`, `routers/sessions.py` |
| 7 | Main app orchestrator | `main.py` |
| 8 | Gateway integration | Nginx config, systemd service, health registry |

### Unified Converse API (UI integration — 2026-03-18)
| Phase | Scope | Files |
|-------|-------|-------|
| C1 | Cookie config + file dedup in session | `config.py` (add `cookie_secure`), `services/session_service.py` (add `file_hashes`, `has_file_hash()`, `add_file_hash()`) |
| C2 | ConverseResponse schema | `models/schemas.py` (add `ConverseResponse`) |
| C3 | Converse router — blocking | `routers/converse.py` (add `POST /api/converse`) |
| C4 | Converse router — streaming | `routers/converse.py` (add `POST /api/converse/stream`) |
| C5 | Mount router | `main.py` (import + `app.include_router(converse.router)`) |

---

## Modification Guide

### Adding a new file format
1. Add parser function in `services/file_processor.py`
2. Add extension to `ALLOWED_EXTENSIONS` in `.env`
3. Register parser in `FileProcessor.PARSERS` dict

### Changing the LLM model
1. Update `OPENAI_CHAT_MODEL` in `.env`
2. Adjust `MAX_CONTEXT_TOKENS` and `OPENAI_MAX_OUTPUT_TOKENS` for new model's limits
3. Token costs in `token_tracker.py` may need updating

### Adjusting hallucination sensitivity
1. `GROUNDEDNESS_THRESHOLD` in `.env`: raise to reject more, lower to accept more
2. Stopword list in `hallucination_guard.py` for domain-specific filtering

### Scaling for high traffic
1. Run multiple uvicorn workers: `--workers 4`
2. Enable Redis L2 cache: set `REDIS_URL` in `.env`
3. Add session persistence (currently in-memory; swap to Redis-backed sessions)
4. Mount persistent storage for FAISS indices if sessions must survive restarts

---

## Phase S3: Storage Migration to AWS S3 (2026-03-20)

### Status: COMPLETED & ACTIVATED (2026-03-23)
- **STORAGE_BACKEND=s3** set in `.env` — S3 session persistence is LIVE
- **Bucket:** `agentic-ai-production` | **Region:** `us-east-1` | **Prefix:** `document-qa-agent/`
- **AWS Key:** `<configured in .env>` (configured in `.env`)
- **Session restore on startup:** enabled (`S3_RESTORE_SESSIONS_ON_STARTUP=true`, max 50)
- Modified: `config.py` (6 new S3 settings including restore options)
- Modified: `services/session_service.py` (S3 persist on create/turn, S3 delete)
- Modified: `services/index_service.py` (S3 FAISS save + chunks JSONL, S3 delete)
- Test suite: `tests/test_s3_docqa.py` (7 tests, all PASS)

### Objective
Add session persistence to S3. Currently this agent is **100% in-memory** — all sessions, FAISS indexes, uploaded file chunks, and conversation history are lost on restart. S3 migration adds persistence without changing the in-memory-first architecture.

### Current Local Storage (what moves to S3)

| Data Type | Current Location | Persistence | S3 Target |
|-----------|-----------------|-------------|-----------|
| Sessions (metadata + history) | In-memory `_sessions` dict | Lost on restart | `document-qa-agent/session_data/{session_id}/session_meta.json` |
| FAISS indexes (per-session) | In-memory `_indices` dict | Lost on restart | `document-qa-agent/session_data/{session_id}/faiss_index.bin` |
| Chunk metadata | In-memory (part of index) | Lost on restart | `document-qa-agent/session_data/{session_id}/chunks.jsonl` |
| Uploaded file content | Processed in memory, discarded | Never persisted | N/A (not stored — re-upload required) |
| Query cache | L1 TTLCache + optional Redis | L1 lost on restart | N/A (cache rebuilt naturally) |
| Logs | Console only | Ephemeral | `document-qa-agent/api_logs/{YYYY-MM-DD}/docqa_agent.log` |

### S3 Folder Structure

```
document-qa-agent/
├── session_data/
│   ├── a1b2c3d4e5f6g7h8/
│   │   ├── session_meta.json          # session metadata, file list, conversation history
│   │   ├── faiss_index.bin            # serialized FAISS index for this session
│   │   └── chunks.jsonl               # chunk text + metadata (one JSON per line)
│   ├── i9j0k1l2m3n4o5p6/
│   │   ├── session_meta.json
│   │   ├── faiss_index.bin
│   │   └── chunks.jsonl
│   └── ...
└── api_logs/
    └── 2026-03-20/
        └── docqa_agent.log
```

### Files to Modify

| File | Change | Details |
|------|--------|---------|
| `config.py` | Add S3 settings | `STORAGE_BACKEND`, `S3_BUCKET_NAME`, `S3_REGION`, AWS creds, `S3_AGENT_PREFIX` |
| `services/session_service.py` | Persist session to S3 on create/update | `_persist_to_s3()`: serialize `Session` → JSON → upload. `_load_from_s3()`: list S3 sessions → restore on startup |
| `services/index_service.py` | Save/load FAISS index to/from S3 | After `index.add()`: save `.bin` to S3. Add `_save_index_to_s3()` and `_load_index_from_s3()`. Chunks serialized as `.jsonl` |
| `services/session_service.py` | Session delete: also delete from S3 | In `delete_session()`: remove `session_data/{id}/` prefix from S3 |
| `main.py` | On startup: optionally restore sessions from S3 | In `lifespan()`: if `STORAGE_BACKEND=s3`, list and restore recent sessions |
| `main.py` | Add file-based logging + S3 upload | Add `FileHandler` + async S3 upload for daily logs |

### Session Persistence Strategy
- **In-memory first**: All operations use in-memory dicts (fast, no latency change)
- **Write-behind to S3**: After session update, async upload to S3 (non-blocking)
- **Startup restore**: On app start, optionally download recent sessions from S3 → rebuild in-memory state
- **TTL enforcement**: Sessions older than `SESSION_TTL_HOURS` not restored from S3

### FAISS Index Serialization
```python
# Save to S3 (after index built in memory):
import tempfile, faiss
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    faiss.write_index(session_index.index, f.name)
    upload_file(f.name, f"{prefix}/session_data/{session_id}/faiss_index.bin")

# Load from S3 (on session restore):
with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    download_file(f"{prefix}/session_data/{session_id}/faiss_index.bin", f.name)
    index = faiss.read_index(f.name)
```

### New .env Variables
```
STORAGE_BACKEND=s3
S3_BUCKET_NAME=vcs-ai-agents-data
S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<from user>
AWS_SECRET_ACCESS_KEY=<from user>
S3_AGENT_PREFIX=document-qa-agent
S3_RESTORE_SESSIONS_ON_STARTUP=true         # set false to skip S3 restore
S3_MAX_RESTORE_SESSIONS=50                   # limit how many sessions to restore
```

### New Dependency
```
boto3>=1.34.0   # add to requirements.txt
```

### Rollback
Set `STORAGE_BACKEND=local` in `.env` → no S3 writes, fully in-memory (original behavior).

### Activation Instructions
When user says **"use the new S3 code"**:
1. Set `STORAGE_BACKEND=s3` in `.env`
2. Restart: `sudo systemctl restart docqa-agent`
3. Upload a file, ask a question → verify session saved in S3
4. Restart agent → verify session restored from S3
