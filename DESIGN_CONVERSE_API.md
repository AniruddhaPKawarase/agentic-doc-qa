# Unified Converse API — Design Document

**Date:** 2026-03-18
**Status:** Approved (pending CLAUDE.md confirmation)
**Author:** Claude Code (based on user requirements)

---

## 1. User Story

> As a UI developer integrating the Document Q&A Agent, I need a **single API endpoint**
> that accepts file(s) and a query together, so the frontend doesn't have to orchestrate
> two separate calls. The backend must maintain a `session_id` via cookies so follow-up
> conversations on the same uploaded document(s) work without the UI explicitly managing
> session state. The agent must generate follow-up questions automatically.

---

## 2. Requirements Clarification Q&A

| # | Question | Answer |
|---|----------|--------|
| Q1 | Should `file` be optional (to allow follow-ups with no new file)? | **No — file is mandatory** in the unified endpoint on every request |
| Q2 | Can user add more files to an existing session mid-conversation? | **Yes** — pass existing `session_id` (cookie or form field) + new files + query |
| Q3 | Response format preference? | **Both** — `POST /api/converse` (blocking) + `POST /api/converse/stream` (SSE) |
| Q4 | Keep existing `/api/upload` and `/api/chat` endpoints? | **Yes** — keep for backward compatibility |
| Q5 | If file uploads but LLM query fails, what happens to session? | **Preserve session** — indexed files survive, user can retry query |
| Q6 | How does the UI manage session identity? | **Cookie-based** — backend sets `docqa_session_id` HttpOnly cookie |

---

## 3. Architecture Overview

```
UI (Browser)
    │
    │  POST /api/converse   (multipart/form-data)
    │  files[]  ← required, 1–10 files
    │  query    ← required, the question
    │  session_id ← optional form override
    │  Cookie: docqa_session_id=<id>  ← set by backend on prior response
    │
    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      ConverseRouter (/api/converse)                  │
│                                                                      │
│  Step 1: Resolve Session                                             │
│    form.session_id  →  cookie docqa_session_id  →  new session       │
│    (form field wins over cookie if both present)                     │
│                                                                      │
│  Step 2: Upload Phase  (per file, independent error boundaries)      │
│    ├── Validate extension + size                                     │
│    ├── SHA-256 hash → check session.file_hashes (dedup)              │
│    ├── FileProcessor → extract text + chunk                          │
│    ├── EmbeddingService → embed chunks (batched)                     │
│    └── IndexService → upsert into session's FAISS index              │
│                                                                      │
│  Step 3: Abort gate                                                  │
│    If session has 0 indexed chunks total → HTTP 400                  │
│    (all files failed AND no prior chunks in session)                 │
│                                                                      │
│  Step 4: Query Phase  (existing pipeline, unchanged)                 │
│    ├── Check L1/L2 cache                                             │
│    ├── FAISS retrieve (top-k=8, threshold=0.30)                      │
│    ├── Build LLM prompt (system + history + context + query)         │
│    ├── LLM call (gpt-4o-mini, blocking or stream)                    │
│    ├── Hallucination guard (token-overlap ≥ 0.65)                    │
│    ├── Generate follow-up questions                                   │
│    └── Store turn in session history                                 │
│                                                                      │
│  Step 5: Set-Cookie header on response                               │
│    docqa_session_id=<id>; HttpOnly; SameSite=Lax; Max-Age=86400      │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. New Endpoints

### 4.1 Blocking — `POST /api/converse`

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | `List[UploadFile]` | **Yes** | 1–10 files (PDF/Word/Excel/TXT/CSV/JSON) |
| `query` | `str` (Form) | **Yes** | The user's question (1–5000 chars) |
| `session_id` | `str` (Form) | No | Explicit session override (wins over cookie) |
| `Cookie: docqa_session_id` | str | No | Set by backend on prior response |

**Response:** `application/json` — `ConverseResponse`

```json
{
  "session_id": "a3f8c2d1b0e94f5a",
  "files_processed": [
    {
      "file_name": "contract.pdf",
      "file_type": ".pdf",
      "size_bytes": 204800,
      "chunk_count": 42,
      "status": "processed",
      "error": null
    }
  ],
  "new_chunks_added": 42,
  "total_session_files": 3,
  "total_session_chunks": 118,
  "answer": "The contract expires on December 31, 2026...",
  "sources": [...],
  "follow_up_questions": [
    "What are the renewal conditions?",
    "Who are the parties involved?",
    "What penalties apply for early termination?"
  ],
  "groundedness_score": 0.87,
  "needs_clarification": false,
  "clarification_questions": [],
  "token_usage": {
    "embedding_tokens": 1320,
    "prompt_tokens": 4200,
    "completion_tokens": 312,
    "total_tokens": 5832,
    "estimated_cost_usd": 0.0012
  },
  "pipeline_ms": {
    "retrieval_ms": 14.2,
    "llm_ms": 1840.5,
    "guard_ms": 3.1,
    "total_ms": 2890.0
  },
  "cached": false
}
```

**Response Headers:**
```
Set-Cookie: docqa_session_id=a3f8c2d1b0e94f5a; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400
```

---

### 4.2 Streaming — `POST /api/converse/stream`

Same request format as `/api/converse`.
Response: `text/event-stream` (SSE).

**SSE Event sequence:**
```
data: {"type": "upload_done", "files_processed": [...], "new_chunks_added": 42, "total_session_chunks": 118}

data: {"type": "sources", "sources": [...]}

data: {"type": "chunk", "content": "The contract expires on..."}
data: {"type": "chunk", "content": " December 31, 2026..."}
... (tokens stream as they arrive)

data: {"type": "done", "session_id": "...", "follow_up_questions": [...], "groundedness_score": 0.87, "needs_clarification": false, "pipeline_ms": {...}}
```

On hallucination rollback, a final `chunk` event is appended:
```
data: {"type": "chunk", "content": "\n\n---\n\nI'm not confident enough..."}
data: {"type": "done", "needs_clarification": true, "clarification_questions": [...]}
```

---

## 5. Session Cookie Design

| Attribute | Value | Reason |
|-----------|-------|--------|
| Cookie name | `docqa_session_id` | Prefixed to avoid collision with other agents |
| `HttpOnly` | `true` | XSS protection — JS cannot read session ID |
| `SameSite` | `Lax` | CSRF protection while allowing normal navigation |
| `Path` | `/` | Cookie sent to all paths under the domain |
| `Max-Age` | `86400` (24h) | Matches `SESSION_TTL_HOURS` config |
| `Secure` | `false` (dev) / `true` (prod) | Toggle via env `COOKIE_SECURE=true` in prod |

**Priority resolution:**
```
form field session_id  >  Cookie docqa_session_id  >  new session created
```

This allows the UI to override the cookie when needed (e.g., "start new conversation").

**Cookie refresh:** Set on **every** response — refreshes the 24h TTL as long as user is active.

---

## 6. File Deduplication (SHA-256)

**Problem:** Without dedup, uploading the same file twice in the same session causes:
- Duplicate embedding API calls (cost waste)
- Duplicate FAISS vectors (inflates retrieval results)
- Confusing chunk counts in responses

**Solution:** SHA-256 content hash stored per session.

```
File bytes → SHA-256(32-byte hex) → check session.file_hashes set → O(1)
If already present: status = "duplicate_skipped", chunk_count = 0
If new: process + store hash → O(file_size) once
```

**Session change:** Add `file_hashes: Set[str]` to `Session` dataclass.
**New methods on SessionService:**
- `has_file_hash(session_id, sha256_hex) → bool`
- `add_file_hash(session_id, sha256_hex) → None`

**FileInfo status values** (extended):
- `"processed"` — successfully embedded and indexed
- `"failed"` — processing error (see `error` field)
- `"unsupported"` — file extension not allowed
- `"duplicate_skipped"` ← **new** — same file already indexed in this session

---

## 7. Error Handling Strategy

| Scenario | Behavior |
|----------|----------|
| File extension not allowed | `status="unsupported"`, continue processing other files |
| File too large | `status="failed"`, continue processing other files |
| File parsing fails | `status="failed"`, continue processing other files |
| File is a duplicate | `status="duplicate_skipped"`, continue processing other files |
| Embedding API fails | `status="failed"`, session NOT rolled back |
| **All files fail + session has 0 chunks** | HTTP 400: "No documents could be indexed. Please check your files." |
| **All files fail + session has prior chunks** | Proceed to query phase using prior indexed content |
| LLM call fails (timeout/5xx) | HTTP 502, session preserved (files already indexed), `session_id` in error body + cookie still set |
| Session not found (bad cookie) | Create new session, process files, continue |

---

## 8. Files to Create / Modify

### New Files
| File | Purpose |
|------|---------|
| `routers/converse.py` | `POST /api/converse` + `POST /api/converse/stream` |

### Modified Files
| File | Change |
|------|--------|
| `models/schemas.py` | Add `ConverseResponse` schema with upload + query fields combined |
| `services/session_service.py` | Add `file_hashes: Set[str]` to `Session`; add `has_file_hash()`, `add_file_hash()` |
| `main.py` | Import + mount `converse.router`; add `COOKIE_SECURE` env read |
| `config.py` | Add `cookie_secure: bool = False` setting |

### Unchanged Files (existing endpoints untouched)
`routers/upload.py`, `routers/chat.py`, `routers/sessions.py`, all `services/*.py` (except session_service.py)

---

## 9. DSA & System Design Rationale

### Why SHA-256 for dedup?
- Collision-resistant for content identity (far better than filename comparison)
- O(file_size) one-time hashing, O(1) set lookup thereafter
- 32-byte hex fits trivially in session memory

### Why file-first, then query?
- **Atomicity boundary**: upload phase is committed before LLM phase begins
- LLM failure cannot corrupt indexed data
- Session is always in a consistent state (files indexed = accessible)
- Enables retry: user can re-send the same query (cache will hit if identical)

### Why cookie over localStorage?
- `HttpOnly` cookie is inaccessible to JS → XSS cannot steal session
- Browser sends cookie automatically → frontend needs zero session management code
- Cookie TTL refresh on every response = session stays alive during active use

### Why keep existing endpoints?
- Zero breaking changes for any existing integrations
- `/api/upload` + `/api/chat` remain the "power user" flow (upload once, query many)
- `/api/converse` is the "UI-friendly" unified flow (file + query together)

### Cookie vs Token auth tradeoff
- Cookies are browser-native, auto-sent, auto-expired
- No token refresh logic needed in the frontend
- Appropriate since sessions are short-lived (24h) and contain no privileged data

---

## 10. Implementation Phases

| Phase | Scope | Files |
|-------|-------|-------|
| 1 | Session dedup + cookie support | `services/session_service.py`, `config.py` |
| 2 | ConverseResponse schema | `models/schemas.py` |
| 3 | Converse router (blocking) | `routers/converse.py` |
| 4 | Converse router (streaming) | `routers/converse.py` (add `/stream`) |
| 5 | Mount router + wire up | `main.py` |

---

## 11. Complete Request / Response Flow (Sequence)

```
Browser                      ConverseRouter            Services
   │                               │                      │
   │─── POST /api/converse ────────►│                      │
   │    (multipart: files, query,   │                      │
   │     Cookie: docqa_session_id)  │                      │
   │                               │                      │
   │                               │─ resolve session ────►│ SessionService.get_or_create()
   │                               │◄─ session ────────────│
   │                               │                      │
   │                               │── For each file ─────►│
   │                               │   SHA-256 hash check  │ SessionService.has_file_hash()
   │                               │   Process+embed+index │ FileProcessor, EmbeddingService,
   │                               │                       │ IndexService
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ abort gate ─────────►│ IndexService.get_chunk_count()
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ retrieve ───────────►│ RetrievalService.retrieve()
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ generate ───────────►│ GenerationService.generate()
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ guard ──────────────►│ HallucinationGuard.check()
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ follow-ups ─────────►│ GenerationService.generate_followups()
   │                               │◄──────────────────────│
   │                               │                      │
   │                               │─ store turn ─────────►│ SessionService.add_turn()
   │                               │◄──────────────────────│
   │                               │                      │
   │◄── ConverseResponse ──────────│                      │
   │    Set-Cookie: docqa_session_id=...                  │
```