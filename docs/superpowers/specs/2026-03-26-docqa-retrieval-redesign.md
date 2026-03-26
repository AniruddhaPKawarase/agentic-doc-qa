# DocQA Agent — Retrieval Pipeline Redesign

## User Story

**As a construction professional** uploading PDFs, images, and documents to the DocQA agent,
**I want** the agent to answer questions about the document I JUST uploaded, not mix up answers from previously uploaded files,
**so that** I can have a focused, accurate conversation about each document — just like ChatGPT or Claude.

---

## Problem Statement

When multiple files exist in a session, the agent cannot distinguish which file the user is asking about. FAISS retrieval returns top-k chunks from ALL files ranked by cosine similarity. When the user asks "Give me information about this document" after uploading a new PDF, the answer comes from whichever file has the highest-similarity chunks — often a DIFFERENT file.

**Example from production:**
- User uploads `23 37 13-1 Air Diffusers and Registers Product Data.pdf`
- Asks: "Give me information about this document"
- Agent answers about `A101-FLOOR-PLAN.pdf` (similarity 0.434) instead of the just-uploaded file (similarity 0.301)

---

## Confirmed Design Decisions

| # | Decision | Choice | Detail |
|---|----------|--------|--------|
| Q1 | File scoping on upload+query | **(A) Strict scoping** | When files attached → answer ONLY from those files. If user mentions a past file by name → intelligent context switching |
| Q2 | Follow-up (no file) | **(A) Global search** | Search all session files when no file is attached |
| Q3 | Index structure | **(B) Single index + metadata filter** | Keep single FAISS per session, filter by file_name post-search |
| Q4 | Duplicate file handling | **(B) Scope to existing copy** | Duplicate files still get scoped answers from their already-indexed chunks |
| Q5 | LLM file awareness | **(C) Pre-filter + prompt injection** | Filter chunks to current file AND inject filename into system prompt |

---

## Architecture Design

### Core Concept: File-Aware Retrieval with Context Switching

```
User uploads file + asks question
         │
         ▼
┌─────────────────────────────────┐
│ 1. FILE SCOPE RESOLVER          │
│    - Files attached? → scope to │
│      those files only           │
│    - No files + mentions name?  │
│      → detect file reference    │
│    - No files + no mention?     │
│      → search all (global)      │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│ 2. FAISS SEARCH (unchanged)     │
│    - Embed query                │
│    - Search session index       │
│    - Return top-k * 2 (overfetch)│
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│ 3. POST-RETRIEVAL FILE FILTER   │
│    - If scoped: keep only chunks│
│      from target file(s)        │
│    - If global: keep all chunks │
│    - Re-rank by similarity      │
│    - Take top-k                 │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│ 4. FILE-AWARE PROMPT BUILDER    │
│    - Inject current file name   │
│    - Add conversation awareness │
│    - Build context from filtered│
│      chunks only                │
└────────────┬────────────────────┘
             ▼
┌─────────────────────────────────┐
│ 5. LLM GENERATION               │
│    - System prompt with file    │
│      context + scoping rules    │
│    - Conversation history       │
│    - Generate answer            │
└─────────────────────────────────┘
```

### Component 1: File Scope Resolver

**Location:** New function in `services/retrieval_service.py`

**Logic:**
```python
def resolve_file_scope(
    uploaded_file_names: List[str],  # files in THIS request
    query: str,                       # user's question
    session_file_names: List[str],    # all files in session
) -> Tuple[List[str], str]:
    """
    Returns:
        - target_files: list of file names to scope retrieval to (empty = global)
        - scope_mode: "current_upload" | "referenced_file" | "global"
    """
    # Case 1: Files were uploaded in this request → strict scope
    if uploaded_file_names:
        return uploaded_file_names, "current_upload"

    # Case 2: User mentions a file name in their query
    for fname in session_file_names:
        # Match by exact name or partial (without extension)
        name_stem = fname.rsplit(".", 1)[0].lower()
        if name_stem in query.lower() or fname.lower() in query.lower():
            return [fname], "referenced_file"

    # Case 3: No files, no reference → search all
    return [], "global"
```

### Component 2: Post-Retrieval File Filter

**Location:** Modified `services/retrieval_service.py` → `retrieve()` method

**Logic:**
```python
# After FAISS search returns top-k*2 results:
if target_files:
    # Keep only chunks from target files
    filtered = [r for r in results if r.chunk.file_name in target_files]
    # If too few results, fall back to global (safety net)
    if len(filtered) < 2:
        filtered = results  # fall back
    results = filtered[:top_k]
else:
    results = results[:top_k]
```

### Component 3: File-Aware Prompt Builder

**Location:** Modified `services/generation_service.py`

**New system prompt additions:**
```
CURRENTLY UPLOADED FILE(S): {file_names}
SCOPE: {scope_mode_description}

FILE AWARENESS RULES:
1. When the user uploads a file with their question, answer STRICTLY from that file.
2. When the user mentions a specific file name, focus your answer on that file.
3. When the user asks a general follow-up (no file reference), use all available documents.
4. Always clearly state which file your answer comes from.
5. If asked "about this document" — it means the most recently uploaded file.
```

### Component 4: Duplicate File Scoping

**Location:** Modified `routers/converse.py` → `_upload_phase()`

When a file is `duplicate_skipped`, still pass its name as a "target file" for scoping:
```python
# After dedup check
if is_duplicate:
    file_results.append(FileInfo(status="duplicate_skipped", ...))
    # BUT still scope the query to this file
    scoped_file_names.append(file_name)
```

---

## Files to Modify

| File | Change |
|------|--------|
| `services/retrieval_service.py` | Add `resolve_file_scope()`, modify `retrieve()` to accept `target_files` param and filter results |
| `services/generation_service.py` | Modify `_build_messages()` to accept `current_files` param, inject file context into system prompt |
| `routers/converse.py` | Pass uploaded file names through pipeline, handle duplicate scoping |
| `routers/chat.py` | Pass empty file list (global search behavior preserved) |

---

## Behavioral Matrix

| Scenario | Files Attached | Query | Scope | Behavior |
|----------|---------------|-------|-------|----------|
| Upload + ask | Yes | "Give me info about this document" | Current file only | Answer from uploaded file |
| Upload + ask specific | Yes | "What are the HVAC specs?" | Current file only | Search uploaded file for HVAC |
| Follow-up, no file | No | "Tell me more" | Global | Search all session files |
| Reference past file | No | "What about the floor plan?" | Referenced file | Detect "floor plan" → match file |
| Generic question | No | "What materials are specified?" | Global | Search all session files |
| Duplicate upload + ask | Yes (dup) | "Summarize this" | Duplicate file chunks | Answer from existing indexed copy |

---

## Testing Scenarios

After implementation, ALL of these must pass:

| # | Action | Expected |
|---|--------|----------|
| 1 | Upload `Terminal Units.pdf` + ask "Give me info about this document" | Answer from Terminal Units ONLY |
| 2 | Upload `Air Diffusers.pdf` + ask "Give me info about this document" | Answer from Air Diffusers ONLY (not Terminal Units) |
| 3 | No file + ask "What about the Terminal Units?" | Detect file reference → answer from Terminal Units |
| 4 | No file + ask "Compare the two documents" | Global search → answer from both |
| 5 | Re-upload `Air Diffusers.pdf` (duplicate) + ask "Summarize" | Duplicate skipped but answer scoped to Air Diffusers |
| 6 | Upload `A101-FLOOR-PLAN.pdf` + ask "What rooms are shown?" | Answer from floor plan ONLY |
