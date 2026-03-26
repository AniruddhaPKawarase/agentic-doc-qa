"""
Converse Router — Unified upload + Q&A endpoints (UI-friendly).
──────────────────────────────────────────────────────────────────────────────
POST /api/converse        — Blocking: upload file(s) + query in one request
POST /api/converse/stream — SSE streaming variant

Session is managed via an HttpOnly cookie (docqa_session_id).
Priority: form field session_id > cookie > new session created.

File deduplication: SHA-256 content hash per session prevents re-embedding
the same file twice (returns status="duplicate_skipped").

Upload phase is committed before LLM phase — session is preserved even if
the LLM call fails, so the user can retry the query without re-uploading.
"""

import json
import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Cookie, File, Form, HTTPException, Request, Response, UploadFile
from sse_starlette.sse import EventSourceResponse

from models.schemas import (
    ConverseResponse,
    FileInfo,
    PipelineTimings,
    SourceChunk,
    TokenUsage,
)
from services.token_tracker import PipelineTokenLog

logger = logging.getLogger("docqa.router.converse")

router = APIRouter(prefix="/api", tags=["converse"])

_COOKIE_NAME = "docqa_session_id"
_COOKIE_MAX_AGE = 86400  # 24 h — matches SESSION_TTL_HOURS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_session_cookie(response: Response, session_id: str, secure: bool) -> None:
    """Set (or refresh) the session cookie on any response."""
    response.set_cookie(
        key=_COOKIE_NAME,
        value=session_id,
        max_age=_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=secure,
        path="/",
    )


async def _upload_phase(
    request: Request,
    files: List[UploadFile],
    session_id: str,
) -> tuple[List[FileInfo], int]:
    """
    Process each uploaded file: validate → dedup → extract → embed → index.
    Returns (file_results, new_chunks_added).
    Per-file errors are captured in FileInfo.status; processing continues for
    remaining files regardless of individual failures.
    """
    settings = request.app.state.settings
    file_processor = request.app.state.file_processor
    embedding_service = request.app.state.embedding_service
    index_service = request.app.state.index_service
    session_service = request.app.state.session_service

    file_results: List[FileInfo] = []
    new_chunks = 0

    for upload_file in files:
        file_name = upload_file.filename or "unknown"
        ext = ("." + file_name.rsplit(".", 1)[-1].lower()) if "." in file_name else ""

        # ── Validate extension ────────────────────────────────────────────
        if ext not in settings.allowed_extensions_list:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=0,
                chunk_count=0,
                status="unsupported",
                error=f"Unsupported file type: {ext}. Allowed: {settings.allowed_extensions}",
            ))
            continue

        # ── Read content ──────────────────────────────────────────────────
        content = await upload_file.read()

        # ── Validate size ─────────────────────────────────────────────────
        if len(content) > settings.max_file_size_bytes:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status="failed",
                error=f"File too large: {len(content) / (1024 * 1024):.1f}MB (max {settings.max_file_size_mb}MB)",
            ))
            continue

        # ── SHA-256 deduplication ─────────────────────────────────────────
        file_hash = session_service.compute_file_hash(content)
        if session_service.has_file_hash(session_id, file_hash):
            logger.info(f"Duplicate file skipped: {file_name} (session {session_id})")
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status="duplicate_skipped",
                error=None,
            ))
            continue

        # ── Extract text + chunk ──────────────────────────────────────────
        processed = file_processor.process(file_name, content)
        session_service.add_file(session_id, processed)

        if processed.status != "processed" or not processed.chunks:
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status=processed.status,
                error=processed.error,
            ))
            continue

        # ── Embed + index ─────────────────────────────────────────────────
        try:
            chunk_texts = [c.text for c in processed.chunks]
            vectors = await embedding_service.embed_texts(chunk_texts)
            index_service.create_or_update(
                session_id=session_id,
                chunks=processed.chunks,
                vectors=vectors,
            )
            session_service.add_file_hash(session_id, file_hash)
            new_chunks += len(processed.chunks)

            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=len(processed.chunks),
                status="processed",
            ))

        except Exception as e:
            logger.error(f"Failed to embed/index {file_name}: {e}")
            file_results.append(FileInfo(
                file_name=file_name,
                file_type=ext,
                size_bytes=len(content),
                chunk_count=0,
                status="failed",
                error=f"Embedding failed: {str(e)}",
            ))

    return file_results, new_chunks


async def _query_phase(
    request: Request,
    session_id: str,
    query: str,
    uploaded_file_names: Optional[List[str]] = None,
) -> dict:
    """
    Run the full RAG pipeline for a query against the session's FAISS index.
    Returns a plain dict with all response fields.

    Args:
        uploaded_file_names: Files uploaded in THIS request. When provided,
            retrieval is scoped to those files only (file-aware retrieval v2).
    """
    session_service = request.app.state.session_service
    retrieval_service = request.app.state.retrieval_service
    generation_service = request.app.state.generation_service
    hallucination_guard = request.app.state.hallucination_guard
    cache_service = request.app.state.cache_service
    token_tracker = request.app.state.token_tracker
    settings = request.app.state.settings

    pipeline_start = time.perf_counter()
    log = PipelineTokenLog()

    # ── File scope resolution ─────────────────────────────────────────────
    from services.retrieval_service import resolve_file_scope

    # Get all file names in this session for reference matching
    session = session_service.get_session(session_id)
    session_file_names = [f.file_name for f in (session.files if session else [])]

    target_files, scope_mode = resolve_file_scope(
        uploaded_file_names=uploaded_file_names or [],
        query=query,
        session_file_names=session_file_names,
    )

    # ── Cache check (skip cache when scoped to specific files) ────────────
    if scope_mode == "global":
        cached = await cache_service.get(session_id, query)
        if cached:
            return {**cached, "cached": True}

    # ── Retrieve (with file scoping) ──────────────────────────────────────
    retrieval = await retrieval_service.retrieve(
        session_id, query,
        target_files=target_files,
        scope_mode=scope_mode,
    )
    log.record_step(
        "retrieval",
        embedding_tokens=retrieval.embedding_tokens,
        elapsed_ms=retrieval.retrieval_ms,
    )

    if not retrieval.has_results:
        return {
            "answer": (
                "I can only answer questions based on your uploaded documents. "
                "Your question doesn't appear to be covered in the provided documents. "
                "Please rephrase or upload relevant documents."
            ),
            "sources": [],
            "follow_up_questions": [
                "Could you rephrase your question to relate to the uploaded documents?",
                "Would you like to upload additional documents that cover this topic?",
                "What specific section of the documents are you interested in?",
            ],
            "groundedness_score": 0.0,
            "needs_clarification": True,
            "clarification_questions": [],
            "token_usage": {"embedding_tokens": retrieval.embedding_tokens},
            "pipeline_ms": {
                "retrieval_ms": retrieval.retrieval_ms,
                "total_ms": retrieval.retrieval_ms,
            },
            "cached": False,
        }

    context = retrieval.build_context(settings.max_context_tokens)
    sources = [
        {
            "file_name": chunk.file_name,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "sheet_name": chunk.sheet_name,
            "score": round(score, 3),
            "text_preview": chunk.text[:200],
        }
        for chunk, score in retrieval.chunks
    ]

    # ── Generate (file-aware) ────────────────────────────────────────────
    history = session_service.build_history_messages(session_id)
    result = await generation_service.generate(
        query, context, history,
        current_files=target_files or None,
        scope_mode=scope_mode,
    )
    log.record_step(
        "generation",
        prompt_tokens=result["prompt_tokens"],
        completion_tokens=result["completion_tokens"],
        elapsed_ms=result["llm_ms"],
    )

    answer = result["answer"]

    # ── Hallucination guard ───────────────────────────────────────────────
    guard_start = time.perf_counter()
    guard_result = hallucination_guard.check(answer, context)
    guard_ms = (time.perf_counter() - guard_start) * 1000

    needs_clarification = not guard_result["passed"]
    clarification_questions = []

    if needs_clarification:
        clarification_questions = hallucination_guard.generate_clarification_questions(
            context, count=settings.min_followup_questions
        )
        answer = (
            "I'm not confident enough to answer this accurately based on the documents. "
            "Let me ask some clarifying questions to help you better:"
        )

    # ── Follow-up questions ───────────────────────────────────────────────
    follow_ups = await generation_service.generate_followups(
        context, answer, settings.min_followup_questions
    )

    # ── Token accounting ──────────────────────────────────────────────────
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    cost = token_tracker.estimate_cost(log)

    session_service.add_turn(session_id, "user", query)
    session_service.add_turn(
        session_id, "assistant", answer,
        groundedness=guard_result["groundedness"],
    )
    session_service.add_tokens(session_id, log.total_tokens)

    response_data = {
        "answer": answer,
        "sources": sources,
        "follow_up_questions": follow_ups,
        "groundedness_score": guard_result["groundedness"],
        "needs_clarification": needs_clarification,
        "clarification_questions": clarification_questions,
        "token_usage": {
            "embedding_tokens": log.total_embedding_tokens,
            "prompt_tokens": log.total_prompt_tokens,
            "completion_tokens": log.total_completion_tokens,
            "total_tokens": log.total_tokens,
            "estimated_cost_usd": cost,
        },
        "pipeline_ms": {
            "retrieval_ms": round(retrieval.retrieval_ms, 1),
            "llm_ms": round(result["llm_ms"], 1),
            "guard_ms": round(guard_ms, 1),
            "total_ms": round(total_ms, 1),
        },
        "cached": False,
    }

    if not needs_clarification:
        await cache_service.set(session_id, query, response_data)

    return response_data


# ── Blocking endpoint ─────────────────────────────────────────────────────────

@router.post("/converse", response_model=ConverseResponse)
async def converse(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(..., description="One or more files to upload and query against"),
    query: str = Form(..., min_length=1, max_length=5000, description="Question to ask about the uploaded files"),
    session_id: Optional[str] = Form(None, description="Existing session ID (overrides cookie)"),
    cookie_session_id: Optional[str] = Cookie(None, alias=_COOKIE_NAME),
):
    """
    Unified endpoint: upload file(s) and ask a question in a single request.

    Session flow:
    - First call (no cookie, no session_id): creates a new session.
    - Follow-up (cookie set): resolves the existing session and adds new files.
    - Explicit override: pass session_id in the form body to target a specific session.

    Files are mandatory. To add more documents mid-conversation, pass the existing
    session_id (or rely on the cookie) together with the new files and your question.
    """
    settings = request.app.state.settings
    session_service = request.app.state.session_service
    index_service = request.app.state.index_service

    # Validate file count
    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_upload} files per upload",
        )

    # Resolve session: form field wins over cookie
    effective_id = session_id or cookie_session_id
    session = session_service.get_or_create(effective_id)
    sid = session.session_id

    # ── Phase 1: Upload ───────────────────────────────────────────────────
    file_results, new_chunks = await _upload_phase(request, files, sid)

    total_chunks = index_service.get_chunk_count(sid)

    # Abort if nothing is indexed at all (all files failed + no prior content)
    if total_chunks == 0:
        _set_session_cookie(response, sid, settings.cookie_secure)
        raise HTTPException(
            status_code=400,
            detail={
                "message": "No documents could be indexed. Please check your files and try again.",
                "session_id": sid,
                "files": [f.model_dump() for f in file_results],
            },
        )

    # Collect uploaded file names (processed + duplicate_skipped) for scoping
    uploaded_file_names = [
        fr.file_name for fr in file_results
        if fr.status in ("processed", "duplicate_skipped")
    ]

    # ── Phase 2: Query (file-aware) ───────────────────────────────────────
    data = await _query_phase(request, sid, query, uploaded_file_names=uploaded_file_names)

    # Refresh session cookie on every response
    _set_session_cookie(response, sid, settings.cookie_secure)

    return ConverseResponse(
        session_id=sid,
        files_processed=file_results,
        new_chunks_added=new_chunks,
        total_session_files=session.file_count,
        total_session_chunks=total_chunks,
        answer=data["answer"],
        sources=[SourceChunk(**s) for s in data.get("sources", [])],
        follow_up_questions=data.get("follow_up_questions", []),
        groundedness_score=data.get("groundedness_score", 0.0),
        needs_clarification=data.get("needs_clarification", False),
        clarification_questions=data.get("clarification_questions", []),
        token_usage=TokenUsage(**data.get("token_usage", {})),
        pipeline_ms=PipelineTimings(**data.get("pipeline_ms", {})),
        cached=data.get("cached", False),
    )


# ── SSE Streaming endpoint ────────────────────────────────────────────────────

@router.post("/converse/stream")
async def converse_stream(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(..., description="One or more files to upload and query against"),
    query: str = Form(..., min_length=1, max_length=5000, description="Question to ask about the uploaded files"),
    session_id: Optional[str] = Form(None, description="Existing session ID (overrides cookie)"),
    cookie_session_id: Optional[str] = Cookie(None, alias=_COOKIE_NAME),
):
    """
    SSE streaming variant of /api/converse.

    Event sequence:
      1. upload_done  — file processing results + chunk counts
      2. sources      — retrieved document chunks (sent before LLM starts)
      3. chunk        — one or more LLM token chunks (streaming)
      4. done         — final metadata (follow_ups, groundedness, pipeline_ms, session_id)

    The docqa_session_id cookie is set on the SSE response headers.
    """
    settings = request.app.state.settings
    session_service = request.app.state.session_service
    retrieval_service = request.app.state.retrieval_service
    generation_service = request.app.state.generation_service
    hallucination_guard = request.app.state.hallucination_guard
    cache_service = request.app.state.cache_service
    token_tracker = request.app.state.token_tracker
    index_service = request.app.state.index_service

    if len(files) > settings.max_files_per_upload:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {settings.max_files_per_upload} files per upload",
        )

    effective_id = session_id or cookie_session_id
    session = session_service.get_or_create(effective_id)
    sid = session.session_id

    # Set cookie on SSE response headers before streaming begins
    _set_session_cookie(response, sid, settings.cookie_secure)

    async def event_generator():
        pipeline_start = time.perf_counter()
        log = PipelineTokenLog()

        # ── Phase 1: Upload ───────────────────────────────────────────────
        file_results, new_chunks = await _upload_phase(request, files, sid)
        total_chunks = index_service.get_chunk_count(sid)

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "upload_done",
                "session_id": sid,
                "files_processed": [f.model_dump() for f in file_results],
                "new_chunks_added": new_chunks,
                "total_session_files": session.file_count,
                "total_session_chunks": total_chunks,
            }),
        }

        if total_chunks == 0:
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "message": "No documents could be indexed. Please check your files and try again.",
                }),
            }
            return

        # ── Cache check ───────────────────────────────────────────────────
        cached = await cache_service.get(sid, query)
        if cached:
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": cached["answer"]}),
            }
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "done",
                    "session_id": sid,
                    "sources": cached.get("sources", []),
                    "follow_up_questions": cached.get("follow_up_questions", []),
                    "groundedness_score": cached.get("groundedness_score", 0),
                    "needs_clarification": cached.get("needs_clarification", False),
                    "token_usage": cached.get("token_usage", {}),
                    "pipeline_ms": cached.get("pipeline_ms", {}),
                    "cached": True,
                }),
            }
            return

        # ── Phase 2: Retrieve ─────────────────────────────────────────────
        retrieval = await retrieval_service.retrieve(sid, query)
        log.record_step(
            "retrieval",
            embedding_tokens=retrieval.embedding_tokens,
            elapsed_ms=retrieval.retrieval_ms,
        )

        if not retrieval.has_results:
            oor_msg = (
                "I can only answer questions based on your uploaded documents. "
                "Your question doesn't appear to be covered."
            )
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": oor_msg}),
            }
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "done",
                    "session_id": sid,
                    "sources": [],
                    "follow_up_questions": [
                        "Could you rephrase your question?",
                        "Would you like to upload additional documents?",
                        "What section of the documents interests you?",
                    ],
                    "groundedness_score": 0.0,
                    "needs_clarification": True,
                    "cached": False,
                }),
            }
            return

        context = retrieval.build_context(settings.max_context_tokens)
        sources = [
            {
                "file_name": c.file_name,
                "chunk_index": c.chunk_index,
                "page_number": c.page_number,
                "sheet_name": c.sheet_name,
                "score": round(s, 3),
                "text_preview": c.text[:200],
            }
            for c, s in retrieval.chunks
        ]

        # Send sources before LLM starts so the UI can display them immediately
        yield {
            "event": "message",
            "data": json.dumps({"type": "sources", "sources": sources}),
        }

        # ── Stream LLM response ───────────────────────────────────────────
        history = session_service.build_history_messages(sid)
        full_answer = ""
        llm_start = time.perf_counter()

        async for chunk_text in generation_service.generate_stream(query, context, history):
            full_answer += chunk_text
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": chunk_text}),
            }

        llm_ms = (time.perf_counter() - llm_start) * 1000

        # ── Hallucination guard ───────────────────────────────────────────
        guard_result = hallucination_guard.check(full_answer, context)

        if not guard_result["passed"]:
            clarifications = hallucination_guard.generate_clarification_questions(
                context, settings.min_followup_questions
            )
            rollback_msg = (
                "\n\n---\n\n"
                "I'm not confident enough in this answer. "
                "Let me ask some clarifying questions:\n"
                + "\n".join(f"- {q}" for q in clarifications)
            )
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": rollback_msg}),
            }
            full_answer += rollback_msg

        # ── Follow-up questions ───────────────────────────────────────────
        follow_ups = await generation_service.generate_followups(
            context, full_answer, settings.min_followup_questions
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        session_service.add_turn(sid, "user", query)
        session_service.add_turn(
            sid, "assistant", full_answer,
            groundedness=guard_result["groundedness"],
        )
        session_service.add_tokens(sid, log.total_tokens)

        yield {
            "event": "message",
            "data": json.dumps({
                "type": "done",
                "session_id": sid,
                "follow_up_questions": follow_ups,
                "groundedness_score": guard_result["groundedness"],
                "needs_clarification": not guard_result["passed"],
                "clarification_questions": (
                    hallucination_guard.generate_clarification_questions(
                        context, settings.min_followup_questions
                    ) if not guard_result["passed"] else []
                ),
                "pipeline_ms": {
                    "retrieval_ms": round(retrieval.retrieval_ms, 1),
                    "llm_ms": round(llm_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
                "cached": False,
            }),
        }

    return EventSourceResponse(event_generator())
