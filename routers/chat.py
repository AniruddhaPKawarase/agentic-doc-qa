"""
Chat Router — Blocking and streaming Q&A endpoints.
──────────────────────────────────────────────────────────────────────────────
POST /api/chat        — Blocking answer
POST /api/chat/stream — SSE streaming answer
"""

import json
import logging
import time

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from models.schemas import (
    ChatRequest,
    ChatResponse,
    PipelineTimings,
    SourceChunk,
    StreamChunkEvent,
    TokenUsage,
)
from services.token_tracker import PipelineTokenLog

logger = logging.getLogger("docqa.router.chat")

router = APIRouter(prefix="/api", tags=["chat"])


async def _run_pipeline(request: Request, body: ChatRequest):
    """
    Core pipeline: retrieve → generate → guard → follow-ups.
    Returns all components needed for both blocking and streaming responses.
    """
    session_service = request.app.state.session_service
    retrieval_service = request.app.state.retrieval_service
    generation_service = request.app.state.generation_service
    hallucination_guard = request.app.state.hallucination_guard
    cache_service = request.app.state.cache_service
    token_tracker = request.app.state.token_tracker
    settings = request.app.state.settings

    # Validate session exists and has documents
    session = session_service.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found")

    if session.total_chunks == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents uploaded in this session. Upload files first.",
        )

    pipeline_start = time.perf_counter()
    log = PipelineTokenLog()

    # ── Check cache ───────────────────────────────────────────────────
    cached = await cache_service.get(body.session_id, body.query)
    if cached:
        return {**cached, "cached": True}

    # ── File scope resolution (chat = no file upload, but user may reference a file) ─
    from services.retrieval_service import resolve_file_scope
    session = session_service.get_session(body.session_id)
    session_file_names = [f.file_name for f in (session.files if session else [])]
    target_files, scope_mode = resolve_file_scope(
        uploaded_file_names=[],  # chat endpoint never has uploads
        query=body.query,
        session_file_names=session_file_names,
    )

    # ── Retrieve (with file scoping if user referenced a file) ─────────
    retrieval = await retrieval_service.retrieve(
        body.session_id, body.query,
        target_files=target_files,
        scope_mode=scope_mode,
    )
    log.record_step(
        "retrieval",
        embedding_tokens=retrieval.embedding_tokens,
        elapsed_ms=retrieval.retrieval_ms,
    )

    # Out-of-context check
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
            "pipeline_ms": {"retrieval_ms": retrieval.retrieval_ms, "total_ms": retrieval.retrieval_ms},
            "cached": False,
        }

    # Build context from retrieved chunks
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

    # ── Generate (file-aware) ────────────────────────────────────────
    history = session_service.build_history_messages(body.session_id)
    result = await generation_service.generate(
        body.query, context, history,
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

    # ── Hallucination guard ───────────────────────────────────────────
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

    # ── Follow-up questions ───────────────────────────────────────────
    follow_ups = await generation_service.generate_followups(
        context, answer, settings.min_followup_questions
    )

    # ── Track tokens ──────────────────────────────────────────────────
    total_ms = (time.perf_counter() - pipeline_start) * 1000
    cost = token_tracker.estimate_cost(log)

    # Store turn in session
    session_service.add_turn(body.session_id, "user", body.query)
    session_service.add_turn(
        body.session_id, "assistant", answer,
        groundedness=guard_result["groundedness"],
    )
    session_service.add_tokens(body.session_id, log.total_tokens)

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

    # Cache only non-hallucinated responses
    if not needs_clarification:
        await cache_service.set(body.session_id, body.query, response_data)

    return response_data


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest):
    """Blocking Q&A endpoint."""
    data = await _run_pipeline(request, body)

    return ChatResponse(
        session_id=body.session_id,
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


@router.post("/chat/stream")
async def chat_stream(request: Request, body: ChatRequest):
    """SSE streaming Q&A endpoint."""
    session_service = request.app.state.session_service
    retrieval_service = request.app.state.retrieval_service
    generation_service = request.app.state.generation_service
    hallucination_guard = request.app.state.hallucination_guard
    cache_service = request.app.state.cache_service
    token_tracker = request.app.state.token_tracker
    settings = request.app.state.settings

    # Validate session
    session = session_service.get_session(body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {body.session_id} not found")
    if session.total_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded. Upload files first.")

    async def event_generator():
        pipeline_start = time.perf_counter()
        log = PipelineTokenLog()

        # Check cache first
        cached = await cache_service.get(body.session_id, body.query)
        if cached:
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": cached["answer"]}),
            }
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "done",
                    "sources": cached.get("sources", []),
                    "follow_up_questions": cached.get("follow_up_questions", []),
                    "groundedness_score": cached.get("groundedness_score", 0),
                    "token_usage": cached.get("token_usage", {}),
                    "pipeline_ms": cached.get("pipeline_ms", {}),
                    "cached": True,
                }),
            }
            return

        # Retrieve
        retrieval = await retrieval_service.retrieve(body.session_id, body.query)
        log.record_step("retrieval", embedding_tokens=retrieval.embedding_tokens, elapsed_ms=retrieval.retrieval_ms)

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
                    "sources": [],
                    "follow_up_questions": [
                        "Could you rephrase your question?",
                        "Would you like to upload additional documents?",
                        "What section of the documents interests you?",
                    ],
                    "groundedness_score": 0.0,
                    "needs_clarification": True,
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

        # Send sources early
        yield {
            "event": "message",
            "data": json.dumps({"type": "sources", "sources": sources}),
        }

        # Stream LLM response
        history = session_service.build_history_messages(body.session_id)
        full_answer = ""
        llm_start = time.perf_counter()

        async for chunk_text in generation_service.generate_stream(body.query, context, history):
            full_answer += chunk_text
            yield {
                "event": "message",
                "data": json.dumps({"type": "chunk", "content": chunk_text}),
            }

        llm_ms = (time.perf_counter() - llm_start) * 1000

        # Hallucination guard on complete answer
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

        # Follow-ups
        follow_ups = await generation_service.generate_followups(
            context, full_answer, settings.min_followup_questions
        )

        total_ms = (time.perf_counter() - pipeline_start) * 1000

        # Store in session
        session_service.add_turn(body.session_id, "user", body.query)
        session_service.add_turn(
            body.session_id, "assistant", full_answer,
            groundedness=guard_result["groundedness"],
        )

        # Final metadata event
        yield {
            "event": "message",
            "data": json.dumps({
                "type": "done",
                "follow_up_questions": follow_ups,
                "groundedness_score": guard_result["groundedness"],
                "needs_clarification": not guard_result["passed"],
                "pipeline_ms": {
                    "retrieval_ms": round(retrieval.retrieval_ms, 1),
                    "llm_ms": round(llm_ms, 1),
                    "total_ms": round(total_ms, 1),
                },
            }),
        }

    return EventSourceResponse(event_generator())
