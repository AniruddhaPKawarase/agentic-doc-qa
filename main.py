"""
Document Q&A Agent — FastAPI Application Entry Point.
──────────────────────────────────────────────────────────────────────────────
Port: 8006 | Prefix: /docqa/

Upload documents and ask questions answered strictly from uploaded content.
Per-session FAISS indices for data isolation.

Endpoints:
    POST /api/upload           — Upload files to a session
    POST /api/chat             — Blocking Q&A
    POST /api/chat/stream      — SSE streaming Q&A
    GET  /api/sessions         — List sessions
    GET  /api/sessions/{id}    — Session detail
    DELETE /api/sessions/{id}  — Delete session
    GET  /health               — Health check
"""

import os
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from services.file_processor import FileProcessor
from services.embedding_service import EmbeddingService
from services.index_service import IndexService
from services.retrieval_service import RetrievalService
from services.generation_service import GenerationService
from services.session_service import SessionService
from services.hallucination_guard import HallucinationGuard
from services.token_tracker import TokenTracker
from services.cache_service import CacheService
from routers import upload, chat, sessions, converse

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("docqa")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all services on startup, cleanup on shutdown."""
    settings = get_settings()

    logger.info("Initializing Document Q&A Agent services...")

    # Create services
    file_processor = FileProcessor(
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )
    embedding_service = EmbeddingService(settings)
    index_service = IndexService()
    retrieval_service = RetrievalService(embedding_service, index_service, settings)
    generation_service = GenerationService(settings)
    session_service = SessionService(max_history_messages=settings.max_history_messages)
    hallucination_guard = HallucinationGuard(
        threshold=settings.groundedness_threshold,
        enabled=settings.hallucination_guard_enabled,
        general_threshold=settings.guard_general_threshold,
        specific_threshold=settings.guard_specific_threshold,
        comparison_threshold=settings.guard_general_threshold + 0.05,
        marginal_high=settings.guard_marginal_high,
    )
    token_tracker = TokenTracker(
        chat_model=settings.openai_chat_model,
        embedding_model=settings.openai_embedding_model,
    )
    cache_service = CacheService(
        l1_maxsize=settings.cache_l1_maxsize,
        l1_ttl=settings.cache_l1_ttl,
        redis_url=settings.redis_url,
    )

    # ── v2 Services (Hybrid Context Pipeline) ────────────────────────────
    from services.fulltext_store import FullTextStore
    from services.context_manager import ContextManager
    from services.summary_service import SummaryService, SummaryStore
    from services.bm25_service import BM25Service
    from services.monitoring_service import MonitoringService

    fulltext_store = FullTextStore()
    context_manager = ContextManager(
        fulltext_store=fulltext_store,
        full_context_threshold=settings.full_context_threshold,
        summary_threshold=settings.summary_threshold,
        primary_model=settings.primary_model,
    )
    summary_service = SummaryService(
        api_key=settings.openai_api_key,
        model=settings.secondary_model,
    )
    summary_store = SummaryStore()
    bm25_service = BM25Service()
    monitoring_service = MonitoringService(
        latency_warning_ms=settings.alert_latency_warning_ms,
        latency_critical_ms=settings.alert_latency_critical_ms,
        cost_warning_usd=settings.alert_cost_warning_usd,
        cost_critical_usd=settings.alert_cost_critical_usd,
        groundedness_warning=settings.alert_groundedness_warning,
        groundedness_critical=settings.alert_groundedness_critical,
    )

    # Attach to app.state for dependency injection
    app.state.settings = settings
    app.state.file_processor = file_processor
    app.state.embedding_service = embedding_service
    app.state.index_service = index_service
    app.state.retrieval_service = retrieval_service
    app.state.generation_service = generation_service
    app.state.session_service = session_service
    app.state.hallucination_guard = hallucination_guard
    app.state.token_tracker = token_tracker
    app.state.cache_service = cache_service

    # Attach v2 services to app.state
    app.state.fulltext_store = fulltext_store
    app.state.context_manager = context_manager
    app.state.summary_service = summary_service
    app.state.summary_store = summary_store
    app.state.bm25_service = bm25_service
    app.state.monitoring_service = monitoring_service

    logger.info(
        f"Document Q&A Agent ready | "
        f"Pipeline: {settings.pipeline_version} | "
        f"Model: {settings.openai_chat_model} | "
        f"Primary: {settings.primary_model} | "
        f"Embedding: {settings.openai_embedding_model} | "
        f"Max file: {settings.max_file_size_mb}MB"
    )

    # Start background session cleanup task
    async def _cleanup_expired_sessions():
        """Periodically remove expired sessions to prevent OOM from FAISS index accumulation."""
        interval = settings.session_cleanup_interval_min * 60
        ttl = timedelta(hours=settings.session_ttl_hours)
        while True:
            await asyncio.sleep(interval)
            try:
                cutoff = datetime.now(timezone.utc) - ttl
                expired = []
                for sid, session in list(session_service._sessions.items()):
                    created = datetime.fromisoformat(session.created_at)
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    if created < cutoff:
                        expired.append(sid)
                for sid in expired:
                    session_service.delete_session(sid)
                    index_service.delete_session(sid)
                if expired:
                    logger.info(f"Session cleanup: removed {len(expired)} expired sessions")
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")

    cleanup_task = asyncio.create_task(_cleanup_expired_sessions())

    yield

    # Cleanup
    cleanup_task.cancel()
    logger.info("Shutting down Document Q&A Agent...")
    await cache_service.close()


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="VCS Document Q&A Agent",
    description="Upload documents and ask questions — RAG with per-session FAISS indices.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — configurable via CORS_ALLOWED_ORIGINS env var
_cors_origins_raw = os.getenv("CORS_ALLOWED_ORIGINS", "https://ai.ifieldsmart.com,https://ai5.ifieldsmart.com,http://localhost:3000,http://localhost:8501")
_cors_origins = ["*"] if _cors_origins_raw == "*" else [o.strip() for o in _cors_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(converse.router)   # unified upload+query (UI-friendly)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(sessions.router)


# ── Root + Health ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Service overview."""
    return {
        "name": "VCS Document Q&A Agent",
        "version": "1.0.0",
        "description": "Upload documents and ask questions based on their content",
        "endpoints": {
            "converse": "POST /api/converse",
            "converse_stream": "POST /api/converse/stream",
            "upload": "POST /api/upload",
            "chat": "POST /api/chat",
            "stream": "POST /api/chat/stream",
            "sessions": "GET /api/sessions",
            "session_detail": "GET /api/sessions/{id}",
            "delete_session": "DELETE /api/sessions/{id}",
            "session_files": "GET /api/sessions/{id}/files",
            "health": "GET /health",
        },
    }


@app.get("/health")
async def health():
    """Health check."""
    settings = app.state.settings
    index_service = app.state.index_service
    session_service = app.state.session_service

    sessions = session_service.list_sessions()
    indices = index_service.list_sessions()

    return {
        "status": "ok",
        "model": settings.openai_chat_model,
        "embedding_model": settings.openai_embedding_model,
        "active_sessions": len(sessions),
        "total_indexed_vectors": sum(indices.values()),
        "max_file_size_mb": settings.max_file_size_mb,
    }


# ── Metrics ──────────────────────────────────────────────────────────────────


@app.get("/api/metrics/summary")
async def metrics_summary():
    """Dashboard metrics summary."""
    monitoring = app.state.monitoring_service
    return {
        "1h": monitoring.get_summary(hours=1.0),
        "24h": monitoring.get_summary(hours=24.0),
        "7d": monitoring.get_summary(hours=168.0),
    }


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("DOCQA_PORT", "8006"))
    host = os.getenv("DOCQA_HOST", "0.0.0.0")

    logger.info(f"Starting Document Q&A Agent on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
