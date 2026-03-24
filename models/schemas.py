"""
Request / Response schemas for the Document Q&A Agent.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class FileInfo(BaseModel):
    file_name: str
    file_type: str
    size_bytes: int
    chunk_count: int
    status: str  # "processed" | "failed" | "unsupported"
    error: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    files: List[FileInfo]
    total_chunks: int
    message: str


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Session with uploaded documents")
    query: str = Field(..., min_length=1, max_length=5000, description="User question")


class SourceChunk(BaseModel):
    file_name: str
    chunk_index: int
    page_number: Optional[int] = None
    sheet_name: Optional[str] = None
    score: float
    text_preview: str = Field(description="First 200 chars of the chunk")


class TokenUsage(BaseModel):
    embedding_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


class PipelineTimings(BaseModel):
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    guard_ms: float = 0.0
    total_ms: float = 0.0


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: List[SourceChunk] = []
    follow_up_questions: List[str] = []
    groundedness_score: float = 0.0
    needs_clarification: bool = False
    clarification_questions: List[str] = []
    token_usage: TokenUsage = TokenUsage()
    pipeline_ms: PipelineTimings = PipelineTimings()
    cached: bool = False


# ── Stream Event ──────────────────────────────────────────────────────────────

class StreamChunkEvent(BaseModel):
    """Single SSE chunk during streaming."""
    type: str = "chunk"  # "chunk" | "sources" | "metadata" | "error" | "done"
    content: Optional[str] = None
    sources: Optional[List[SourceChunk]] = None
    follow_up_questions: Optional[List[str]] = None
    groundedness_score: Optional[float] = None
    needs_clarification: Optional[bool] = None
    clarification_questions: Optional[List[str]] = None
    token_usage: Optional[TokenUsage] = None
    pipeline_ms: Optional[PipelineTimings] = None


# ── Converse (Unified Upload + Q&A) ──────────────────────────────────────────

class ConverseResponse(BaseModel):
    """Response from POST /api/converse and POST /api/converse/stream (final done event)."""
    session_id: str

    # Upload phase results
    files_processed: List[FileInfo] = []
    new_chunks_added: int = 0
    total_session_files: int = 0
    total_session_chunks: int = 0

    # Query phase results
    answer: str
    sources: List[SourceChunk] = []
    follow_up_questions: List[str] = []
    groundedness_score: float = 0.0
    needs_clarification: bool = False
    clarification_questions: List[str] = []
    token_usage: TokenUsage = TokenUsage()
    pipeline_ms: PipelineTimings = PipelineTimings()
    cached: bool = False


# ── Session ───────────────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    file_count: int
    total_chunks: int
    message_count: int


class SessionDetail(BaseModel):
    session_id: str
    created_at: str
    files: List[FileInfo]
    total_chunks: int
    message_count: int
    total_tokens_used: int
    history: List[Dict] = []


class SessionFileList(BaseModel):
    session_id: str
    files: List[FileInfo]
