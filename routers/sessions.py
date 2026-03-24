"""
Sessions Router — Session CRUD endpoints.
──────────────────────────────────────────────────────────────────────────────
GET    /api/sessions           — List all sessions
GET    /api/sessions/{id}      — Session detail
DELETE /api/sessions/{id}      — Delete session
GET    /api/sessions/{id}/files — List files in session
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from models.schemas import FileInfo, SessionDetail, SessionFileList, SessionInfo

logger = logging.getLogger("docqa.router.sessions")

router = APIRouter(prefix="/api", tags=["sessions"])


@router.get("/sessions")
async def list_sessions(request: Request):
    """List all active sessions."""
    session_service = request.app.state.session_service
    sessions = session_service.list_sessions()
    return {"sessions": sessions, "total": len(sessions)}


@router.get("/sessions/{session_id}")
async def get_session(request: Request, session_id: str):
    """Get session detail including files, history, and token totals."""
    session_service = request.app.state.session_service
    index_service = request.app.state.index_service

    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    files = [
        FileInfo(
            file_name=f.file_name,
            file_type=f.file_type,
            size_bytes=f.size_bytes,
            chunk_count=len(f.chunks),
            status=f.status,
            error=f.error,
        )
        for f in session.files
    ]

    history = [
        {
            "role": turn.role,
            "content": turn.content[:500],  # Truncate for listing
            "groundedness": turn.groundedness,
            "timestamp": turn.timestamp,
        }
        for turn in session.history
    ]

    return SessionDetail(
        session_id=session.session_id,
        created_at=session.created_at,
        files=files,
        total_chunks=index_service.get_chunk_count(session_id),
        message_count=session.message_count,
        total_tokens_used=session.total_tokens_used,
        history=history,
    )


@router.delete("/sessions/{session_id}")
async def delete_session(request: Request, session_id: str):
    """Delete a session and its FAISS index."""
    session_service = request.app.state.session_service
    index_service = request.app.state.index_service

    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    session_service.delete_session(session_id)
    index_service.delete_session(session_id)

    return {"session_id": session_id, "deleted": True}


@router.get("/sessions/{session_id}/files")
async def list_session_files(request: Request, session_id: str):
    """List files in a session with processing status."""
    session_service = request.app.state.session_service

    session = session_service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    files = [
        FileInfo(
            file_name=f.file_name,
            file_type=f.file_type,
            size_bytes=f.size_bytes,
            chunk_count=len(f.chunks),
            status=f.status,
            error=f.error,
        )
        for f in session.files
    ]

    return SessionFileList(session_id=session_id, files=files)
