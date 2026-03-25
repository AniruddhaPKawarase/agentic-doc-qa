"""
Session Service — Per-user session management with conversation history.
──────────────────────────────────────────────────────────────────────────────
Tracks: files uploaded, conversation turns, token totals, timestamps.
Sliding window history with anti-hallucination filtering.
"""

import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set

from pathlib import Path
from services.file_processor import ProcessedFile

logger = logging.getLogger("docqa.session")


@dataclass
class ConversationTurn:
    role: str  # "user" | "assistant"
    content: str
    groundedness: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Session:
    session_id: str
    created_at: str
    files: List[ProcessedFile] = field(default_factory=list)
    history: List[ConversationTurn] = field(default_factory=list)
    total_tokens_used: int = 0
    file_hashes: Set[str] = field(default_factory=set)

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_chunks(self) -> int:
        return sum(len(f.chunks) for f in self.files)

    @property
    def message_count(self) -> int:
        return len(self.history)


class SessionService:
    """Manage user sessions with conversation history."""

    def __init__(self, max_history_messages: int = 20):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()
        self.max_history = max_history_messages

    def create_session(self) -> Session:
        """Create a new session."""
        session_id = uuid.uuid4().hex[:16]
        session = Session(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        with self._lock:
            self._sessions[session_id] = session
        logger.info(f"Created session {session_id}")
        self._persist_to_s3(session)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID. Falls back to S3 if not in memory."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                return session
        # S3 fallback: try loading from S3 if not in memory
        session = self._load_from_s3(session_id)
        if session:
            with self._lock:
                self._sessions[session_id] = session
            logger.info(f"Session {session_id} restored from S3 (cache miss)")
        return session

    def get_or_create(self, session_id: Optional[str] = None) -> Session:
        """Get existing session or create new one."""
        if session_id:
            session = self.get_session(session_id)
            if session:
                return session
        return self.create_session()

    # ── File deduplication ────────────────────────────────────────────────────

    @staticmethod
    def compute_file_hash(content: bytes) -> str:
        """Return SHA-256 hex digest of raw file bytes."""
        return hashlib.sha256(content).hexdigest()

    def has_file_hash(self, session_id: str, sha256_hex: str) -> bool:
        """Return True if this file content was already indexed in the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            return session is not None and sha256_hex in session.file_hashes

    def add_file_hash(self, session_id: str, sha256_hex: str) -> None:
        """Record a file hash after successful indexing."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.file_hashes.add(sha256_hex)

    def add_file(self, session_id: str, processed_file: ProcessedFile) -> None:
        """Register a processed file with the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.files.append(processed_file)

    def add_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        groundedness: Optional[float] = None,
    ) -> None:
        """Add a conversation turn to the session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return

            turn = ConversationTurn(
                role=role,
                content=content,
                groundedness=groundedness,
            )
            session.history.append(turn)

            # Trim to sliding window
            if len(session.history) > self.max_history * 2:
                session.history = session.history[-self.max_history * 2:]
            self._persist_to_s3(session)

    def add_tokens(self, session_id: str, tokens: int) -> None:
        """Accumulate token count and persist to S3."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.total_tokens_used += tokens
                self._persist_to_s3(session)

    def build_history_messages(self, session_id: str) -> List[Dict]:
        """
        Build message list for LLM context.
        Filters out assistant turns with low groundedness to prevent
        hallucination cascading.
        """
        session = self.get_session(session_id)
        if not session:
            return []

        messages = []
        for turn in session.history[-self.max_history:]:
            # Skip low-groundedness assistant messages
            if (
                turn.role == "assistant"
                and turn.groundedness is not None
                and turn.groundedness < 0.50
            ):
                continue

            messages.append({
                "role": turn.role,
                "content": turn.content,
            })

        return messages

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Deleted session {session_id}")
                # --- S3 DELETE (Phase 7) ---
                self._delete_from_s3(session_id)
                # --- END S3 DELETE ---
                return True
        return False

    def list_sessions(self) -> List[Dict]:
        """List all sessions with summary info."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "created_at": s.created_at,
                    "file_count": s.file_count,
                    "total_chunks": s.total_chunks,
                    "message_count": s.message_count,
                }
                for s in self._sessions.values()
            ]

    # ── S3 Persistence (Phase 7) ──────────────────────────────────────────

    def _load_from_s3(self, session_id: str) -> Optional[Session]:
        """Load a single session from S3 by session_id."""
        import os
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return None
        try:
            import sys, json
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from s3_utils.operations import download_bytes
            from s3_utils.helpers import docqa_session_meta_key
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "document-qa-agent")
            s3_key = docqa_session_meta_key(s3_prefix, session_id)
            raw = download_bytes(s3_key)
            if raw:
                data = json.loads(raw.decode("utf-8"))
                session = Session(
                    session_id=data["session_id"],
                    created_at=data.get("created_at", ""),
                    total_tokens_used=data.get("total_tokens_used", 0),
                    file_hashes=set(data.get("file_hashes", [])),
                )
                session.history = [
                    ConversationTurn(
                        role=t["role"],
                        content=t["content"],
                        groundedness=t.get("groundedness"),
                        timestamp=t.get("timestamp", ""),
                    )
                    for t in data.get("history", [])
                ]
                return session
        except Exception as e:
            logger.warning(f"S3 session load failed for {session_id}: {e}")
        return None

    def _persist_to_s3(self, session: Session) -> None:
        """Write-behind: serialize session to JSON and upload to S3."""
        import os
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return
        try:
            import sys, json
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from s3_utils.operations import upload_bytes
            from s3_utils.helpers import docqa_session_meta_key
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "document-qa-agent")
            s3_key = docqa_session_meta_key(s3_prefix, session.session_id)
            session_data = {
                "session_id": session.session_id,
                "created_at": session.created_at,
                "file_count": session.file_count,
                "total_chunks": session.total_chunks,
                "total_tokens_used": session.total_tokens_used,
                "history": [
                    {"role": t.role, "content": t.content,
                     "groundedness": t.groundedness, "timestamp": t.timestamp}
                    for t in session.history
                ],
                "file_hashes": list(session.file_hashes),
            }
            upload_bytes(json.dumps(session_data).encode("utf-8"), s3_key)
        except Exception as e:
            logger.warning(f"S3 session persist failed: {e}")

    def _delete_from_s3(self, session_id: str) -> None:
        """Delete session data from S3."""
        import os
        if os.getenv("STORAGE_BACKEND", "local") != "s3":
            return
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from s3_utils.operations import delete_prefix
            s3_prefix = os.getenv("S3_AGENT_PREFIX", "document-qa-agent")
            delete_prefix(f"{s3_prefix}/session_data/{session_id}/")
        except Exception as e:
            logger.warning(f"S3 session delete failed: {e}")
