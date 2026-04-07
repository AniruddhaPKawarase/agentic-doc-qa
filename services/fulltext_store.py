"""
Full Text Store — In-memory store for complete document texts.
──────────────────────────────────────────────────────────────────────────────
Stores full document text keyed by (session_id, file_name) for the v2
full-context pipeline.  Thread-safe via threading.Lock.
"""

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger("docqa.fulltext_store")

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


@dataclass(frozen=True)
class StoredDocument:
    """Immutable representation of a stored document."""

    file_name: str
    full_text: str
    token_count: int
    page_texts: tuple  # tuple for immutability (frozen=True)


class FullTextStore:
    """Thread-safe in-memory store for complete document texts.

    Data layout: {session_id: {file_name: StoredDocument}}
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, StoredDocument]] = {}
        self._lock = threading.Lock()

    def store(
        self,
        session_id: str,
        file_name: str,
        full_text: str,
        page_texts: List[str],
    ) -> StoredDocument:
        """Store a document's full text for a session. Returns the stored document."""
        token_count = _count_tokens(full_text)
        doc = StoredDocument(
            file_name=file_name,
            full_text=full_text,
            token_count=token_count,
            page_texts=tuple(page_texts),
        )
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][file_name] = doc
        logger.info(
            "Stored document %s for session %s (%d tokens)",
            file_name,
            session_id,
            token_count,
        )
        return doc

    def get_document(
        self, session_id: str, file_name: str
    ) -> Optional[StoredDocument]:
        """Retrieve a single stored document, or None if not found."""
        with self._lock:
            session_docs = self._store.get(session_id, {})
            return session_docs.get(file_name)

    def get_session_text(
        self,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> str:
        """Return concatenated text for a session with file headers.

        If file_names is provided, only those files are included.
        Returns empty string for unknown sessions.
        """
        with self._lock:
            session_docs = self._store.get(session_id, {})
            if not session_docs:
                return ""
            docs = (
                [
                    session_docs[name]
                    for name in file_names
                    if name in session_docs
                ]
                if file_names is not None
                else list(session_docs.values())
            )

        if not docs:
            return ""

        sections = []
        for doc in docs:
            sections.append(f"=== FILE: {doc.file_name} ===")
            sections.append(doc.full_text)
        return "\n".join(sections)

    def get_session_token_count(
        self,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> int:
        """Return total token count for a session (or subset of files)."""
        with self._lock:
            session_docs = self._store.get(session_id, {})
            if not session_docs:
                return 0
            docs = (
                [
                    session_docs[name]
                    for name in file_names
                    if name in session_docs
                ]
                if file_names is not None
                else list(session_docs.values())
            )
        return sum(doc.token_count for doc in docs)

    def get_file_names(self, session_id: str) -> List[str]:
        """Return list of file names stored for a session."""
        with self._lock:
            session_docs = self._store.get(session_id, {})
            return list(session_docs.keys())

    def delete_session(self, session_id: str) -> None:
        """Remove all stored documents for a session."""
        with self._lock:
            self._store.pop(session_id, None)
        logger.info("Deleted fulltext store for session %s", session_id)
