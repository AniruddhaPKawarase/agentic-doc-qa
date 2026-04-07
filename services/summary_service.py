"""
Document Summary Service — Generate structured summaries on upload.
──────────────────────────────────────────────────────────────────
Generates comprehensive summaries of uploaded documents using gpt-4o-mini.
Summaries are cached per session + file and used for SUMMARY_PLUS_RETRIEVAL
context strategy.
"""

import logging
import threading
from typing import Dict, List, Optional

import tiktoken
from openai import AsyncOpenAI

logger = logging.getLogger("docqa.summary_service")

_ENCODING = tiktoken.get_encoding("cl100k_base")

_SUMMARY_SYSTEM_PROMPT = (
    "You are a document analysis expert. Produce a comprehensive structured "
    "summary of the provided document.\n\n"
    "Include ALL of the following:\n"
    "1. DOCUMENT TYPE: What kind of document is this?\n"
    "2. MAIN SUBJECT: What is this document about?\n"
    "3. KEY SECTIONS: List every major section/topic with a brief description\n"
    "4. KEY DATA POINTS: Important numbers, specs, dates, names, requirements\n"
    "5. SCOPE: What does this document cover and NOT cover?\n"
    "6. RELATIONSHIPS: How do different sections relate to each other?\n\n"
    "Be exhaustive. Include every topic, requirement, and data point mentioned.\n"
    "Do not omit anything — completeness is more important than brevity."
)


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> tuple:
    """Truncate text to fit within max_tokens. Returns (truncated_text, was_truncated)."""
    tokens = _ENCODING.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    truncated = _ENCODING.decode(tokens[:max_tokens])
    return truncated, True


# ── Summary Generation Service ───────────────────────────────────────────


class SummaryService:
    """Generates structured document summaries via OpenAI API.

    Uses the secondary model (gpt-4o-mini) for cost efficiency.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_input_tokens: int = 100_000,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._max_input_tokens = max_input_tokens

    async def generate_summary(self, file_name: str, full_text: str) -> str:
        """Generate a structured summary of the document text.

        If text exceeds max_input_tokens, it is truncated with a note.
        On API error, returns a fallback string instead of raising.
        """
        token_count = _count_tokens(full_text)
        text_to_send, was_truncated = _truncate_to_tokens(
            full_text, self._max_input_tokens
        )

        user_content = f"Document: {file_name}\n\n{text_to_send}"
        if was_truncated:
            user_content += (
                f"\n\n[NOTE: Document was truncated from {token_count} tokens "
                f"to {self._max_input_tokens} tokens for summary generation.]"
            )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                temperature=0.1,
                max_tokens=2048,
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            summary = response.choices[0].message.content
            logger.info(
                "Generated summary for %s (%d doc tokens -> %d summary chars)",
                file_name,
                token_count,
                len(summary) if summary else 0,
            )
            return summary or ""
        except Exception:
            logger.warning(
                "Summary generation failed for %s (%d tokens)",
                file_name,
                token_count,
                exc_info=True,
            )
            return (
                f"Summary generation failed for {file_name}. "
                f"Document contains {token_count} tokens."
            )


# ── Summary Store ────────────────────────────────────────────────────────


class SummaryStore:
    """Thread-safe in-memory store for document summaries.

    Data layout: {session_id: {file_name: summary_text}}
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, str]] = {}
        self._lock = threading.Lock()

    def store(self, session_id: str, file_name: str, summary: str) -> None:
        """Store a summary for a session + file."""
        with self._lock:
            if session_id not in self._store:
                self._store[session_id] = {}
            self._store[session_id][file_name] = summary
        logger.info(
            "Stored summary for %s in session %s (%d chars)",
            file_name,
            session_id,
            len(summary),
        )

    def get(self, session_id: str, file_name: str) -> Optional[str]:
        """Retrieve a summary, or None if not found."""
        with self._lock:
            session_summaries = self._store.get(session_id, {})
            return session_summaries.get(file_name)

    def get_session_summaries(
        self,
        session_id: str,
        file_names: Optional[List[str]] = None,
    ) -> str:
        """Return combined summaries with file headers.

        If file_names is provided, only those files are included.
        Returns empty string for unknown sessions.
        """
        with self._lock:
            session_summaries = self._store.get(session_id, {})
            if not session_summaries:
                return ""
            items = (
                [
                    (name, session_summaries[name])
                    for name in file_names
                    if name in session_summaries
                ]
                if file_names is not None
                else list(session_summaries.items())
            )

        if not items:
            return ""

        sections = []
        for name, summary in items:
            sections.append(f"=== SUMMARY: {name} ===")
            sections.append(summary)
        return "\n".join(sections)

    def delete_session(self, session_id: str) -> None:
        """Remove all summaries for a session."""
        with self._lock:
            self._store.pop(session_id, None)
        logger.info("Deleted summary store for session %s", session_id)
