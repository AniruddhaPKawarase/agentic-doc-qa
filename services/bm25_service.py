"""
BM25 Keyword Search Service -- Per-session BM25 index for hybrid retrieval.
--------------------------------------------------------------------------
Adds keyword-based retrieval alongside FAISS semantic search.
Hybrid scoring: final = alpha * semantic + (1-alpha) * bm25_normalized
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from services.file_processor import Chunk


# -- Stopwords ----------------------------------------------------------------

_STOPWORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "was", "one", "our", "has", "have", "been", "some", "them", "than",
    "its", "over", "such", "that", "this", "with", "will", "each", "from",
    "what", "when", "which", "their", "there", "about", "would", "these",
    "other", "into", "more", "also", "could", "does", "just", "like",
    "they", "very", "your", "most", "only", "where", "should", "being",
})

_PUNCT_RE = re.compile(r"[^\w\s]")


# -- Tokenizer ----------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, filter stopwords and short words."""
    text = text.lower()
    text = _PUNCT_RE.sub(" ", text)
    words = text.split()
    return [w for w in words if len(w) >= 3 and w not in _STOPWORDS]


# -- Session index data -------------------------------------------------------

class _SessionIndex:
    """Immutable-style container for per-session BM25 state."""

    __slots__ = ("bm25", "tokenized_corpus", "chunk_indices")

    def __init__(
        self,
        bm25: BM25Okapi,
        tokenized_corpus: List[List[str]],
        chunk_indices: List[int],
    ) -> None:
        self.bm25 = bm25
        self.tokenized_corpus = tokenized_corpus
        self.chunk_indices = chunk_indices


# -- BM25 Service -------------------------------------------------------------

class BM25Service:
    """Per-session BM25 keyword search index."""

    def __init__(self) -> None:
        self._sessions: Dict[str, _SessionIndex] = {}
        self._lock = threading.Lock()

    # -- Public API -----------------------------------------------------------

    def index_chunks(self, session_id: str, chunks: List[Chunk]) -> None:
        """Build (or append to) a BM25 index for *session_id*.

        If the session already has an index the new chunks are appended
        and the BM25 model is rebuilt over the combined corpus.
        """
        new_tokenized = [tokenize(c.text) for c in chunks]
        new_indices = [c.chunk_index for c in chunks]

        with self._lock:
            existing = self._sessions.get(session_id)

            if existing is not None:
                combined_corpus = existing.tokenized_corpus + new_tokenized
                combined_indices = existing.chunk_indices + new_indices
            else:
                combined_corpus = new_tokenized
                combined_indices = new_indices

            bm25 = BM25Okapi(combined_corpus)

            self._sessions[session_id] = _SessionIndex(
                bm25=bm25,
                tokenized_corpus=combined_corpus,
                chunk_indices=combined_indices,
            )

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 30,
    ) -> List[Tuple[int, float]]:
        """Return up to *top_k* ``(chunk_index, normalized_score)`` pairs.

        Scores are normalized so the top result equals 1.0.
        Returns an empty list when the session has no index or all scores
        are zero.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []

        raw_scores: List[float] = session.bm25.get_scores(tokenized_query).tolist()

        max_score = max(raw_scores) if raw_scores else 0.0
        if max_score == 0.0:
            return []

        scored = [
            (session.chunk_indices[i], score / max_score)
            for i, score in enumerate(raw_scores)
            if score > 0.0
        ]

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]

    def delete_session(self, session_id: str) -> None:
        """Remove all BM25 data for *session_id*."""
        with self._lock:
            self._sessions.pop(session_id, None)
