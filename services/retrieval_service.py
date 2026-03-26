"""
Retrieval Service — Query embedding → FAISS search → ranked chunks.
──────────────────────────────────────────────────────────────────────────────
Combines embedding + index services into a single retrieval call.

v2 (2026-03-26): File-aware retrieval with scope resolver.
When files are uploaded alongside a query, retrieval scopes to those files.
When user mentions a file name, retrieval detects and scopes to that file.
Follow-up queries without file context search all session files.
"""

import logging
import re
import time
from typing import List, Optional, Tuple

from services.embedding_service import EmbeddingService
from services.index_service import IndexService
from services.file_processor import Chunk
from config import Settings

logger = logging.getLogger("docqa.retrieval")


# ── File Scope Resolver ──────────────────────────────────────────────────────

def resolve_file_scope(
    uploaded_file_names: List[str],
    query: str,
    session_file_names: List[str],
) -> Tuple[List[str], str]:
    """
    Determine which files the retrieval should be scoped to.

    Args:
        uploaded_file_names: Files uploaded in THIS request (may be empty).
        query: The user's question text.
        session_file_names: All file names in the session.

    Returns:
        (target_files, scope_mode) where:
        - target_files: list of file names to filter by (empty = global search)
        - scope_mode: "current_upload" | "referenced_file" | "global"
    """
    # Case 1: Files were uploaded in this request → strict scope
    if uploaded_file_names:
        return uploaded_file_names, "current_upload"

    # Case 2: User mentions a file name in their query → scope to that file
    query_lower = query.lower()
    for fname in session_file_names:
        # Match by full name (case-insensitive)
        if fname.lower() in query_lower:
            return [fname], "referenced_file"
        # Match by name stem (without extension)
        stem = fname.rsplit(".", 1)[0].lower()
        # Only match stems that are meaningful (> 3 chars, not generic)
        if len(stem) > 3 and stem in query_lower:
            return [fname], "referenced_file"

    # Case 3: Try partial keyword matching for common construction doc names
    # e.g., "terminal units" → match "23 36 00-1 Terminal Units Product Data.pdf"
    for fname in session_file_names:
        stem = fname.rsplit(".", 1)[0].lower()
        # Extract meaningful words (3+ chars, no numbers-only)
        words = [w for w in re.split(r'[\s\-_.,]+', stem) if len(w) >= 3 and not w.isdigit()]
        # If 2+ meaningful words from the filename appear in the query, it's a match
        matches = sum(1 for w in words if w in query_lower)
        if len(words) >= 2 and matches >= 2:
            return [fname], "referenced_file"

    # Case 4: No files, no reference → search all
    return [], "global"


class RetrievalResult:
    """Container for retrieval output."""

    def __init__(
        self,
        chunks: List[Tuple[Chunk, float]],
        embedding_tokens: int,
        retrieval_ms: float,
        scope_mode: str = "global",
        target_files: Optional[List[str]] = None,
    ):
        self.chunks = chunks  # [(chunk, score), ...]
        self.embedding_tokens = embedding_tokens
        self.retrieval_ms = retrieval_ms
        self.scope_mode = scope_mode
        self.target_files = target_files or []

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    def build_context(self, max_tokens: int = 80000) -> str:
        """Build context string from retrieved chunks, respecting token budget."""
        context_parts = []
        token_count = 0

        for chunk, score in self.chunks:
            if token_count + chunk.token_count > max_tokens:
                break

            source_label = f"[{chunk.file_name}"
            if chunk.page_number:
                source_label += f", page {chunk.page_number}"
            if chunk.sheet_name:
                source_label += f", sheet '{chunk.sheet_name}'"
            source_label += f"] (relevance: {score:.2f})"

            context_parts.append(f"{source_label}\n{chunk.text}")
            token_count += chunk.token_count

        return "\n\n---\n\n".join(context_parts)


class RetrievalService:
    """Orchestrates query embedding + FAISS search with file-aware scoping."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        index_service: IndexService,
        settings: Settings,
    ):
        self.embedding = embedding_service
        self.index = index_service
        self.top_k = settings.retrieval_top_k
        self.score_threshold = settings.retrieval_score_threshold
        self.max_context_tokens = settings.max_context_tokens

    async def retrieve(
        self,
        session_id: str,
        query: str,
        target_files: Optional[List[str]] = None,
        scope_mode: str = "global",
    ) -> RetrievalResult:
        """
        Embed query → search FAISS → optionally filter by file → return ranked chunks.

        Args:
            session_id: Session to search.
            query: User's question.
            target_files: If provided, post-filter results to only these file names.
            scope_mode: "current_upload", "referenced_file", or "global".
        """
        start = time.perf_counter()

        # Embed query
        query_vector = await self.embedding.embed_query(query)
        embedding_tokens = max(1, len(query) // 4)

        # Over-fetch when filtering by file (we'll discard non-matching chunks)
        fetch_k = self.top_k * 3 if target_files else self.top_k

        # Search FAISS
        results = self.index.search(
            session_id=session_id,
            query_vector=query_vector,
            top_k=fetch_k,
            score_threshold=self.score_threshold,
        )

        # Post-retrieval file filter
        if target_files:
            target_set = {f.lower() for f in target_files}
            filtered = [
                (chunk, score) for chunk, score in results
                if chunk.file_name.lower() in target_set
            ]
            # Fallback: if file filter yields < 2 results, include all results
            # (safety net for edge cases where file name doesn't match exactly)
            if len(filtered) < 2 and len(results) > len(filtered):
                logger.info(
                    f"File filter too restrictive ({len(filtered)} results for {target_files}), "
                    f"falling back to global ({len(results)} results)"
                )
                results = results[:self.top_k]
            else:
                results = filtered[:self.top_k]
        else:
            results = results[:self.top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            f"Retrieved {len(results)} chunks for session {session_id} "
            f"(scope={scope_mode}, target_files={target_files}) in {elapsed_ms:.1f}ms"
        )

        return RetrievalResult(
            chunks=results,
            embedding_tokens=embedding_tokens,
            retrieval_ms=elapsed_ms,
            scope_mode=scope_mode,
            target_files=target_files or [],
        )
